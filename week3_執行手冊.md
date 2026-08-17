# Week 3 執行手冊：Gemma 3 + Shadow-FT + LoRA 參數掃描（Colab 免費版）

> **Week 2 的結論是「微調失敗了」，Week 3 要回答的是「為什麼，以及怎麼修」。**
>
> 承接：`week2_執行手冊.md` / `week2_執行總結.md` / `ultrascale-proposal_1.pptx`
> 配套文件：
> - `week3_主管提問回覆.md` —— 主管四個提問的逐條回覆（先看這份）
> - `moe_routing_分析.md` —— MoE 路由專篇（Week 2 E3/E4/E5 的深入版 + 實作提案）
> - `notebooks/week3_colab.ipynb` —— 可以直接上傳 Colab 跑的完整流程

---

## Step 0：Week 2 交接的五個關鍵事實

**這五件事是 Week 3 實驗設計的前提，開跑前先讀過。**都已經反映在 `week2_執行總結.md`、`notes/`、`configs/` 與 `scripts/` 裡，這裡只是集中列出。

### 0.1 LoRA 的 `scale` 是直接乘數，我們的強度是常規的 10 倍

翻 `.venv/lib/python3.11/site-packages/mlx_lm/tuner/lora.py`：

```python
def __call__(self, x):
    y = self.linear(x)
    z = (self.dropout(x) @ self.lora_a) @ self.lora_b
    return y + (self.scale * z).astype(x.dtype)     # ← scale 直接乘，沒有除以 rank
```

整個 `mlx_lm/tuner/` 裡 grep 不到 `alpha`。所以：

| | Week 2 用的 | 業界常規 |
|---|---:|---:|
| 有效 scaling factor | **20.0** | 2.0 |
| 等效 PEFT `lora_alpha`（r=16） | **320** | 32 |

`scale: 20.0` 是 mlx-lm 的預設值，我從頭到尾沒動過它。

### 0.2 `k_proj` / `v_proj` 一層都沒訓練到

直接解析 `out/lora-e4b/adapters.safetensors` 的 header：

```
64 個 tensor，全部是 layers.{26..41}.self_attn.{q_proj,o_proj}.{lora_a,lora_b}
可訓練參數 2,555,904（2.56M）；設定檔對 16 層的估計是 3.5M
```

E4B 的 `num_kv_shared_layers = 18` → layer 24–41 共用前面算好的 K/V，**這些層沒有 `k_proj`/`v_proj` 模組**。而我們掛的是最後 16 層（26–41），完全落在裡面。mlx-lm 對找不到的 key 靜默略過。

設定檔裡原本的註解「k/v 的 adapter 只會作用在前 24 層」**是錯的，正確答案是「一層都沒有」**。

### 0.3 E4 的訊噪比是 6.6×

plug-in KL 的雜訊底線不能用解析式 `(K−1)/(2N) = 0.0306` —— 那是 N ≫ K 的漸近展開，本例 N/K ≈ 16 且 40.8% 的專家幾乎沒被選到，條件不成立，會低估 3.5 倍。用參數化 bootstrap 實測底線是 0.1063（`scripts/analyze_router_lang.py`）。

中英路由確實不同：平滑後的 KL 與 JSD 都是 **29/30 層 p < 0.01**（未平滑的 plug-in 是 26/30）。詳見 `moe_routing_分析.md` §3.4。

### 0.4 準確率要同時看科目平均與逐題平均

twinkle-eval 的 `average_accuracy` 是對三個科目取平均，但三科題數是 768 / 139 / 129。改成逐題加權：

| | 科目平均（macro） | 逐題平均（micro） |
|---|---:|---:|
| base | 53.98% | **59.75%** |
| tuned | 36.57% | **38.22%** |
| **差距** | **−17.4 pt** | **−21.5 pt** |

**逐題看跌幅是 21.5 pt。**另外「知識面只掉 1.7 pt」是在 tuned 有正確格式的那 630 題上比的，而 base 在那 630 題得 64.6%、在崩掉的 406 題只得 52.2% —— **那 630 題是比較簡單的一群**，所以 −1.7 pt 不能直接推論到全體。

### 0.5 MLX 4-bit 的等效位元數是 4.50

量 `out/gemma4-e4b-tw/model.safetensors`：weights 是 U32 的 4-bit，scale 和 bias **都是 bf16、每 64 個元素一組** → 4 + (2+2)×8/64 = **4.50 bit/參數**（全檔實測 4.501）。4.25 是 group_size 128 的值。

用 4.50 之後，權重預測對兩個模型的誤差都是 **−0.0%**（E4B 3.91 vs 3.91、26B 13.22 vs 13.23）。**同樣叫「4-bit」，group_size 不同就差 5.6%** —— Week 3 對 bitsandbytes 的 nf4 要重新量一次，不要沿用這個數字。

### 驗收

- [ ] `week3_主管提問回覆.md` 已讀過一遍
- [ ] 理解 0.1（LoRA 強度）和 0.4（macro/micro）為什麼是 Step 5 與 Step 7 的設計前提

---

## Step 1：本機收尾（不吃 Colab 配額，可以和 Colab 並行）

**要做什麼**：把 Week 2 已經存下來、但還沒分析的東西榨乾。這些全部在 Mac 上跑，**不消耗 Colab 的 GPU 配額**。

### 1.1 五個 checkpoint 的格式保留率曲線

`out/lora-e4b/` 裡有 `0000200 / 0000400 / 0000600 / 0000800 / 0001000` 五個 adapter（`save_every: 200`）。**不用重訓**，各跑一次 smoke 評測就能畫出「訓練步數 vs 格式保留率」曲線。

**⚠️ `mlx_lm.fuse` 沒有 `--adapter-file` 這個參數。**它只吃 `--model / --save-path / --adapter-path / --upload-repo / --dequantize / --export-gguf / --gguf-path`，而 `tuner/utils.py` 的 `load_adapters()` 把檔名寫死成 `adapter_path / "adapters.safetensors"`。argparse 遇到不認得的參數會直接報錯。**所以要先把 checkpoint 複製成 `adapters.safetensors`**：

```bash
for step in 0000200 0000400 0000600 0000800 0001000; do
  mkdir -p out/ckpt-${step}
  cp out/lora-e4b/adapter_config.json           out/ckpt-${step}/
  cp out/lora-e4b/${step}_adapters.safetensors  out/ckpt-${step}/adapters.safetensors

  python -m mlx_lm.fuse \
      --model mlx-community/gemma-4-e4b-it-4bit \
      --adapter-path out/ckpt-${step} \
      --save-path    out/fused-${step}

  sed -i '' "s|name: .*|name: \"out/fused-${step}\"|" configs/eval_gemma4_e4b_tuned.yaml
  bash scripts/run_eval.sh tuned
done
```

**磁碟**：每個融合模型 4.0 GB × 5 = 20 GB。跑完一個就 `rm -rf out/fused-${step}`，不要五個一起放著。

**注意**：這條曲線的橫軸是**訓練步數**，Colab 上 Stage A 的橫軸是 **LoRA 強度**。兩條是不同的軸，都要。

**驗收**：無法解析率應該隨步數單調上升。如果 200 步就已經 40%，代表崩潰發生得很早，那 Stage A 的 200 步設定要往下調。

### 1.2 重跑路由二次分析

```bash
python scripts/analyze_router_lang.py
```

**驗收**：終端輸出的訊噪比 6.6× / 7.1× / 4.4×，`reports/router_lang_analysis.json` 產生。

### 1.3 補上 `notes/ch05.md`

Week 3 用不到管線平行，但 ch05 是 Week 1 就欠的。有空檔時補。

---

## Step 2：Colab 環境

**要做什麼**：拿到 T4、裝好套件、掛好 Drive。

### 2.1 先認清免費版的限制

| 限制 | 值 | 應對 |
|---|---|---|
| GPU 型號 | T4 16GB（不保證拿得到） | 拿不到就等，不要改用 CPU 跑訓練 |
| **沒有 bf16** | T4 是 Turing (cc 7.5) | **這是主線改用 Unsloth 的主要理由**，見 Step 4（該假設本身也會在 Step 4 被驗證） |
| 單次 session | 12 小時 | 所有產出寫進 Drive，實驗設計成可斷點續跑 |
| 每週配額 | 約 15–30 GPU 小時（會浮動、不公布） | 本手冊全跑約 8–10 小時 |
| 閒置逾時 | 90 分鐘無互動就斷 | 掛著跑時每小時回去點一下 |

### 2.2 操作

1. 上傳 `notebooks/week3_colab.ipynb` 到 Colab
2. 執行階段 → 變更執行階段類型 → **T4 GPU**
3. 依序跑 §1.1 → §1.4

**HuggingFace 授權**：notebook §1.4 預設用 `unsloth/gemma-3-4b-it` 和 `unsloth/gemma-3-4b-pt` 鏡像 —— **權重相同、不需要接受授權條款**，這是刻意的預設值，讓沒有 token 的人也能直接跑。

想改用 `google/` 官方 repo 的話，`google/gemma-3-4b-it` 和 `google/gemma-3-4b-pt` 是**兩個分開的 repo，兩個都要接受條款**，然後在 Colab 左側「🔑 密鑰」加一個 `HF_TOKEN`，並把 §1.4 的 `USE_OFFICIAL_REPO` 勾選成 `True`（沒有 token 卻打開會直接報錯，不會安靜地退回鏡像）。

### 驗收

- [ ] `nvidia-smi` 顯示 Tesla T4，15.x GiB
- [ ] `torch.cuda.is_bf16_supported()` → `False`（這是預期的，不是錯誤）
- [ ] §1.2b 的九個套件全部 `ok`
- [ ] Drive 掛載成功，`ultrascale-lab-week3/` 下有六個子目錄

---

## Step 3：資料

**要做什麼**：把 Week 2 的同一份 8,000 筆抽樣，換成 Gemma 3 的模板重新渲染。

### 3.1 一個必須先做的決定

Gemma 4 有 thinking channel，Week 2 把 CoT 放進 `<|channel>thought`。**Gemma 3 沒有這個東西。**所以 `think` 欄位要重新決定怎麼處理：

| 版本 | 做法 | 用在哪 |
|---|---|---|
| **`inline`** | `<think>…</think>` 內嵌在 model turn 裡，接著才是回答 | **Stage A/B/C、Shadow-FT 全部用這個**（最接近 Week 2，才能比較） |
| `drop` | 丟掉 think，只留最終回答 | 備用對照組 |
| `inline_mixed` | `inline` + 混入 5% 的 `\box{X}` 樣本 | Week 2 修法 B 的驗證，Step 5 的加分項 |

notebook §2.1、§2.2 會把三份都產生出來。

### 3.2 兩個容易錯的地方

**（a）雙 BOS。**`apply_chat_template` 會在最前面加 `<bos>`，而 SFTTrainer tokenize 時預設 `add_special_tokens=True` 又會再加一個。訓練前綴和推論前綴不一致，會安靜地劣化效果。notebook 的 `render()` 已經把它剝掉。

**（b）Gemma 3 支不支援 system role？**不要假設，notebook §2.1 會實測 `supports_system()` 並設 `HAS_SYSTEM`。不支援就把 system 折進第一個 user turn —— **訓練和評測必須用同一套規則**，這是 Week 1「`reasoning_effort` 要同檔位」的同一條原則。

### 3.3 格式樣本不能有資料洩漏

`inline_mixed` 的 `\box{}` 樣本**只能從 TMMLU+ 的非評測科目抽**。評測用的三科（台灣地理、台語、三民主義）一題都不能碰。notebook §2.2 最後有一段 assert 在檢查這件事。

### 驗收

- [ ] `data/` 下有 6 個 jsonl（train/valid × inline/drop/inline_mixed）
- [ ] `train_inline.jsonl` 是 7,600 筆左右（和 Week 2 同一批抽樣）
- [ ] 印出來的樣本裡看得到 `<start_of_turn>user` / `<start_of_turn>model`，且**開頭沒有 `<bos>`**
- [ ] 資料洩漏檢查 assert 通過

---

## Step 4：框架對照（答主管 Q1）

**要做什麼**：把「有沒有記憶體優化」從一句形容詞變成一張表。

### 4.1 為什麼一定要做這個對照

主管問的是「現在用的是純 transformer 還是有記憶體優化的框架」。誠實的答案是：**Week 2 用的是 MLX，第三種東西**。它有量化和梯度檢查點，但沒有 fused cross-entropy、也沒有 Flash Attention 開關——而這兩個缺口正好解釋了 Week 2 兩個對不上的數字：

- logits 3.00 GiB 整張落地 → 把梯度檢查點的效益從 −91%（活化）稀釋成 −30.5%（峰值）
- 假設 P2 做不出對照組

Week 3 換到 CUDA，這兩件事都可以量了。

### 4.2 先寫預測，再跑

**這一步很重要，先寫在紙上：**

| 組別 | 預測 | 理由 |
|---|---|---|
| HF + fp16 | **會 NaN** | Gemma 3 layernorm 後 activation ~800,000 > fp16 上限 65,504 |
| HF + fp32 | 能跑，但**慢 3–5 倍** | T4 fp32 8.1 TFLOPS vs fp16 65 TFLOPS |
| Unsloth | 能跑，峰值記憶體**明顯較低** | fused CE 不落地完整 logits（seq1024×bs2 是 3.00 GiB） |
| **三者的 loss 曲線** | **應該接近** | 這是最關鍵的對帳 |

最後一列是重點：**優化框架的正當性建立在數值等價上。**如果 Unsloth 的 loss 和 HF 對不起來，省下來的記憶體就不能算數，那是換了一個不同的數學，不是優化。

### 4.3 怎麼做

notebook §4.1 → §4.4。三組各跑 30 步，和 Week 2 的消融同樣的做法（取後半平均，前幾步含 warmup 與編譯）。

**⚠️ §4.3 跑完務必「執行階段 → 重新啟動工作階段」再跑 Step 5。**Unsloth 會 patch transformers，和上一格的原生 HF 混在同一個 process 裡容易出怪事。

### 驗收

- [ ] 三組都有數字（HF fp16 那組「失敗並拋出 NaN/溢位錯誤」也算有效結果，要記錄下來）
- [ ] Unsloth 的峰值記憶體 < HF fp32 的峰值記憶體
- [ ] Unsloth 與 HF fp32 的 **末 loss 差距 < 10%**。差更多就要查，不要當成「優化的代價」帶過

---

## Step 5：LoRA 參數掃描（答主管 Q2）

**要做什麼**：把 Week 2 那 17.4 個百分點裡「超參數開太大」的成分量出來。

### 5.1 Stage A —— scaling 掃描（Week 3 最重要的一組）

固定 `r=16`、all-linear、`inline` 資料、200 步，只動 `lora_alpha`：

| run | alpha | scaling | 意義 |
|---|---:|---:|---|
| A1 | 8 | 0.5 | 很保守 |
| A2 | 16 | 1.0 | |
| A3 | 32 | **2.0** | **業界常規** |
| A4 | 64 | 4.0 | |
| A5 | **320** | **20.0** | **重現 Week 2** |

**先寫下預測**：無法解析率隨 scaling 單調上升；A5 應該重現 Week 2 的崩潰（~40%）；A3 應該 < 5%。

**判讀**：

- A5 崩、A3 不崩 → **Week 2 的主因是超參數。**這是最可能的結果，也是最有價值的結論
- A3 也崩 → 才輪到「Gemma 的 post-training 特別脆弱」這個解釋，Step 6 的 Shadow-FT 就變成主角
- 全都不崩 → 那 Week 2 的崩潰另有原因（資料模板？步數？），要回頭查

### 5.2 Stage B —— rank 掃描

固定 scaling = Stage A 的最佳值，掃 r ∈ {8, 16, 32, 64}。

**這一軸要和 Step 6 一起看。**Shadow-FT 論文的 ablation 顯示：常規 LoRA 是 **rank 越大傷害越大**（Llama-3.2-1B：r=4 得 30.11 → r=512 得 28.47），Shadow-FT 剛好相反（30.26 → 32.03）。如果我們也重現出這個交叉，那會是一張很有說服力的圖。

### 5.3 Stage C —— target module

`attn-only`（q,k,v,o）vs `all-linear`（再加 gate/up/down）。

Week 2 的 H6 只比了「掛幾層」沒比「掛哪些模組」，而且**實際上只掛到 q 和 o**（Step 0.2）。這一組把它補齊。

### 5.4 GPU 配額預算

**⚠️ 以下時間全部是估計值，不是實測。**第一組跑完就用實際的 `wall_min` 重算整份預算，不要照抄。

| 項目 | 組數 | 每組（估） | 小計 |
|---|---:|---:|---:|
| Stage A | 5 | ~12 min | 1.0 h |
| Stage B | 3（B2 = A3，不重跑） | ~12 min | 0.6 h |
| Stage C | 1（C1 = attn-only；all-linear 那一組就是 A3，不重跑） | ~12 min | 0.2 h |
| 快速評測（每科 100 題） | 10 | ~8 min | 1.3 h |
| **小計** | | | **~3.1 h** |

**跑不完怎麼辦**：優先序是 **Stage A > Step 6 Shadow-FT > Stage B > Stage C**。Stage A 沒跑完，後面全部沒有基準。

### 驗收

- [ ] 每組的 `train_*.json` 都有 `trainable_params` 且數字合理（r=16 all-linear 應該接近 §3.1 預測的 29.8M）
- [ ] 每組的 `loss_last` 都不是 NaN
- [ ] Stage A 的無法解析率**單調**（允許一個非單調點，兩個以上代表 100 題的雜訊太大，要加大 `QUICK_N`）

---

## Step 6：Shadow-FT（答主管 Q4）

**要做什麼**：驗證「在 base 上訓練、把 delta 搬到 instruct」能不能保住輸出格式。

### 6.1 方法

*Shadow-FT: Tuning Instruct Model via Training on Paired Base Model*（arXiv 2505.12716）

```
Step 1:  W_B⁺ = Tune(W_B)                  ← 在 BASE 上訓練
Step 2:  W_I⁺ = W_I + (W_B⁺ − W_B)         ← 把差值搬到 INSTRUCT
```

LoRA 下 base 項會抵消：`W_I⁺ = W_I + (W_B + BA − W_B) = W_I + BA`。**所以實作就是「在 `-pt` 上訓練 adapter，然後把同一個 adapter 搬到 `-it` 上」，零額外成本。**

### 6.2 先驗證前提，再做實驗

論文的立論基礎是 base 和 instruct 的權重非常接近，定義

```
σ = Σ|W_B − W_I| / (Σ|W_B| + Σ|W_I|)
```

論文實測所有模型 σ < 0.05。notebook §3.2 會對我們這一對算出來。

**驗收**：σ < 0.05。如果我們這一對算出來很大，**要先停下來想為什麼**，不要當作前提成立就往下做。

### 6.3 2×2 設計

| | scaling = 2 | scaling = 20 |
|---|---|---|
| 常規 LoRA on `-it` | **A3** | **A5** |
| Shadow-FT（train on `-pt`） | **D1** | **D2** |

三條假設：

| 編號 | 假設 | 判準 |
|---|---|---|
| **S1** | Shadow-FT 保住輸出格式 | D1 的無法解析率 < 5%，且明顯低於 A3 |
| **S2** | 格式保住之後，知識增益才看得到 | D1 嚴格正確率 ≥ **未微調的 `gemma-3-4b-it`**（§7 的 `base_it` 那一列）。注意基準線是 instruct 模型，不是 `-pt` |
| **S3** | Shadow-FT 對超參數不敏感 | \|D1 − D2\| **顯著小於** \|A3 − A5\| |

**S3 是我們自己加的，論文沒做。**如果成立，它比論文的說法更強：Shadow-FT 的價值不只是分數更高，是**讓超參數不再那麼要命**。

### 6.4 三個實作上的坑

**（a）訓練 `-pt` 時要用 `-it` 的 chat template。**Base 模型不保證有 chat template，就算有也可能不一樣。delta 必須活在和目標模型同一個座標系裡。notebook §2.1 一律用 `tok_it` 渲染，所以訓練 base 時只是在編碼同一批字串，天然正確。

**（b）不要用 `PeftModel.from_pretrained` 自動掛，改成逐張量手動相加。**PEFT 的模組命名依賴載入路徑，unsloth 包過一層之後不保證能對上 `-it` 的 state dict。notebook §6.2 的 `shadow_graft()` 手動做，對不上就 assert 失敗，**不會靜默錯掉**。這是 Week 2「三個 bug 會安靜地產生合理結果」的教訓。

**（c）比較的兩邊都要融合成完整權重。**不能拿「融合後的 Shadow-FT」比「掛 adapter 的常規 LoRA」——推論路徑不同，數值會有微小差異。notebook §6.3 把 A3/A5 也融合。

**（d）QLoRA 的量化誤差會被一起搬過去。**adapter 是對著 4-bit 的 `W_B` 訓練的，其中含有補償量化誤差的成分，搬到 `W_I` 上時那部分是雜訊。緩解：評測時兩邊都用同樣的 4-bit 載入，讓量化處理對稱。**這一條要寫進報告的限制。**

### 驗收

- [ ] σ < 0.05
- [ ] `shadow_graft` 的 `n_grafted` **等於** adapter 裡的模組數（程式已 assert）
- [ ] `‖ΔW‖/‖W‖` 和 σ 在同一個數量級。大很多代表 LoRA 太強
- [ ] 四個融合模型都在磁碟上（A3_merged / A5_merged / D1_shadow_merged / D2_shadow_merged）

---

## Step 7：評測

**要做什麼**：拿到可以直接比較的數字。

### 7.1 為什麼不直接用 twinkle-eval

讀 `Eval/twinkle_eval/` 原始碼發現兩件會影響可信度的事：

1. **`shuffle_options` 用沒設種子的全域 `random`**（整個套件 grep 不到任何 `seed`）。所以 **Week 2 的 base 跑和 tuned 跑，選項順序是不一樣的**——那 17.4 pt 裡有一部分是不同排列造成的雜訊。
2. **`average_accuracy` 是對「科目」平均，不是對「題目」平均。**三科題數 768 / 139 / 129，台語（129 題）和台灣地理（768 題）在總分裡權重相同。

notebook §7.1 的評測器**逐字複製了 twinkle-eval 的 prompt 組法與 box 解析邏輯**（`\\{1,2}box{([A-Z])}` 兩條 pattern、`question\nA: …\nB: …` 的組法、system prompt 全文），但把種子固定，並同時報四個數字：

- **macro**（可和 Week 2 對照）+ **micro**（題目加權）
- **嚴格**（只認 `\box{X}`）+ **寬鬆**（再接受「答案是 X」）

### 7.2 兩層評測

| 層 | 題數 | 用途 | 每次 |
|---|---:|---|---:|
| 快速 | 每科 100 題（共 300） | Stage A/B/C 掃描 | ~8 min |
| 完整 | 三科全部 1,036 題 | 決選的 5 個模型 | ~30 min |

**快速評測只用來排序，不用來下結論。**notebook §9 有一條對帳：快速評測和完整評測的**排序必須一致**。不一致代表 100 題的雜訊太大，掃描結論不可信。

### 7.3 開跑前的防呆

Week 2 有一次評測請求全部成功、解析率卻是 0%，浪費一整晚。notebook §7.2 會先送 3 題，抽不出 `\box{}` 就 assert 失敗。**不要跳過這一格。**

### 7.4 和 Week 2 的可比性

**不能直接並列。**Gemma 3 不是推理模型：

| | Week 2（Gemma 4 E4B） | Week 3（Gemma 3 4B） |
|---|---|---|
| `enable_thinking` | true | 不適用 |
| `max_tokens` | 2048 | 512 |
| 選項洗牌種子 | 未固定 | 固定 42 |
| 平均方式 | macro | macro + micro 並列 |

報告裡 Week 2 / Week 3 分表呈現，**只有同一份 Week 3 config 底下的組間比較才宣稱有因果意義**。

### 驗收

- [ ] 未微調 `-it` 的無法解析率 < 5%（一開始就高是 prompt 問題，不是模型問題）
- [ ] 每個 eval 的 `n_questions` 是預期值（快速 300、完整 1,036）
- [ ] 快速與完整評測的排序一致

---

## Step 8：MoE routing 後續（不吃 GPU 配額）

**要做什麼**：把主管有興趣的那部分補深。

主體已經寫在 `moe_routing_分析.md`。Week 3 要動手的是**前置作業**（§5.1），因為後面所有提案都建立在「能穩定認出中文專家」上：

1. 改 `scripts/inspect_router_mlx.py`，**逐 prompt 存 counts**（目前只存加總）
2. 加**以 prompt 為單位的 permutation test**（把 10 組配對隨機交換中英標籤）
3. 做**繁簡對照材料** —— 這是最關鍵的一條：分辨我們找到的是**語言**專家還是**字符集／tokenization** 專家

**驗收**：繁簡對照的 KL 應該**顯著小於**中英對照的 KL。如果兩者差不多，`moe_routing_分析.md` §5 的四個微調提案就要重寫立論。

**已經算好、可以直接用的結論**（`scripts/analyze_router_lang.py`）：

- 平均只要 **2.9 個專家**就涵蓋單層 80% 的中英路由 KL
- 全模型 3,840 個專家裡只有 **128 個**明確中文傾向，其中 **26 個**英文樣本裡一次都沒選到
- 每層取 top-3 掛 LoRA r=16 只要 **15.2M** 參數（0.23 GiB），對照全部路由專家掛滿的 648.8M（9.67 GiB）—— **省 43 倍**

**⚠️ 但實作路徑要先改。**我原本寫的「用 PEFT 的 `target_modules` 正則指到單一專家、估一天」**是錯的**。Gemma 4 MoE 的專家權重是**融合成一張 3-D 張量**（mlx-lm 是 `SwitchGLU` + `gather_mm`，HF 是 `layers.{L}.moe.gate_up_proj [n_experts, 2×inter, hidden]`），根本沒有 `experts.30.gate_proj` 這種模組路徑可以指。`moe_routing_分析.md` §5.3 已改寫成三條可行路徑，建議先做成本最低的「遮罩法」驗證假設。

---

## Step 9：收尾

### 9.1 產出 `week3_執行總結.md`

結構沿用 Week 2：

1. 兩句話總結
2. Week 2 交接的五個關鍵事實（Step 0）—— **放前面，不要藏在附錄**
3. 框架對照表（答 Q1）
4. Stage A 主圖 + 判讀（答 Q2）
5. 2×2 表 + S1/S2/S3 判定（答 Q3、Q4）
6. MoE routing 摘要，細節指向專篇
7. 下週規劃

### 9.2 更新假設狀態總表

### 9.3 把結果同步回 repo

notebook §9.1 打包下載，或：

```bash
rsync -av ~/Google\ Drive/My\ Drive/ultrascale-lab-week3/results/ results/week3/
rsync -av ~/Google\ Drive/My\ Drive/ultrascale-lab-week3/reports/ reports/week3/
git add -A && git commit -m "week3: Gemma 3 + Shadow-FT + LoRA 參數掃描"
```

---

## 一頁時間表

**⚠️ GPU 時數欄全部是估計值。**Colab 的每週配額（15–30 h）也是第三方觀察，Google 不公布。第一天跑完就用實測值重算。

| 天 | 項目 | GPU 時數（估） | 產出 |
|---|---|---:|---|
| D1 | Step 0 讀交接事實 + Step 1 本機收尾 | 0 | checkpoint 格式保留率曲線 |
| D1 | Step 2 Colab 環境 + Step 3 資料 | 0.3 | 三份訓練資料 |
| D2 | Step 4 框架對照 | 0.7 | **框架對照表（答 Q1）** |
| D2–D3 | Step 5 Stage A + 快速評測 | 2.3 | **scaling 主圖（答 Q2）** |
| D3 | Step 5 Stage B/C + 快速評測 | 0.8 | rank / target module 表 |
| D4 | Step 6 Shadow-FT D1/D2 + 融合 | 0.8 | 四個可比模型 |
| D4 | Step 7 完整評測 × 5 | 2.5 | **2×2 表（答 Q3、Q4）** |
| D5 | Step 8 MoE 前置作業 | 0 | 繁簡對照結果 |
| D5 | Step 9 收尾 | 0 | `week3_執行總結.md` |
| — | 緩衝（斷線、重跑、拿不到 GPU） | 2.4 | |
| | **合計** | **~10 h** | 免費版週配額 15–30 h，有餘裕 |

---

## 假設狀態總表

| 來源 | 編號 | 主題 | 狀態 |
|---|---|---|---|
| ch01 | H1 | 權重記憶體預測 | ✅ Week 1/2 已證實（誤差 7–9%） |
| ch01 | H3 | 梯度檢查點取捨 | ✅ Week 2 已證實（活化 −91.0%） |
| ch01 | H4 | logits 隨 seq 線性成長 | ✅ Week 2 已證實 |
| ch01 | H5 | 活化隨 batch 的成長律 | ✅ Week 2 已證實（扣掉固定項後成立） |
| ch01 | H6 | LoRA 掛的層數 | ⚠️ Week 2 只看 train loss；**Week 3 Stage C 補下游指標** |
| ch10 | P1 | 4-bit vs bf16 記憶體 | 待 Week 3 |
| ch10 | **P2** | **Flash Attention 效益** | **Week 3 才做得了**（MLX 沒有開關，CUDA 有 SDPA backend context manager） |
| ch06 | E1/E2 | active 比例、roofline | ✅ Week 1/2 已證實 |
| ch06 | E3 | 專家負載均衡 | ✅ 已證實（訊噪比 7.6×） |
| ch06 | **E4** | **中英路由差異** | ✅ 已證實：訊噪比 6.6×，平滑 KL 與 JSD 各 29/30 層 p<0.01 |
| ch06 | E5 | 微調後路由偏移 | ✅ 已證實（前 22 層精確為 0） |
| ch02 | D1–D5 | DP / ZeRO | ❌ 單卡，只能理論 |
| **新增** | **S1** | **Shadow-FT 保住格式** | **Week 3 主結果** |
| **新增** | **S2** | **Shadow-FT 知識增益** | **Week 3 主結果** |
| **新增** | **S3** | **Shadow-FT 對超參數不敏感** | **Week 3 主結果（論文沒做）** |

---

## 常見坑（先知道就不會踩）

| 坑 | 症狀 | 解 |
|---|---|---|
| **T4 沒有 bf16** | Gemma 3 訓練 loss 直接變 NaN | 用 Unsloth。這不是選擇，是唯一解 |
| **雙 BOS** | 沒有錯誤訊息，只是效果變差 | `render()` 剝掉 template 加的 `<bos>`，讓 tokenizer 統一負責 |
| **Unsloth patch 污染** | 先跑 HF 再跑 Unsloth（或反過來）出現奇怪的錯 | §4.3 跑完重啟工作階段 |
| **Drive I/O 很慢** | 訓練速度只有預期的 1/3 | 大檔（模型、adapter）寫 `/content/scratch`，只把小結果同步回 Drive |
| **Colab 閒置斷線** | 90 分鐘沒互動就掉 | 掛著跑時每小時回去點一下；所有實驗設計成可斷點續跑 |
| **`shuffle_options` 沒種子** | 兩次評測的分數莫名差 1–2 pt | 用 notebook 的評測器，種子固定 |
| **macro vs micro** | 三科平均和逐題平均差很多 | 兩個都報 |
| **快速評測雜訊** | 100 題的排序和 1,036 題不一致 | 只用快速評測排序，結論一定要跑完整版 |
| **adapter 掛不上目標模型** | `merge_and_unload()` 沒報錯但權重沒變 | 用 `shadow_graft()` 手動相加 + assert，不要用自動掛載 |
| **磁碟爆掉** | 四個融合模型 × 8.6 GB + HF 快取 | Colab 有 ~107 GB，夠用但要留意；跑完 Step 7 就可以刪掉融合模型 |

---

## 若升級 Colab Pro，額外可以做什麼

手冊主線是照免費版 T4 設計的。升級之後（L4 / A100，**有 bf16**）可以加做：

| 項目 | 為什麼免費版做不了 |
|---|---|
| **P2：Flash Attention on/off 對照** | 需要 SDPA backend context manager 明確切換，且 T4 不支援 FA2（需要 Ampere 以上） |
| HF transformers 當主線 | 有 bf16 就不會 NaN，可以用最接近教科書公式的框架做所有實驗 |
| `gemma-3-12b` 驗證可擴展性 | T4 上 4-bit QLoRA 約 9–11 GiB、慢 3 倍，一次跑滿就吃掉大半週配額 |
| P1：bf16 vs 4-bit 準確率對照 | 需要 bf16 基準線 |
| rank 掃到 128 / 256 | 驗證 Shadow-FT 論文「rank 越大越好」的說法（論文掃到 512） |

---

## 附：這週新增／變動的檔案

| 檔案 | 內容 | 狀態 |
|---|---|---|
| `week3_執行手冊.md` | 本檔 | 新增 |
| `week3_主管提問回覆.md` | 主管四個提問的逐條回覆 | 新增 |
| `moe_routing_分析.md` | MoE 路由專篇 + 四個微調提案（A/B/C/E）與成本表 | 新增 |
| `notebooks/week3_colab.ipynb` | Colab 完整流程（47 格） | 新增 |
| `scripts/analyze_router_lang.py` | 路由二次分析：bootstrap 底線、逐層 p-value、專家層級 KL 貢獻、LoRA 預算 | 新增 |
| `reports/router_lang_analysis.json` | 上者的輸出 | 新增 |
| `scripts/predict_memory_gemma.py` | 4-bit 常數改為 4.50 bit/param，記憶體表已重算 | 已更新 |
| `scripts/verify_load_mlx.py` | H1 的預期值改為 3.91 / 13.22 GiB | 已更新 |
| `configs/lora_gemma4_*.yaml` | 補上 `scale` 語意與 k/v 掛不上的說明 | 已更新 |
| `week2_執行總結.md`、`week2_執行手冊.md`、`notes/ch01,02,06,10.md`、`README.md` | 記憶體與 LoRA 數字已與上述一致 | 已更新 |
| `results/week3/`、`reports/week3/` | 從 Drive 同步回來的結果 | 待產生 |
| `week3_執行總結.md` | Step 9 產出 | 待產生 |

**待補**：`notes/ch05.md`、五個 checkpoint 的格式保留率曲線、繁簡對照的路由結果。

---

## Sources

- Shadow-FT: Tuning Instruct Model via Training on Paired Base Model — arXiv 2505.12716（本地：`shadow-ft.pdf`）
- [Fine-tune Gemma 3 with Unsloth](https://unsloth.ai/blog/gemma3) —— fp16 溢位的成因與修法
- [Google Colab | Unsloth Documentation](https://unsloth.ai/docs/get-started/install/google-colab) —— Colab 安裝方式
- [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) / [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
- [Google Colab Free Tier Limits (2026)](https://joshthompson.co.uk/ai/google-colab-2026-guide-free-compute-automations-pro-tips/) —— 配額與 12 小時上限
