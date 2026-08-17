# Week 3 — 回覆主管四個提問

## Q1. 現在用的框架是純 Transformer 還是有記憶體優化的框架？

**答：都不是「純 PyTorch transformers」。Week 2 用的是 Apple 的 MLX（`mlx-lm 0.31.3` / `mlx 0.32.0`），不是 HuggingFace transformers，也不是 unsloth / ms-swift 這類優化框架**

MLX 是一個獨立的陣列框架，不是 PyTorch 的封裝。它有一部分記憶體優化，但缺的正好是幾個關鍵項目——而 Week 2 那些「公式對不上」的地方，有一半來自這裡。

### MLX 有的

| 機制 | 說明 | Week 2 的證據 |
|---|---|---|
| Unified memory | CPU/GPU 共用同一塊實體記憶體，沒有 H2D/D2H 複製，也沒有獨立的 VRAM 上限 | 24 GB 全部可用，但也代表沒有「GPU 使用率」這個可觀測維度 |
| Lazy evaluation + graph fusion | 運算先建圖，`mx.eval()` 才落地，中間張量可被融合掉 | 活化實測 3.54 GiB，比公式預測低 2.4 倍 |
| 原生 4-bit 量化 | group_size 64，等效 **4.50 bit/param**（4 bit + 每 64 個元素一組 bf16 的 scale 和 bias → 4 + 32/64） | 權重預測誤差 **−0.0%** |
| Gradient checkpointing | `grad_checkpoint: true` | H3：活化 −91.0%，與預測的 92% 重合 |

> **這個常數很容易寫錯**直覺會寫 4.25（4 bit + 每 32 個元素一個 byte），但那是 group_size 128 的值。MLX 預設 group_size 64，而且 scale 和 bias 都是 bf16，所以是 4 + (2+2)×8/64 = **4.50**。直接量 `out/gemma4-e4b-tw/model.safetensors` 得到 4.501。用對之後，權重預測對 E4B 和 26B 的誤差都是 **−0.0%****同樣叫「4-bit」，group_size 不同就差 5.6%** —— Week 3 換到 bitsandbytes 的 nf4 要重新量，不能沿用。

### MLX 沒有的（這是重點）

| 缺的東西 | 後果 | Week 2 撞到的具體現象 |
|---|---|---|
| **Fused / chunked cross-entropy** | 262,144 vocab 的 logits 必須整張落地 | seq 2048 × bs 1 的 logits 就吃掉 **3.00 GiB**，佔峰值的 28%，把梯度檢查點的效益從 −91%（活化）稀釋成 −30.5%（峰值） |
| **Flash Attention 開關** | attention 實作寫死，沒有 on/off | 假設 P2「Flash Attention 省下多少活化」在 MLX 上**做不出對照組**，只能等 CUDA |
| 8-bit / paged optimizer | 優化器狀態固定 16 bytes/param | LoRA 情境下影響小，全參數微調就會是瓶頸 |
| CPU offload / ZeRO / FSDP | 單機單卡，沒有分片 | ch02 的 D1–D5 全部只能停在理論 |
| Triton / CUDA custom kernel | 沒有 fused RoPE、fused RMSNorm、fused MLP | 這是 unsloth 主要的速度來源 |

### 四個框架的定位（Week 3 要換到哪一個，以及為什麼）

| 框架 | 後端 | 記憶體優化程度 | 適合什麼 |
|---|---|---|---|
| **mlx-lm**（Week 2 用的） | Apple Metal | 中：量化 + 檢查點，但無 fused CE、無 FA 開關 | Mac 本機，零成本迭代 |
| **HF transformers + peft + TRL** | CUDA | 低（=「純 Transformer」baseline）：SDPA/FA2 可切、bnb 量化，其餘靠自己 | **當對照組用**：最接近教科書公式，最好 debug |
| **Unsloth** | CUDA | 高：Triton 手寫 kernel、fused CE（不落地完整 logits）、重寫的 gradient checkpointing、動態 4-bit 量化 | Week 3 主線 |
| **ms-swift** | CUDA | 中高：整合 DeepSpeed/FSDP/vLLM，偏「全家桶」 | 多卡、多方法（DPO/GRPO）時才划算 |

**Week 3 的決定：主線用 Unsloth，並跑一組 HF transformers 的對照**

---

## Q2. 目前 LoRA 參數是什麼？有比較不同參數的 performance 嗎？

### 先回答第二個問題：**沒有。這是 Week 2 的缺口**

Week 2 的 H6 只比較了「LoRA 掛幾層」（4 / 16 / 全部 42），而且**只看 train loss，沒有接到下游準確率**。rank、alpha、target module 三個軸完全沒動過，用的是我從一開始就寫死的一組值。

### 現在的參數（`configs/lora_gemma4_e4b.yaml`）

| 參數 | 值 |
|---|---|
| rank `r` | 16 |
| `scale` | **20.0** |
| dropout | 0.0 |
| keys | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `num_layers` | 16（最後 16 層，不是前 16 層） |
| lr | 1e-4，cosine decay，warmup 50 步 |
| iters | 1000，batch 1，seq 2048，grad_checkpoint on |

### 發現 1：我們的 LoRA 強度是業界常規的 10 倍

我去翻了 `mlx_lm/tuner/lora.py` 的原始碼：

```python
def __call__(self, x):
    y = self.linear(x)
    z = (self.dropout(x) @ self.lora_a) @ self.lora_b
    return y + (self.scale * z).astype(x.dtype)
```

**`scale` 是直接乘在 `x @ A @ B` 上的，沒有除以 rank**這和 HuggingFace PEFT 的慣例不同——PEFT 的 `scaling = lora_alpha / r`。

換算過去：

| | 我們的設定 | 業界常規 |
|---|---|---|
| mlx `scale`（等效 scaling factor） | **20.0** | 2.0 |
| 等效 PEFT `lora_alpha`（r=16） | **320** | 32 |

而 `scale: 20.0` 是 mlx-lm 的預設值，我從來沒有動過它，也沒有意識到它的語意和 PEFT 不一樣。

**這件事直接連到 Week 2 的失敗**LoRA 的更新量被放大 10 倍，等於用一個非常大的步伐去覆寫原模型的行為——這正是「格式遵循被洗掉、知識沒掉」這種失敗長相的典型成因**在排除這個因素之前，我不認為可以說「Gemma 4 因為 RL 所以 SFT 會崩」**（見 Q3。）

### 發現 2：`k_proj` / `v_proj` 其實一個都沒訓練到

我直接解析 `out/lora-e4b/adapters.safetensors` 的 header：

- 64 個 tensor，全部是 `layers.{26..41}.self_attn.{q_proj,o_proj}.{lora_a,lora_b}`
- **沒有任何 `k_proj` / `v_proj`**
- 實際可訓練參數 **2,555,904（2.56M）**，設定檔註解對 16 層的估計是 3.5M

原因：E4B 的 `num_kv_shared_layers = 18`，最後 18 層（layer 24–41）共用前面算好的 K/V，**這些層根本沒有 `k_proj` / `v_proj` 模組**。而我們掛的是最後 16 層（26–41），完全落在這個區間裡。mlx-lm 對找不到的 key 是靜默略過的。

**所以 Week 2 真正跑的是：最後 16 層的 q_proj + o_proj，scaling = 20**

這兩個發現的共同教訓是：**設定檔不是事實，產出的檔案才是**兩個都是解析 `adapters.safetensors` 的 header 才看出來的，訓練過程一個警告都沒有。

### Week 3 的參數掃描（手冊 Step 5）

用 `gemma-3-4b`、固定資料與步數，每組跑完都接一次快速評測，主指標是**無法解析率**（格式保留）和**嚴格正確率**。

**Stage A — scaling 掃描**（這是最重要的一組，直接檢驗上面的假設）

| run | r | lora_alpha | scaling = α/r | 備註 |
|---|---:|---:|---:|---|
| A1 | 16 | 8 | 0.5 | 很保守 |
| A2 | 16 | 16 | 1.0 | |
| A3 | 16 | 32 | 2.0 | **業界常規** |
| A4 | 16 | 64 | 4.0 | |
| A5 | 16 | **320** | **20.0** | **重現 Week 2 的設定** |

（Week 3 全程用 **A1–A5 / B1–B4 / C1 / D1–D2** 這一套編號，手冊和 notebook 一致。Stage C 只有 C1（attn-only），all-linear 那一組就是 A3，不重跑。`A3` = 常規 scaling 的對照組，`A5` = Week 2 設定，`D1/D2` = Shadow-FT 的兩點。）

**事先預測**：無法解析率會隨 scaling 單調上升，A5 應該重現 Week 2 的崩潰。如果 A5 崩、A3 不崩 → 主因是超參數，不是模型。如果 A3 也崩 → 才輪到「Gemma 的 post-training 特別脆弱」這個解釋。

**Stage B — rank 掃描**（固定 scaling = Stage A 的最佳值）：r ∈ {8, 16, 32, 64}
Shadow-FT 論文的 rank ablation 顯示，**常規 LoRA 是 rank 越大傷害越大，Shadow-FT 是 rank 越大越好**（r=512 仍在漲）。所以這一軸要和 Q4 一起看。

**Stage C — target module**：`attn-only`（q,k,v,o）vs `all-linear`（再加 gate/up/down），固定其他。

---

## Q3. Gemma 4 有 RL 架構，直接 SFT 準確率崩掉是已知行為，希望用 Gemma 3 跑

**答：同意，Week 3 主線換成 `gemma-3-4b`。但我想把「已知行為」這件事當成待驗證的假設而不是前提，理由如下**

### 換 Gemma 3 的三個理由

1. **繞開 post-training 的干擾**主管的判斷方向和 Shadow-FT 論文一致：instruct 模型的 instruction-following 先驗會抵抗新知識的學習，直接 SFT 容易兩敗俱傷。

2. **Gemma-3-4B 正好是 Shadow-FT 論文裡效果最顯著的那一個**論文 Table 6：常規 FT 讓 Gemma-3-4B 掉 **6.81 分**，是全篇 19 個 benchmark 平均裡**掉最慘的模型**（次慘的 Yi-Coder-9B 掉 4.27），而 Shadow-FT 把它從 51.45 拉到 52.55**我們的失敗長相和它一模一樣**，所以在同一個模型上重現是最直接的驗證路徑。

   > 附帶一提：`google/gemma-4-E4B`（pre-trained）是有的，所以 Shadow-FT 在 Gemma 4 上技術上也做得了。換 Gemma 3 是因為**論文在它上面驗證過而且是最強的案例**，不是因為 Gemma 4 做不了**如果 Week 3 在 Gemma 3 上證實 Shadow-FT 有效，Week 4 就把它搬回 Gemma 4 E4B** —— 那才是直接回答 Week 2 那次失敗。

3. **成本**Gemma 3 4B 在 T4 上 4-bit QLoRA 約 6–8 GiB（估計值），一天可以跑完十幾組參數對照；Gemma 4 E4B 因為 Per-Layer Embeddings 與 262K vocab，同樣的 seq 長度下光 logits 就要 3 GiB。

### 但有兩件事要先講清楚

**（a）Week 2 的證據還不足以歸因到 RL**Week 2 觀察到的是：逐題正確率 −21.5 pt、無法解析率 0.68% → 39.19%、而逐題控制對照顯示知識面只掉 1.7 pt。這是「格式崩了、知識沒崩」。但如同 Q2 所述，那一輪的 **scaling = 20（常規的 10 倍）**，這個混淆項還沒有被排除。

**另外，準確率要看兩個平均**twinkle-eval 的 `average_accuracy` 是對三個科目取平均，但三科題數是 768 / 139 / 129 —— 台語（129 題）和台灣地理（768 題）在總分裡權重相同。改成逐題加權之後：

| | 科目平均（macro） | 逐題平均（micro） |
|---|---:|---:|
| base 正確率 | 53.98% | **59.75%** |
| tuned 正確率 | 36.57% | **38.22%** |
| **差距** | **−17.4 pt** | **−21.5 pt** |
| base 無法解析率 | 0.52% | 0.68%（7/1,036） |
| tuned 無法解析率 | 39.73% | 39.19%（406/1,036） |

**逐題看的話跌幅是 21.5 pt**兩個數字之後都會並列。

另外「知識面只掉 1.7 pt」這句要加一個但書：那是在「tuned 有正確輸出格式」的 630 題上比的，而 base 在那 630 題的正確率是 64.6%、在崩掉的 406 題上只有 52.2% —— **那 630 題是比較簡單的一群**。所以 −1.7 pt 是「在較簡單子集上的知識沒掉」，不能直接推論到全體。

回到歸因問題。我的建議是把它當成兩個可分離的因素，Week 3 用 2×2 設計把它拆開：

| | 常規 scaling (α/r=2) | Week 2 scaling (α/r=20) |
|---|---|---|
| **常規 LoRA on `-it`** | **A3** | **A5**（≈ 重現 Week 2） |
| **Shadow-FT（train on `-pt`）** | **D1** | **D2** |

如果 A5 崩而 A3 不崩 → 是超參數。如果 A3 也崩而 D1 不崩 → 主管的判斷成立，而且 Shadow-FT 是解法。兩者都可能同時成立，這個設計都能看得出來。

**（b）換 Gemma 3 之後，Week 3 的準確率不能和 Week 2 直接並列**Gemma 3 不是推理模型，沒有 thinking channel：

| | Week 2（Gemma 4 E4B） | Week 3（Gemma 3 4B） |
|---|---|---|
| `enable_thinking` | true | 不適用，評測 config 要移除 |
| `max_tokens` | 2048 | 512 就夠（也讓評測快 4 倍） |
| 訓練資料的 `think` 欄位 | 走 `<\|channel>thought` | 要重新決定怎麼渲染（手冊 Step 3 有三個選項） |

所以報告裡我會把 Week 2 / Week 3 的數字分表呈現，只有**同一份 Week 3 config 底下的組間比較**才宣稱有因果意義。

### 另外，Week 3 順便修掉評測腳本的兩個問題

讀 twinkle-eval 原始碼時發現的，兩個都會影響數字的可信度：

1. **`shuffle_options` 用的是沒有設種子的全域 `random`**整個套件 grep 不到任何 `seed`。所以 Week 2 的 base 跑和 tuned 跑，**選項順序是不一樣的**——17.4 pt 的差距裡有一部分是不同的題目排列造成的雜訊。Week 3 的評測腳本會固定種子。
2. **`average_accuracy` 是對「科目」取平均，不是對「題目」取平均**三科題數是 768 / 139 / 129，所以台語（129 題）和台灣地理（768 題）在總分裡權重相同。Week 3 會同時報 macro（可比 Week 2）和 micro（題目加權）兩個數字。

---

## Q4. 參考 Shadow-FT 論文，並嘗試應用到這個實驗

### 論文在說什麼

*Shadow-FT: Tuning Instruct Model via Training on Paired Base Model*（Wu, Yang et al., arXiv 2505.12716v3, 2025-09-26；code: `github.com/wutaiqiang/Shadow-FT`）

**核心觀察**：直接 SFT instruct 模型往往不但沒有變好，反而變差。論文在 Qwen-3-4B 上量到常規 LoRA 使 Math-7 掉 2.6、Code-3 掉 6.8、Knowledge-9 掉 2.6。

**機制**：instruct 模型「是好的 backbone，但是壞的學習者」。訓練初期 instruct 的 loss 比 base 高 22.6%、梯度大 **3.25 倍**，接著梯度在 11/61 步就崩塌，進入一個更新被壓抑的僵硬狀態——它在抵抗，而不是在學。

**方法**（兩步，零額外參數、零額外訓練成本）：

```
Step 1:  W_B⁺ ← Tune(W_B)                     ← 在 BASE 上訓練
Step 2:  W_I⁺ = W_I + (W_B⁺ − W_B)            ← 把差值搬到 INSTRUCT 上
```

前提是 base 和 instruct 的權重非常接近。論文定義相對差距 σ = Σ|W_B − W_I| / (Σ|W_B| + Σ|W_I|)，實測 Gemma-3、Qwen-3、Llama-3.1 全部 σ < 0.05。

**在 LoRA 下會退化成一件很簡單的事**：`Tune(W_B) = W_B + BA`，代進去 base 項互相抵消——

```
W_I⁺ = W_I + (W_B + BA − W_B) = W_I + BA
```

**也就是說：在 base 上訓練 LoRA adapter，然後把「同一個 adapter」直接掛到 instruct 上就好**不需要把 base 的完整權重存下來，也不需要做逐張量相減。實作成本幾乎為零。

**Gemma 3 的實測數字（論文 Table 6，19 個 benchmark 平均）**：

| 模型 | 原始 instruct | 常規 FT | **Shadow-FT** |
|---|---:|---:|---:|
| Gemma-3-4B | 51.45 | 44.64（**−6.8**） | **52.55** |
| Gemma-3-12B | 60.14 | 60.07 | **61.49** |

### 為什麼它特別適合我們這個失敗案例

Week 2 的診斷是「格式崩了」，而格式遵循正是 instruct 模型 post-training 學來的東西。Shadow-FT 的主張精確地是「不要碰 post-training 那一層，只把新知識當成一個增量疊上去」。

### Week 3 的三條可驗證假設

| 編號 | 假設 | 預測 | 怎麼量 |
|---|---|---|---|
| **S1** | Shadow-FT 能保住輸出格式 | D1 無法解析率 **< 5%**，且明顯低於同 scaling 的 A3 | 同一份評測，比 unparsed_rate |
| **S2** | 格式保住之後，知識增益才看得到 | D1 嚴格正確率 **≥ 未微調的 `gemma-3-4b-it`**（即手冊 Step 7 / notebook §7 的 `base_it` 那一列） | 嚴格 box 計分 |
| **S3** | Shadow-FT 對 LoRA 強度不敏感，常規 LoRA 敏感 | \|D1 − D2\| **顯著小於** \|A3 − A5\| | Q3 的 2×2 設計 |

### 三個實作上的注意事項（會寫進手冊）

1. **tokenizer 要一律用 `-it` 的**:Base 模型的 chat template 不保證存在或一致。訓練 base 時如果用了不同的模板，學到的 delta 就活在另一個座標系裡，搬過去會失真。
2. **QLoRA 的量化誤差會被一起搬過去**:adapter 是對著 4-bit 量化後的 `W_B` 訓練的，其中含有一部分「補償量化誤差」的成分。搬到 fp16 的 `W_I` 上時這部分是雜訊。緩解方式：評測時 base 與 shadow 兩邊都用同樣的 4-bit 載入，讓量化處理對稱。
3. **rank 要往上掃**: 論文的 ablation 顯示常規 LoRA 是 rank 越大越差（Llama-3.2-1B：r=4 得 30.11 → r=512 得 28.47），Shadow-FT 剛好相反（30.26 → **32.03**）。所以 Stage B 的 rank 掃描要在**兩種訓練法上各跑一次**，這個交叉本身就是一張很好看的圖。

**論文明確的限制**：必須有公開成對的 base 模型（Qwen3-32B 這種沒有 base 的就做不了）；方向不可逆（把 instruct 學到的 delta 搬到 base 上反而更差）；在 Falcon3-10B、Yi-6B、Llama-3.2-Vision-11B 上輸給常規 FT——共通點是這三個模型上常規 FT 本來就沒有退化**換句話說：Shadow-FT 是修 degradation 的，不是普遍加分的。這正好是我們需要的**

---

## 一頁摘要

| 提問 | 一句話答案 | Week 3 的動作 |
|---|---|---|
| **框架** | 是 MLX（mlx-lm），不是純 transformers 也不是優化框架；缺 fused CE 和 FA 開關，這解釋了 Week 2 兩個對不上的數字 | 換 Unsloth（T4 無 bf16，Gemma 3 fp16 會溢位），並跑 HF vs Unsloth 對照表 |
| **LoRA 參數** | r=16 / scaling **20**（= PEFT alpha 320，常規的 10 倍）/ 實際只訓到 q+o / 最後 16 層；**從未比較過不同參數** | Stage A scaling 掃描、Stage B rank 掃描、Stage C target module；每組都接下游評測 |
| **Gemma 3** | 同意換。理由是 Gemma-3-4B 正是 Shadow-FT 論文裡常規 FT 掉最慘、Shadow-FT 救得最多的模型（**不是因為 Gemma 4 沒有 base——它有**） | 主線 `gemma-3-4b`；用 2×2 把「超參數」和「post-training 脆弱性」拆開；同時修掉評測腳本沒設種子與 macro/micro 平均的問題 |
| **Shadow-FT** | 論文的失敗模式與我們 Week 2 相同；LoRA 下實作成本近乎為零 | 假設 S1/S2/S3；rank 軸在常規與 Shadow 兩邊各掃一次 |

**Week 3 結束時要能回答的一句話**：Week 2 那 17.4 個百分點（逐題看是 21.5 pt），有多少是超參數開太大、有多少是 instruct 模型本身抗拒 SFT，以及 Shadow-FT 能收回多少。

**如果 Week 3 證實 Shadow-FT 有效，Week 4 就把它搬回 Gemma 4 E4B** —— 那才是直接回答 Week 2 那個失敗。

---

## 附：這份文件裡的五個關鍵數字

| # | 數字 | 為什麼重要 |
|---|---|---|
| 1 | LoRA scaling = **20**（mlx 的 `scale` 是直接乘數，= PEFT `lora_alpha` 320） | 常規的 10 倍。Week 2 格式崩潰的頭號嫌疑犯，Stage A 直接驗它 |
| 2 | 實際訓練參數 **2.56M**，只有 q + o | k/v 掛的層剛好都在 KV 共用區，一層都沒生效 |
| 3 | 逐題正確率 **−21.5 pt**（科目平均 −17.4 pt） | 兩個平均差 4 pt，之後一律並列 |
| 4 | 4-bit 等效 **4.50 bit/param** | 用對之後權重預測誤差 −0.0%；group_size 不同就差 5.6% |
| 5 | E4 路由訊噪比 **6.6×**、29/30 層 p < 0.01 | 中英路由確實不同，而且集中在每層約 3 個專家上 |

前兩項只有解析 `adapters.safetensors` 才看得到，訓練過程完全沒有警告**這是 Week 2 方法論上最重要的一課：設定檔不是事實，產出的檔案才是**
