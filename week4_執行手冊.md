# Week 4 執行手冊（最後一週）

> **這週在做什麼**：三件事，做完就收工。
> **① AR vs DLM 對比**（proposal 的承諾）：DiffusionGemma 剛好就是 Gemma 4 26B-A4B 的
> 擴散版——同骨幹、同 25.2B/3.8B active、同 30 層 128 專家，這是比 proposal 原案
> 更乾淨的同架構對照。**② MoE routing 主線**（新增交付）：方案 E 偏置探針 → 方案 B
> router-only → 方案 A attention-only 基準線。**③ 完整進度報告**：PPTX 給主管 +
> Markdown 自留，整合四週。
>
> **不做**（和主管口徑一致）：延伸研究（proposal slide 10/11）、多卡 DP/ZeRO、
> MoE 方案 C（選擇性 expert LoRA——C-1 遮罩法要 9.67 GiB 優化器狀態，本機裝不下，
> 寫進報告的 future work）。
>
> 新程式：`scripts/week4_eval_server.py`、`scripts/router_bias_probe.py`、
> `scripts/make_week4_configs.py`、`scripts/verify_adapters.py`、`scripts/dlm_ar_bench.py`、
> `notebooks/week4_colab.ipynb`。報告骨架：`reports/week4/final_report_outline.md`。

---

## 0. 前三週的坑 → 這週的程式怎麼擋（開跑前掃一眼）

| 坑（哪一週） | Week 4 的防線 |
|---|---|
| twinkle-eval shuffle 無種子、macro/micro 混用（W2） | `week4_eval_server.py`：種子固定、四個數字並列 |
| 評測「全部成功、零解析」浪費一晚（W2） | 同上：開跑前強制 3 題防呆 + `/v1/models` 對帳 |
| localhost 被 proxy 吃掉 → 4,144 個 502（W2） | 同上：程式內自動設 `NO_PROXY` |
| k/v adapter 被靜默略過、scale=20 語意誤用（W2） | `make_week4_configs.py` 從真實模組樹盤點 + `verify_adapters.py` 訓後對帳（差 >1% 直接 fail） |
| thinking channel 的「答案是 X」被寬鬆解析誤抓（W2 嫌疑） | 兩個評測器都先剝 `<|channel>thought…<channel|>` 再計分 |
| 洩漏檢查因 set 順序「上次過這次炸」；TMMLU+ 跨科目重複題（W3） | notebook §2.1：確定性全文比對 + 生成端題目層級排除 |
| session 殘留灌水記憶體數字（W3） | `dlm_ar_bench.py`：AR/DLM 各用全新 process，腳本內有提醒 |
| 雙 `<bos>`（W3） | notebook §4.1 有 assert |
| 400 步後格式崩加速（W3 checkpoint 掃描） | 這週所有微調一律 **200 步**、scaling **2.0** |
| 「設定檔不是事實，產出的檔案才是」（W2 總教訓） | 每次訓練後 `verify_adapters.py` 是**必跑步驟**，不是選配 |

---

## 1. Colab Pro 額度預算（先查再排程）

Colab 右上角 → 資源檢視器 → 剩餘 compute units。Week 3 已經用過 T4，**不是滿額**。
概估（實際費率以 Colab 顯示為準）：A100 ≈ 8–12 units/hr、L4 ≈ 2–5、T4 ≈ 1–2。

**（2026-08-24 依官方 notebook 更新）**DiffusionGemma 微調/評測需要 **bf16 ~52GB**
（bnb 4-bit 壓不了 3-D 融合專家）→ Colab 要選 **A100 80GB 或 H100**；40GB A100 會
offload 慢到不可用，不要硬跑。好消息：**DLM base 品質已可在本機免費跑**
（`dlm_cli_eval.py`，見 Step 4.3），Colab 的唯一任務剩「微調 + tuned/base 同 engine 評測」。

| 情況 | 做法 |
|---|---|
| 有 A100 80GB / H100 且 units 足（微調+quick 評測約 1.5–2.5 hr） | 跑 notebook §3→§6 全套 |
| 有大卡但 units 很緊 | 微調 + tuned quick（base quick 可省——用本機 CLI base 對照，注明跨 engine） |
| 選不到 80GB 級的卡 | **降級 baseline-only**：本機 CLI 的 base 品質 + bench 已齊，tuned 記「硬體不可得」；proposal 的 4-checkpoint 改為 3 + 註記 |

本機（M4 Pro）完全免費，Step 1–4 全部本機。

---

## 2. 建議時程（報告日往回推）

| 天 | 白天 | 掛機/夜間 |
|---|---|---|
| D1 | Step 0 環境 + Step 1 Week 2 真因（4 輪快速評測） | Step 2 方案 E 探針（~1 hr）＋（選配）`--gen` |
| D2 | Step 3 configs + 方案 A 訓練（200 步） | 方案 A 的 zh/en 評測 |
| D3 | 方案 B 訓練 + 評測 + 路由漂移量測 | Step 4 DLM 下載（15 GB）+ bench |
| D4 | Step 4 AR vs DLM 本機評測 | Step 5 Colab DLM 微調（A100，盯前 30 分鐘再走） |
| D5 | Step 6 彙整 + Step 7 報告（md → pptx） | buffer |

塞不下就砍：先砍 §Step 1 的 thinking-off 兩輪（保留 thinking-on 重評），再砍方案 A/B 的完整評測（用快速版）。**方案 E 和 AR vs DLM baseline 不能砍**——一個是新交付的核心，一個是 proposal 承諾。

---

## Step 0：環境準備（30 分鐘）

```bash
cd ~/Projects/ultrascale-lab && source .venv/bin/activate

# DiffusionGemma 的 mlx 支援非常新，先升級（W2 的教訓：版本問題先驗，不要邊跑邊猜）
uv pip install -U mlx mlx-lm

# 26B 系列需要放寬 wired limit（重開機後失效）
sudo sysctl iogpu.wired_limit_mb=20480

# 磁碟：diffusiongemma 4-bit 還要 ~15 GB，先確認剩 >40 GB
df -h ~
```

- [x] `python -c "import mlx_lm; print(mlx_lm.__version__)"` 印得出版本
- [x] 磁碟 > 40 GB

---

## Step 1：Week 2 崩潰真因（本機，answers Week 3 遺留的第一優先）

**在做什麼**：Week 3 推翻了「超參數的鍋」，剩下兩個嫌疑：評測器雜訊（twinkle-eval
無種子洗牌）、thinking-channel 模板處理。用 Week 4 的乾淨評測器重評 Week 2 的模型，
thinking on/off 各一次，把兩個嫌疑分開。

**事前預測（寫下來再跑）**：
- **P-W2a** 乾淨評測器（種子固定）下，tuned 的無法解析率仍 >30% → 洗牌雜訊不是主因。
- **P-W2b** thinking off 時 tuned 的無法解析率明顯下降（<15%）→ 主因是 thinking-channel
  模板；兩者都不成立 → 主因回到「訓練資料格式 × 步數」（checkpoint 曲線已支持）。

```bash
# ── 輪 1/2：先跑 base（順序刻意：base 的防呆 3 題要能抽出 \box，證明管線正常）──
mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --host 127.0.0.1 --port 1234
# 另開終端機：
python scripts/week4_eval_server.py --tag w2base_think_on  \
    --model-name mlx-community/gemma-4-e4b-it-4bit --thinking on  --max-tokens 2048 --limit 100
python scripts/week4_eval_server.py --tag w2base_think_off \
    --model-name mlx-community/gemma-4-e4b-it-4bit --thinking off --max-tokens 512  --limit 100

# ── 輪 3/4：Week 2 tuned（fused 權重還在 out/gemma4-e4b-tw）──（重起 server 換模型）
# ⚠️ tuned 就是那個「散文回答不套 \box」的崩潰模型 —— 防呆 3 題全不中是**預期結果**，
#    無法解析率正是要量的東西。所以 tuned 這兩輪要加 --allow-unparsed-probe
#    （前提：上面 base 那兩輪的防呆已通過，管線正常性已被排除）。
mlx_lm.server --model out/gemma4-e4b-tw --host 127.0.0.1 --port 1234
python scripts/week4_eval_server.py --tag w2tuned_think_on  \
    --model-name out/gemma4-e4b-tw --thinking on  --max-tokens 2048 --limit 100 \
    --allow-unparsed-probe
python scripts/week4_eval_server.py --tag w2tuned_think_off \
    --model-name out/gemma4-e4b-tw --thinking off --max-tokens 512  --limit 100 \
    --allow-unparsed-probe
```

- [x] 4 份 `results/week4/eval_w2*.json` 都在（2026-08-23 完成）
- [x] **P-W2a 成立**：乾淨評測器下 tuned think-on 無法解析仍 44.3% → 洗牌雜訊非主因
- [x] **P-W2b 強烈成立**：同一模型 thinking off 後無法解析 44.3%→0.67%、正確率 32.3%→48.0%
- [x] **加碼發現**：tuned think-off 48.0% > base think-off 44.0% —— 微調在非 thinking
      模式下其實是**加分**的；Week 2 損壞的只有「thinking 模式下的格式遵循」

> ⚠️ 快速評測（每科 100 題）只用來歸因；要引用絕對數字就去掉 `--limit` 跑完整 1,036 題。

---

## Step 2：MoE 方案 E — 推論期路由偏置探針（本機，零訓練成本）

**在做什麼**：`moe_routing_分析.md` §5.5。對 Week 3 選出的「中文專家」
（繁簡 top-3 交集）在 router logits 上加偏置 b，看繁中/英文的 NLL 是否反向移動。
成立的話，「語言分工」就從相關性升級成**因果**證據——這是 MoE 主線的核心論述。

腳本內建三道對帳（找 30 顆 router linear、b=0 等於不 patch、b=+8 煙霧測試），
不通過會直接中止；事前預測 P-E1/E2/E3 開跑時會印在畫面上。

```bash
.venv/bin/python scripts/router_bias_probe.py                 # NLL 掃描 b∈{-2,-1,0,0.5,1,2}
# 有時間再加生成式評測（TMMLU+ 每科 25 題 × b∈{0,1,2}，約 1.5 hr）：
.venv/bin/python scripts/router_bias_probe.py --gen
```

- [x] 對帳 A/B/C 三關全過；含 `--gen` 生成式評測（2026-08-23 完成）
- [x] **P-E1 成立**：選中專家份額隨 b 單調上升（zh 0.2%→31.8%）
- [x] **P-E2 以「消融方向」強烈成立**：b=−1/−2 時 zh NLL +3.7~+4.0（崩壞）而 en 幾乎不動
      （−0.1）→ 這 78 個專家是**中文專用**的因果證據；正向 b=+0.5 小幅雙贏（zh −0.81/en −0.22），
      b=+2 兩者皆壞 → 強推流量有害
- [x] **生成式對照**：b=0/+1/+2 正確率 48%→41%→31% —— 推翻「加偏置能提升繁中」，
      支持「router 已近最優，介入只會傷」；語言分工的**因果性**與「偏置當微調替代品」
      的**不可行性**同時定案
- [x] 10 對翻譯配對段落補了 §5.1 的「翻譯配對」對照（小規模版）

---

## Step 3：MoE 方案 A / 方案 B（本機，各約 2–3 hr 訓練 + 評測）

**在做什麼**：機制二分。方案 B 只動 router（「模型不知道該找誰」假設），
方案 A 只動 attention（E5 已證它會間接改路由）。B 有效 → 路由問題；
B 無效而 A 有效 → 能力問題；都無效 → 與 Week 3「微調皆淨傷害」一致，也是可報告的結論。

```bash
# 3.1 產生設定檔（從真實模組樹盤點 key、算精確預測參數）
.venv/bin/python scripts/make_week4_configs.py
cat configs/week4/week4_meta.json          # 看 router key 叫什麼、預測參數多少

# 3.2 方案 A：attention-only 全 30 層、r16、scale 2.0、200 步
# ⚠️ 26B（MoE）訓練一律用 scripts/mlx_lora_train.py，不要直接呼叫 mlx_lm.lora：
#    mlx core 0.32.1 起 gather VJP 變嚴格，mlx-lm 0.31.3 的 gemma4 Router 缺
#    stop_gradient(top_k_indices)，直接跑會炸
#    「Cannot calculate VJP with respect to indices」。啟動器只補這一行，其餘不動。
python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planA.yaml --iters 30 --steps-per-report 5  # 試水溫
python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planA.yaml
python scripts/verify_adapters.py out/week4-26b-planA \
    --expect configs/week4/week4_meta.json --plan planA        # ← 不過就停，先查

# 3.3 方案 B：router-only（LoRA r32；報告要註明是 LoRA 不是全參數）
python scripts/mlx_lora_train.py --config configs/week4/lora_26b_planB.yaml
python scripts/verify_adapters.py out/week4-26b-planB \
    --expect configs/week4/week4_meta.json --plan planB

# 3.4 評測：zh（TMMLU+）+ en 對照組（MMLU）×（base / A / B）
#     英文對照是關鍵：只動中文相關路徑，英文不該掉。
#  base 兩輪已完成（zh 58.0%/13.3%、en 82.7%/7.3%）。
#  ⚠️ 2026-08-25 事故：adapter 評測 4 輪與 base 逐字全同 —— 舊 server 佔著 port，
#     --adapter-path 的新 server 沒接到請求。重跑規則：
#     ① 起 server 前先殺 port：lsof -ti:1234 | xargs kill
#     ② 確認新 server 的啟動 log 有 adapter 載入訊息、terminal 沒有 bind 錯誤
#     ③ 評測加 --differs-from（前 20 題與 base 全同就自動中止，不再白跑 30 分鐘）
lsof -ti:1234 | xargs kill
mlx_lm.server --model mlx-community/gemma-4-26B-A4B-it-4bit --adapter-path out/week4-26b-planA --port 1234
python scripts/week4_eval_server.py --tag planA_zh --model-name mlx-community/gemma-4-26B-A4B-it-4bit --thinking off --limit 100 \
    --differs-from results/week4/eval_moe26b_base_zh.jsonl
python scripts/week4_eval_server.py --tag planA_en --model-name mlx-community/gemma-4-26B-A4B-it-4bit --dataset mmlu --thinking off --limit 100 \
    --differs-from results/week4/eval_moe26b_base_en.jsonl
lsof -ti:1234 | xargs kill
mlx_lm.server --model mlx-community/gemma-4-26B-A4B-it-4bit --adapter-path out/week4-26b-planB --port 1234
python scripts/week4_eval_server.py --tag planB_zh --model-name mlx-community/gemma-4-26B-A4B-it-4bit --thinking off --limit 100 \
    --differs-from results/week4/eval_moe26b_base_zh.jsonl
python scripts/week4_eval_server.py --tag planB_en --model-name mlx-community/gemma-4-26B-A4B-it-4bit --dataset mmlu --thinking off --limit 100 \
    --differs-from results/week4/eval_moe26b_base_en.jsonl

# 3.5 路由漂移（E5 的延伸：方案 B 動 router 本身，漂移形狀應該和 E5「間接漂移」不同）
python scripts/inspect_router_mlx.py --adapter out/week4-26b-planA \
    --save reports/week4/router_after_planA.json --compare-with reports/router_before.json
python scripts/inspect_router_mlx.py --adapter out/week4-26b-planB \
    --save reports/week4/router_after_planB.json --compare-with reports/router_before.json
```

**事前預測**：
- **P-A1** 方案 A 的路由漂移集中在後段層（E5 的重現，這次是全 30 層掛）
- **P-B1** 方案 B 的漂移應該**全層都有**（router 直接被動了）
- **P-AB** zh 增益：參照 Week 3 經驗，兩者可能都 ≤ base；重點看 **A、B 之間的相對差**
  和 **en 是否不掉**（en 掉了 → 動到的不是中文專屬路徑）

- [x] 兩個 `verify_adapters.py` 對帳通過（**+0.0%**：planA 11.49M attention、
      planB 2.83M router.proj×30 層——訓練面完全驗證）
- [x] base 兩輪＋2 份路由漂移 json 齊：**P-B1 成立**（planB 全 30 層 KL>0、平均 1.52）；
      planA 全 30 層 KL>0、平均 0.71（全層掛 adapter，E5 的「未掛層=0」對照不適用，
      形狀判讀改看逐層分布）
- [ ] **待重跑**：planA/planB 的 zh/en 四輪（首輪因舊 server 佔 port 量到 base，
      已移 `_junk`；重跑按上面 3.4 的新規則，約 2 小時）
- [ ] 2×3 表（base/A/B × zh/en）進報告

> ⚠️ 26B 訓練貼邊：跑之前關掉大型 App；若 OOM，`max_seq_length` 已是 1024，
> 下一步是 `num_layers: 30 → 16`（記得同步改預測值的口徑）。

---

## Step 4：AR vs DLM 對比（本機為主）

**在做什麼**：proposal slide 9 的表。AR = `gemma-4-26B-A4B-it-4bit`（Week 2 的 MoE
對照組），DLM = `diffusiongemma-26B-A4B-it-4bit`（同骨幹的擴散版）。同一台 M4 Pro、
同為 MLX 4-bit，量：品質（TMMLU+）、吞吐、峰值記憶體。

**事前預測（接回 Week 2 roofline，報告的亮點）**：AR 解碼是頻寬瓶頸（53.6 tok/s =
上限的 48%）；DLM 的 canvas forward 像 prefill，是算力瓶頸。而 H5 已證 M4 Pro 在
batch=1 就算力飽和 → **官方的 4× 加速在 M4 Pro 上應該打折**。量出多少是多少，
落差本身就是 roofline 的一次應用。

```bash
# 4.1 下載（~15 GB）
HF_HUB_DISABLE_XET=1 hf download mlx-community/diffusiongemma-26B-A4B-it-4bit --max-workers 4

# 4.2 bench：AR、DLM 各用【全新的 process】跑（session 殘留教訓）
.venv/bin/python scripts/dlm_ar_bench.py --model ar                       # AR 走 MLX
# DLM：mlx-lm 尚不支援 diffusiongemma（已實測），走 llama.cpp fallback。
# 先照附錄 F 編好 llama-diffusion-cli、下載 GGUF，然後：
.venv/bin/python scripts/dlm_ar_bench.py --model dlm --engine llama \
    --llama-bin ~/Projects/llama.cpp/build/bin/llama-diffusion-cli \
    --gguf models/dlm-gguf/<檔名>.gguf
.venv/bin/python scripts/dlm_ar_bench.py --report      # 出 reports/week4/bench_ar_vs_dlm.md
# report 會偵測兩邊 engine 不同並自動加警語（MLX 4.50bit ≠ GGUF Q4_K_M）。
# 想要嚴謹的同 engine 比值：cp reports/week4/bench_ar.json reports/week4/bench_ar_mlx.json
# 之後，抓 AR 的 GGUF（HF 搜 gemma-4-26B-A4B-it-GGUF）用 llama-cli + --engine llama 重量 AR。

# 4.3 品質（2026-08-24 依實測改路）：
#   AR base 品質 = Step 3.4 的 moe26b_base_zh 那一輪（同模型同 config，跑一次共用，
#   不要另跑 ar26b_base_quick）。
#   DLM 品質（base）：llama-server 沒接 diffusion（send_error 實測），但
#   llama-diffusion-cli 有套 chat template 且輸出通順（dlm_0.log 實證）→
#   用逐題 CLI 驅動器在本機免費跑（每科 100 題約 2.3 hr，掛機即可）：
.venv/bin/python scripts/dlm_cli_eval.py --tag dlm26b_base_quick \
    --llama-bin ~/Projects/llama.cpp/build/bin/llama-diffusion-cli \
    --gguf models/dlm-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
    --limit 100 --allow-unparsed-probe
#   DLM 品質（tuned）：只能在 Colab（§6.2，bf16 80GB 級）；tuned vs base 的同 engine
#   對照用 Colab 的 base quick（§6.1），本機 CLI 的 base 是另一個 engine 欄位。
```

- [x] `bench_ar_vs_dlm.md` 完成：**DLM/AR 吞吐 0.26×**（14.3 vs 55.8 tok/s）——
      與官方 4× 方向相反、與 roofline 事前預測一致（M4 batch=1 算力飽和）；
      機制拆解：in-step parallel ~295 tok/s ÷ ~20.8 步/canvas ≈ 14.2 ✓
- [x] AR base 品質 = `moe26b_base_zh`（58.0%/13.3%）；DLM base 品質 =
      `dlm26b_base_quick`（49.0%/32.7%，llama.cpp CLI 驅動、thinking on，注腳必標）
- [x] llama.cpp fallback 全程實測：llama-server 無 diffusion 路徑（send_error）→
      bench 用 llama-diffusion-cli、品質用逐題 CLI 驅動器（`dlm_cli_eval.py`）

---

## Step 5：DLM 微調 — **已降級（2026-08-24 定案，硬體不可得）**

**歸因鏈（報告直接引用）**：DiffusionGemma 的 128 個 MoE 專家是 3-D 融合張量，
bitsandbytes 4-bit 無法量化（Week 3 已記錄、官方 notebook 證實）→ 微調必須 bf16
全量 ~52GB VRAM → Colab 可選的最大卡是 A100 **40GB**（無 80GB/H100 選項；
「大量 RAM」開關加的是系統 RAM 不是 VRAM）→ 官方明示 40GB 會 offload
「慢到不可用」→ **tuned DLM 記「硬體不可得，降級」**。

**對交付的影響**：proposal 的 4-checkpoint 表改為 **3 + 註記**（AR base / AR tuned=方案A /
DLM base）。AR vs DLM 的主線結論不受影響：吞吐對照（0.26×）與 DLM base 品質
（49.0% quick）都已在本機取得。DLM 的**可微調性**以官方 Sudoku 案例佐證
（base 1.5% → 微調後 89.5%）。`notebooks/week4_colab.ipynb` 保留為 ready-to-run
產物——future work：80GB 級硬體可得時（租用 A100 80GB 約 US$5/2hr）補第 4 格。

- [x] 決策記錄完成；Colab session 可關閉，不再花 units
- [x] `eval_dlm_base_quick` 由本機 `dlm_cli_eval.py` 取得（49.0% / 無法解析 32.7%）

---

## Step 6：彙整（半天）

**主表（proposal slide 9 的最終版）**——可比性規則直接寫進表：

| 模型 | 範式 | 微調 | TMMLU+ 嚴格 micro/macro | 無法解析 | 吞吐 tok/s（M4/4-bit） | 峰值 GiB（M4/4-bit） |
|---|---|---|---|---|---|---|
| gemma-4-26B-A4B | AR | 無 | Step 4.3 | | `bench_ar` | `bench_ar` |
| gemma-4-26B-A4B | AR | 方案 A 200 步 | Step 3.4 | | —（不重量） | — |
| diffusiongemma-26B-A4B | DLM | 無 | Step 4.3（本機）＋Colab 快速 | | `bench_dlm` | `bench_dlm` |
| diffusiongemma-26B-A4B | DLM | 200 步 | **Colab 快速**（標註 engine） | | —（Colab 欄另列） | — |

注腳必寫：①品質欄同機同 engine 內才可比（本機 MLX 一組、Colab 一組，分開標）；
②吞吐/記憶體只取本機 4-bit 單請求；③ DLM tuned 若降級，該格寫「額度降級，見 §5」。

**回答 proposal slide 9 的三個問題**（每題兩三句，寫進報告）：
1. 微調增益兩範式一致嗎？（AR 方案 A vs DLM 200 步，同資料同步數）
2. 擴散吞吐優勢在繁中成立嗎？代價是什麼？（bench 比值 + 品質差 + 記憶體差）
3. 峰值記憶體 vs Playbook 公式差多少？（DLM 的權重佔用應和 AR 幾乎同——同一組權重；
   差別在 canvas activations，把 `bench` 的 peak−load 差值拿去對公式）

**假設狀態總表**（報告附錄用，從 Week 3 的表延伸）：新增 P-W2a/b、P-E1/2/3、
P-A1/P-B1/P-AB 的判定結果；P2（FA on/off）、P1（bf16 基準線）若沒跑，標「未執行
（額度讓給 DLM 微調）」——這是有意識的取捨，寫明白。

---

## Step 7：最終報告（一天）

1. **`week4_執行總結.md`**：沿用 Week 3 格式（三句話總結 → 各 Step 數字與判定 →
   方法論教訓 → 主表）。
2. **`reports/week4/final_report_outline.md`** 是四週整合報告的骨架，把 `{{ }}`
   佔位符換成實測數字 → 存成 `final_report.md`（自留版）。
3. **PPTX**：把填好的 `final_report.md` 丟回來給我（或任何工具）生成
   `final_report.pptx`；每張 slide 對應 outline 的一節，數字表直接搬。
   結構刻意對齊 proposal 的 slide 順序，主管可以逐頁對「當初說要做什麼 vs 做到什麼」。

- [ ] 交付清單：`final_report.pptx`、`final_report.md`、`week4_執行總結.md`、
      `results/week4/`、`reports/week4/`、可複現指令（本手冊）

---

## 附：這週的降級路徑總表（出事不用現場想）

| 風險 | 降級 |
|---|---|
| Colab 選不到 A100 80GB / H100 | DLM tuned 記「硬體不可得」；base 品質用本機 `dlm_cli_eval.py`＋bench 已齊，報告註明 |
| mlx-lm 不支援 diffusiongemma | llama.cpp `llama-diffusion-cli`（bench）+ llama-server（評測）；量化格式差異要標註 |
| Unsloth diffusion API 對不上 | 以官方 Sudoku notebook 的 model/trainer cell 為準，超參數維持本設定 |
| 26B 方案 A/B OOM | `num_layers` 30→16（改口徑）；再不行方案 B 優先（參數更少） |
| 時間不夠 | 砍序：Step 1 thinking-off 兩輪 → 方案 A/B 完整評測 → `--gen`；**方案 E 與 AR/DLM baseline 不砍** |
| 方案 E 對帳 C 不過（bias 沒生效） | 先 `inspect_router_mlx.py --dump-modules` 看結構，修 `RouterBias` 的 candidates 判斷；不要硬跑 |

---

## 附錄 F：llama.cpp fallback 建置（DLM 專用）

mlx-lm 尚不支援 `diffusion_gemma`（Week 4 實測），DLM 的 bench 走 llama.cpp
的 diffusion runtime（PR #24423，Metal 可用）。

```bash
# F.1 取得原始碼（repo 很大，一定要 shallow；網路斷就用 zip 版）
cd ~/Projects
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch --depth 1 origin pull/24423/head:diffusion-gemma
git switch diffusion-gemma
#（git 一直斷線的替代：curl -L --retry 5 -o pr.zip \
#   https://codeload.github.com/ggml-org/llama.cpp/zip/refs/pull/24423/head && unzip pr.zip）

# F.2 編譯（Mac 的 Metal 預設開啟）
cmake -B build
cmake --build build -j --config Release --target llama-diffusion-cli llama-server

# F.3 下載 GGUF Q4_K_M（~16 GB）
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF --include "*Q4_K_M*" \
    --local-dir ~/Projects/ultrascale-lab/models/dlm-gguf
```

之後回 Step 4.2 用 `--engine llama` 跑 bench。

**品質評測（已實測判定，2026-08-24）**：這個 PR 的 llama-server **沒接 diffusion
解碼路徑**——請求會被 `send_error: the current context does not logits computation`
直接拒絕。所以 DLM 品質一律走 Colab notebook（§6.1 base、§6.2 tuned，同 session
同 config，內部可比性最乾淨）；本機 llama.cpp 只負責吞吐/記憶體（llama-diffusion-cli）。

注意事項：
- `--engine llama` 的 `-p` 模式沒套 chat template（量吞吐夠用，品質不能用它量）。
- 峰值記憶體量法不同（子行程 RSS vs MLX metal API），engine 欄要照實標。
- GGUF Q4_K_M 與 MLX 4-bit（4.50 bit/param）位元組成不同 —— W2 教訓，不可直接並列。
