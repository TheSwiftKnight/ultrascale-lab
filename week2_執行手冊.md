# Week 2 執行手冊：模型換血 → 本機微調 → 消融 → 對照

> 對應提案 `ultrascale-proposal_1.pptx` 第 7 頁 Week 2–3 里程碑，
> 但已依 **Week 1 會議後的兩項變動**改寫：
>
> | 原計畫 | 改成 | 影響 |
> |---|---|---|
> | GPT-OSS 20B | **Gemma 4** | 資料管線的 template 全換、記憶體公式常數全換 |
> | 租雲端 CUDA 卡為主 | **本機 M4 Pro 為主**，只短時租卡驗證 CUDA-only 的項目 | 訓練框架從 PyTorch/Unsloth 換成 **MLX** |

---

## 這份手冊怎麼用

每個 Step 都長這樣：

- **做什麼** — 一句話目標
- **跑什麼** — 可以直接複製貼上的指令
- **讀哪裡** — 這一步對應 Playbook 的哪一章
- **驗收** — 看到什麼才算過關
- **✅ 我已經幫你做好的** / **👉 你要操作的**

Step 0–1 是準備，Step 2–4 是換血，Step 5–9 是主線實驗，Step 10 是租卡，Step 11 是收尾。
**照順序做**，中間卡住就跳到最後的「常見坑」。

---

## 0. 先講清楚：為什麼是「E4B 主線 + 26B MoE 對照」

你的機器是 **M4 Pro / 24GB 統一記憶體**。這個數字決定了一切，先把算術攤開：

### 先講一個硬限制：12B 用不了

Gemma 4 12B Unified 的 `model_type` 是 **`gemma4_unified`**，
mlx-lm 0.31.3（最新版）**不支援**，載入直接噴：

```
ValueError: Model type gemma4_unified not supported.
```

這是 mlx-lm 的已知缺口（[issue #1481](https://github.com/ml-explore/mlx-lm/issues/1481)，
2026-07 開著、無人認領、沒有 PR）。**12B 是唯一用 `gemma4_unified` 的變體**；
E4B / 26B-A4B / 31B 都是 `gemma4`，全部能跑。所以 dense 那一側改用 **E4B**。

### 兩個模型（規格已用 `config.json` 逐項核對，`--verify-config` 全綠）

| | Gemma 4 E4B | Gemma 4 26B-A4B |
|---|---|---|
| 架構 | **dense**，42 層 | **MoE**，30 層，128 專家取 8 |
| 總參數（公式算） | 7.46B（官方 8B with embeddings） | 25.23B（官方 25.2B，誤差 **0.1%**） |
| **每 token 實際用到** | **3.97B**（非嵌入） | **3.82B**（active） |
| 4-bit 權重 | **3.7 GiB** | 12.5 GiB |
| bf16 權重 | **13.9 GiB**（本機塞得下！） | 47.0 GiB |
| 微調總預算（seq2048, bs1, 開檢查點） | **8.5 GiB ✅** | 17.2 GiB ⚠️ |

**換成 E4B 反而讓對照更乾淨。** E4B 的非嵌入參數 3.97B 和 26B-A4B 的 active 3.82B
幾乎一樣 —— 兩者**每 token 的計算量相當**，總參數卻差 3.4 倍。
於是「MoE 到底買到了什麼」變成一個可以乾淨量測的問題：
同樣的每 token 計算量，多花 8.8 GiB 記憶體養 25B 參數，換到多少準確率？

**另一個意外收穫**：E4B 的 bf16 只要 13.9 GiB，
**P1 / P4（bf16 vs 4-bit 的記憶體與掉點）不必租卡就能在本機做完** ——
那本來是 Step 10 租卡清單上最重要的一項。

26B-A4B 的 25.2B total / 3.8B active 對上 GPT-OSS 的 20.9B / 3.6B，
連 MoE 佔全模型 90% 參數這件事都一樣 ——
Week 1 建立的 ch06 分析、roofline 論證、E1–E6 假設**全部可以搬過來**。
但它在 24GB 上微調太貼邊（17.2 GiB，要手動放寬 GPU wired limit），所以只做推論。

Week 2 拆成兩條線：

```
主線 A（本機，寬裕）
  Gemma 4 E4B dense · 4-bit LoRA（8.5 GiB）
  → 微調 + 四組消融 + 微調前後 TMMLU+ + bf16/4-bit 對照
     ← 這條線交付「Week 2 有做出東西」，而且 P1/P4 也在這裡完成

主線 B（本機，只做推論）
  Gemma 4 26B-A4B MoE · 4-bit（12.5 GiB）
  → 載入驗證 + roofline + 路由分析   ← 接住 Week 1 的 ch06 敘事

租卡 2–4 小時（48GB）—— 清單比原本短了
  → torch profiler 的逐項拆解、CUDA 版梯度檢查點消融
     以及（有餘力的話）26B-A4B 的 LoRA
```

**多出一組 dense vs MoE 對照**，正是 ch06 最後一個假設 E6
（「EP／MoE 在小規模情境下划不划算」）要回答的問題。

> 如果 mlx-lm 之後補上 `gemma4_unified`，或你拿到更大記憶體的機器，
> 把主線 A 換成 12B 或 26B-A4B 即可 ——
> `predict_memory_gemma.py` 的 `CONFIGS` 加一筆就好，其餘腳本不用動。

---

## Step 0：清除 GPT-OSS 相關資源

**做什麼** — 把之後不會再用到的 GPT-OSS 資源清掉，空出磁碟給兩個 Gemma 模型（約 23 GB）。

### ✅ 我已經幫你做好的

repo 內的 GPT-OSS 專屬產物已移到 `_to_delete/gptoss_week1/`（78 MB）：

| 移走的東西 | 為什麼 |
|---|---|
| `data/train`, `data/val` | 用 GPT-OSS tokenizer + Harmony template 產生的，必須重做 |
| `scripts/predict_memory.py` | 寫死 GPT-OSS config，已由 `predict_memory_gemma.py` 取代 |
| `scripts/prepare_data.py` | 同上，已由 `prepare_data_gemma.py` 取代 |
| `scripts/verify_load.py` | 已由 `verify_load_mlx.py` 取代 |
| `scripts/inspect_router.py` | 走 HF+CUDA，本機跑不動，已由 `inspect_router_mlx.py` 取代 |
| `scripts/_build_notebook.py` | 產生 Harmony 版 notebook 的工具 |
| `configs/baseline_lmstudio.yaml` | 打 gpt-oss 端點 |
| `reports/` 全部舊產物 | memory_prediction / load_verification / data_stats / 分布圖 / 樣本 |
| `logs/*.log` | Week 1 的評測 log |
| `notebooks/01_data_pipeline.ipynb` | Harmony 版 |
| `week1_執行手冊.md` | 開頭自己標註「已被 week1_執行總結.md 取代」 |

**刻意保留**（這些不是 GPT-OSS 專屬，之後還要用）：

- `week1_執行總結.md` — Week 1 的成果紀錄，是最終報告方法論章節的依據
- `notes/ch01,02,06,10.md` — 公式部分與模型無關；只有「🧪 對照我的實驗」小節要改寫（Step 11）
- `Eval/`、`datasets/ikala__tmmluplus/`、`ultrascale-proposal_1.pptx`

### 👉 你要操作的

我碰不到你的 HF 快取和 LM Studio（那不在連進來的資料夾裡），所以寫成腳本讓你跑：

```bash
cd ~/Projects/ultrascale-lab

# 先看要刪什麼（不動手）
bash scripts/cleanup_gptoss.sh

# 確認沒問題後真的刪
bash scripts/cleanup_gptoss.sh --yes
```

它會處理：HF 快取裡的 `models--openai--gpt-oss-20b` 與 `models--mlx-community--gpt-oss-*`、
Xet 區塊快取、LM Studio 裡的 gpt-oss 模型，最後才刪 `_to_delete/`。

> ⚠️ **`_to_delete/` 建議等 Step 4 跑完、拿到新的 `data/train` 之後再刪。**
> 腳本會問到最後才動它，你也可以先只跑前面幾項。
> LM Studio 若是用 GUI 下載的，腳本可能找不到目錄 —— 開 LM Studio →
> My Models → 找 gpt-oss → 右鍵 Delete。

**驗收** — `df -h` 應該多出 15–25 GB。

---

## Step 1：環境與模型下載

**做什麼** — 把工具鏈從 PyTorch/Unsloth 換成 MLX，並掛著下載兩個模型。

**跑什麼**

```bash
cd ~/Projects/ultrascale-lab
source .venv/bin/activate

# 1-1 套件（transformers 要 v5+ 才認得 gemma4 的 chat template）
uv pip install -U mlx mlx-lm "transformers>=5.0" "datasets>=2.19" \
                  datasketch matplotlib pyyaml twinkle-eval

# 1-2 環境驗收（缺什麼會直接告訴你補的指令）
python scripts/verify_env_week2.py --check-models

# 1-3 掛著下載模型
#     ⚠️ 一定要加 HF_HUB_DISABLE_XET=1 —— 見下方「Xet 傳輸」說明
#     ⚠️ 注意 e4b 是小寫，且**不要**下載 gemma-4-12B（mlx-lm 載不動）
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-e4b-it-4bit --max-workers 4        # 5.2 GB
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-26B-A4B-it-4bit --max-workers 4    # 15.4 GB
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-e4b-it-bf16 --max-workers 4        # 15.9 GB（P1/P4 用）

# 若你已經下載過 12B，可以刪掉（mlx-lm 0.31.3 載不動）
hf cache delete mlx-community/gemma-4-12B-it-4bit

# 1-4 為 26B 放寬 GPU wired limit（重開機後失效，不會永久改動系統）
sudo sysctl iogpu.wired_limit_mb=20480
```

**讀哪裡** — 不用讀書，這一步純環境。

**驗收** — `verify_env_week2.py` 全綠；`df -h` 剩餘 > 25 GB。

> **`--check-models` 會幫你抓兩件事**：E4B / 26B 是否下載完成，以及快取裡有沒有載不動的 gemma-4-12B。

> **關於 Xet 傳輸（Week 1 踩過的坑，這裡一定會再遇到）**
>
> HF 的 Xet 協定會把大檔切成很多小塊分別請求再重組。在慢速或不穩定的連線上
> 重組會失敗，症狀是下載到一半噴：
>
> ```
> RuntimeError: Task error: File reconstruction error:
> CAS Client Error: Format error: I/O error: error decoding response body
> ```
>
> 26B 那個 15.4 GB 的檔特別容易中招（E4B 的 4-bit 只有 5.2 GB，通常會過）。
> 解法是關掉 Xet 退回一般 HTTPS，**已下載的部分會續傳，不用重來**：
>
> ```bash
> HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-26B-A4B-it-4bit --max-workers 4
> ```
>
> `--max-workers 4`（預設 8）在慢速連線上比較穩。若連線速度是 4–5 MB/s，
> 15.4 GB 大約要一小時，掛著去做 Step 2 的讀書。
>
> `prepare_data_gemma.py` 已經內建這個自動退回機制（`--no-xet`），
> 但 `hf download` 是 CLI，要自己加環境變數。

> **另一種長得很像、但成因不同的下載失敗：proxy 的 TLS 被切斷**
>
> ```
> httpx.ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]
>     EOF occurred in violation of protocol (_ssl.c:1016)
> ```
>
> 看 traceback 有沒有經過 `httpcore/_sync/http_proxy.py` —— 有的話代表你的連線
> 走 HTTP proxy（公司網路 / VPN 常見），而 proxy 在建立 TLS 時把連線掐掉了。
> **這和 Xet 無關**，加 `HF_HUB_DISABLE_XET=1` 也治不好。
>
> 主因通常是併發太多：`--max-workers 4` 等於同時要 proxy 開 4 條 TLS。
> 解法是降到 1 條、拉長逾時，並包一個續傳迴圈（`hf download` 本來就會續傳）：
>
> ```bash
> until HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 \
>       hf download mlx-community/gemma-4-e4b-it-bf16 --max-workers 1; do
>   echo "斷線，10 秒後續傳…"; sleep 10
> done
> ```
>
> `huggingface_hub` 內建的 backoff 只重試 5 次、最多等 8 秒，
> 連線很不穩時會用完；外面包一層 `until` 迴圈才撐得住。

> **關於 `iogpu.wired_limit_mb`**：macOS 預設只讓 GPU 用約 2/3 的統一記憶體
> （24GB 機器 ≈ 16 GiB）。E4B 的 4-bit 用不到這個，但 26B-A4B 的 12.5 GiB 權重、
> E4B 的 bf16（13.9 GiB）、以及消融裡「不開檢查點」那一格（16.2 GiB）都會頂到天花板。
> 調到 20480（20 GiB）是安全值；**不要超過 21504**，
> 再上去系統會開始換頁，反而更慢甚至當掉。還原：`sudo sysctl iogpu.wired_limit_mb=0`。

---

## Step 2：讀書（這週要讀哪幾章）

Week 1 已讀完 ch01 / ch02 / ch06 / ch10。Week 2 補的是**方法論**那兩章，
不是新的機制章節 —— 因為這週的重點是「怎麼設計消融實驗」而不是「多學一個平行策略」。

### 必讀

| 順序 | 章節 | 為什麼這週要讀 | 讀完要產出 | 時間 |
|---|---|---|---|---|
| 1 | **`ch08.md` 尋找最佳訓練配置** | 這章的三步驟決策流程（① 先塞進記憶體 → ② 湊到目標 global batch size → ③ 最佳化吞吐）**就是 Step 8 消融實驗的劇本**。不讀這章，消融會變成亂試參數 | 把三步驟對應到你的變因清單：ckpt 開關 / seq / bs / LoRA 層數，寫在 `notes/ch08.md` | 2h |
| 2 | **`appb.md` 分散式訓練效能分析** | Step 10 租卡時要用 profiler。這章講怎麼看 trace、怎麼分辨「計算慢」和「等記憶體」 | 一份 profiler 操作備忘（要抓哪幾個欄位） | 1.5h |

**讀 ch08 時特別注意**：課本假設你有多張卡可以調 DP 大小。你只有一台單機，
所以「湊到 global batch size」那一步在你這裡只能靠**梯度累積**達成。
這個對應關係要自己補上 —— 也是報告裡「課本情境 vs 我的情境」的一個具體落差。

### 選讀（有空檔再看）

| 章節 | 讀法 |
|---|---|
| `ch00.md` 導論 | 30 分鐘略讀，補全書鳥瞰 |
| `appc.md` 典型尺度 | 20 分鐘，把數量級記下來對照實測 |

### 明確不讀

ch03 張量平行、ch04 上下文平行、ch05 管線平行、ch09 GPU 架構 —— 單機 LoRA 用不到。

---

## Step 3：重算記憶體預測（換模型後的第一件事）

**做什麼** — Week 1 的預測表整張作廢（那是 GPT-OSS 的常數），用 Gemma 4 的
`config.json` 重推一次。這張表是 slide 4 第一列「實測 vs 公式落差」的左半邊。

**跑什麼**

```bash
# 兩個模型都算，順便上網抓 config.json 核對本檔寫死的常數
python scripts/predict_memory_gemma.py --verify-config

# 換不同 seq 看 logits 怎麼長（Gemma vocab 262K，這是本週最大的陷阱）
python scripts/predict_memory_gemma.py --seq 1024
python scripts/predict_memory_gemma.py --seq 4096

# 只看 MoE，且把 LoRA 掛到 expert 上（H6）
python scripts/predict_memory_gemma.py --model gemma4-26b-a4b --lora-target all

# 只看 E4B
python scripts/predict_memory_gemma.py --model gemma4-e4b
```

**讀哪裡** — ch01（記憶體剖析）、ch10（混合精度）。這兩章 Week 1 讀過，
這裡只是把同樣的公式換一組常數重算。

**驗收** — `--verify-config` 全部 ✅；`reports/memory_prediction_gemma.md` 產出。

### 換模型後最重要的三個數字（先記住，Step 8 要驗）

1. **E4B 有 38% 的參數在 Per-Layer Embeddings 上**。E4B 是為端上裝置設計的，
   用 PLE 把知識搬到查表裡：262,144 × (42 層 × 256) = **2.82B 參數**，
   而語言主幹只有 3.97B。這解釋了官方為什麼標「4.5B effective / 8B with embeddings」——
   **兩個數字都對，只是算的東西不同**。報告裡要講清楚你用的是哪一個。

2. **logits 是第二大戶**。Gemma 4 的 vocab 是 262,144，
   seq=2048 時光 logits 就是 **3.00 GiB**，seq=4096 時 **6.00 GiB**。
   對 E4B 而言，開了檢查點之後 logits（3.00）比權重（3.7）幾乎一樣大，
   比活化（0.63）大 5 倍。Week 1 的 H4「logits 是被低估的大戶」在這裡格外明顯。

3. **dense 的活化比 MoE 大**。E4B 在 seq=2048、不開檢查點時活化 **8.34 GiB**，
   26B MoE 只要 **6.11 GiB** —— 儘管 MoE 的總參數是它的 3.4 倍。
   因為 MoE 每 token 只過 8/128 個專家，而 dense 的 FFN intermediate 是 10,240（= 4h）
   每層都要全存。**「參數多的那個活化反而小」** 是這週最反直覺、也最值得寫進報告的一句話。

### 👉 這一步會逼出一個設定決定

E4B 在 24GB 上的完整預算（bs=1）：

| seq | bs | 精度 | 檢查點 | 活化 | logits | 合計 | 判定 |
|---:|---:|---|---|---:|---:|---:|---|
| 512 | 1 | 4-bit | 開 | 0.16 | 0.75 | 5.7 GiB | ✅ |
| 2048 | 1 | 4-bit | 開 | 0.63 | 3.00 | **8.5 GiB** | ✅ ← 主線設定 |
| 4096 | 1 | 4-bit | 開 | 1.25 | 6.00 | 12.1 GiB | ✅ |
| 512 | 8 | 4-bit | 開 | 1.28 | 6.00 | 12.1 GiB | ✅ |
| 2048 | 1 | 4-bit | **關** | 8.34 | 3.00 | 16.2 GiB | ⚠️ 需 wired limit |
| 2048 | 1 | **bf16** | 開 | 0.63 | 3.00 | **18.7 GiB** | ⚠️ 需 wired limit |

**和原本的 12B 比，E4B 讓整個 Week 2 寬鬆很多**：

- 主線只用 8.5 GiB，還有一半記憶體可以同時開評測 server。
- **H3（梯度檢查點）可以在正式訓練的同一個 seq=2048 上量**，
  不必為了讓兩端都跑得起來而壓低 seq —— 消融數字直接對得上訓練數字。
- seq 掃到 4096、bs 掃到 8 都不會 OOM，H4/H5 的曲線更完整。
- **bf16 跑得起來（18.7 GiB）** → P1/P4 不用租卡。

要更大的 effective batch 仍然可以靠**梯度累積** —— 這是 ch08「湊到目標 global
batch size」那一步在單機情境下的手段，也是課本假設（多卡 DP）在你這裡的落差。

---

## Step 4：資料管線換 template

**做什麼** — 清理／去重的邏輯完全不動（同樣的 seed、門檻、MinHash 參數），
只換 Step 4 之後的 template 與 tokenizer。

**跑什麼**

```bash
python scripts/prepare_data_gemma.py            # 約 3–5 分鐘
# 連線不穩就加 --no-xet；想快速試跑加 --no-minhash
```

**讀哪裡** — 不用讀書。

**驗收 —— 這一步有一個硬性驗收點**

前三步的數字必須和 Week 1 **完全相同**：

```
50,000 → 49,984（清理）→ 49,968（精確去重）→ 49,965（MinHash）
```

不一樣就代表管線被改壞了，先查清楚再往下走。

### 變了什麼

| | Week 1（GPT-OSS） | Week 2（Gemma 4） |
|---|---|---|
| template | Harmony | Gemma 4 canonical |
| 思考通道 | `<\|channel\|>analysis` / `final` | `<\|channel>thought ... <channel\|>` |
| 開關 | `reasoning_effort="medium"` | `enable_thinking=True` |
| assistant 欄位 | `thinking` | **`reasoning`** |
| vocab | 201,088 | 262,144 |
| 輸出 | HF dataset | HF dataset **＋ MLX jsonl** |

模板行為我已經用實際的 `chat_template.jinja` 渲染驗證過，長這樣：

```
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
台灣的健保制度如何運作？<turn|>
<|turn>model
<|channel>thought
先釐清問題：使用者問的是制度運作機制…
<channel|>全民健保由衛福部主管…<turn|>
```

腳本會在跑全量前先用一筆探針驗證這些標記都在，不對就直接中止 ——
免得跑完兩分鐘才發現格式錯。

### ⚠️ train/eval 一致性規則（Gemma 4 版的「reasoning_effort 要同檔位」）

`enable_thinking` 控制的是**系統回合裡那個 `<|think|>` token**。
assistant 的 `reasoning` 欄位不管開不開都會渲染。所以：

> **訓練資料用 `enable_thinking=True`，評測設定檔就必須也開。**
> 不一致的話模型看到的前綴不同，微調前後的比較會失真。

兩份評測 config 都已經寫好 `enable_thinking: true`，不要改。

### 👉 這一步結束時，你要做一個決定

腳本會印出 Gemma tokenizer 下的 token 長度分布，以及每個候選 `max_seq_len` 的涵蓋率：

```
max_seq_len= 1024：涵蓋 xx.xx%（截斷 xxx 筆）
max_seq_len= 1536：涵蓋 xx.xx%（截斷 xxx 筆）
max_seq_len= 2048：涵蓋 xx.xx%（截斷 xxx 筆） ← 建議
```

**把達到 99% 涵蓋率的那個值填進 `configs/lora_gemma4_e4b.yaml` 的 `max_seq_length`。**

Week 1 在 GPT-OSS tokenizer 下算出 2048（p99=2033）。
Gemma 的 262K vocab 對中文的壓縮率不同，**這個數字很可能會變** ——
這是 Week 2 唯一需要重新決定的超參數，也是一個乾淨的小發現：
「同一批資料，換 tokenizer 後 p99 token 長度變了 X%」。

---

## Step 5：載入驗證 + roofline（兩個模型）

**做什麼** — 拿到第一組實測數字，驗 H1（權重記憶體預測誤差 <10%）與
E2（用記憶體頻寬 roofline 證明 MoE 只讀 active 參數）。

**跑什麼**

```bash
# 兩個 4-bit 模型依序跑，寫出對照表
python scripts/verify_load_mlx.py --both

# P1：E4B 的 bf16 對照（13.9 GiB，本機跑得起來 —— 這是換 E4B 換來的）
python scripts/verify_load_mlx.py --model mlx-community/gemma-4-e4b-it-bf16

# 若你的機器不是 M4 Pro，頻寬要改（M4=120, M4 Pro=273, M4 Max=546 GB/s）
python scripts/verify_load_mlx.py --both --bandwidth 273
```

**讀哪裡** — ch06（EP／MoE）、ch01。都是 Week 1 讀過的。

**驗收**

- H1：實測峰值 vs 預測（E4B 3.7 GiB / 26B 12.5 GiB）誤差 <10%
- E2：26B 的實測 tok/s 應該落在「MoE 理論上限的 70–85%」，
  且明顯高於「假設 dense 時的理論上限」
- `reports/load_verification_gemma.md` 產出

> Week 1 在 GPT-OSS 上量到：預測 12.81 GiB / 實測峰值 13.87 GiB，誤差 +8.3%；
> 吞吐 56.6 tok/s = MoE 理論上限的 77%、dense 上限的 2.62 倍。
> **Gemma 的數字可以直接和這組並排** —— 同一台機器、同一套方法。

### 這一步的看點

E4B dense 每 token 要讀 **全部** 3.7 GiB 權重（其中 PLE 查表只讀命中的那幾列，
實際更少），26B MoE 只讀約 20%（約 2.5 GiB）。
兩者的每 token 讀取量落在同一個量級 —— 所以**吞吐應該接近**，而這正是 E4B/26B 這組配對的意義：計算量對齊，差別只在總參數與記憶體。如果實測真是這樣，
E2 就從「算術上 active 比較小」升級成「物理上真的只搬了那麼多位元組」。

---

## Step 6：Baseline 評測（微調前）

**做什麼** — 拿到微調前的 TMMLU+ 準確率，這是 slide 9 表格的第一列。

**跑什麼**

開兩個終端機視窗。

終端機 A（起 server，不要關）：
```bash
source .venv/bin/activate
mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --host 127.0.0.1 --port 1234
```

終端機 B：
```bash
source .venv/bin/activate

# ⚠️ 先做科目子集資料夾 —— Twinkle Eval 的 dataset_paths 只吃目錄，不吃單一檔案
python scripts/make_eval_subset.py

# 確認端點活著，順便看 model id（要和 config 的 model.name 一致）
curl -s http://127.0.0.1:1234/v1/models | python3 -m json.tool

twinkle-eval --validate --config configs/eval_gemma4_e4b_base.yaml
twinkle-eval --dry-run  --config configs/eval_gemma4_e4b_base.yaml
twinkle-eval           --config configs/eval_gemma4_e4b_base.yaml
```

> **為什麼要多這一步**：Twinkle Eval 2.8 的 `validate_dataset_path()` 會做
> `os.path.isdir()` 檢查，把 `.../geography_of_taiwan.parquet` 這種單檔路徑直接擋掉：
>
> ```
> ❌ 資料集 datasets/ikala__tmmluplus/geography_of_taiwan.parquet：
>    Dataset path is not a directory
> ```
>
> 而它的 `find_all_evaluation_files()` 是用 `os.walk` 掃整個目錄，
> 所以指向 `datasets/ikala__tmmluplus/` 會一次跑滿 66 科。
> **要只跑幾科，就得做一個只放那幾科的目錄** ——
> `make_eval_subset.py` 用相對路徑 symlink 做到（不佔額外磁碟），
> 並在最後模擬一次 `os.walk` 驗收，確認 Twinkle Eval 真的讀得到。
>
> 它會產生兩個子集，換科目只要改 config 裡那一行：
>
> | 目錄 | 科目數 | 用途 |
> |---|---:|---|
> | `datasets/subsets/tmmluplus_smoke` | 3 | 試水溫，確認流程與單科耗時 |
> | `datasets/subsets/tmmluplus_tw10` | 10 | 正式報告用（Week 1 定案的台灣知識子集） |

**驗收** — 三科（台灣地理／台語／三民主義）跑完，拿到準確率。
記錄**單科耗時**，用它推估十科要多久，再決定要不要擴大到正式版的十科。

> config 裡十科的清單已經註解在下面，確認時間可接受就把註解拿掉。
> 選取原則沿用 Week 1：台灣在地情境 + 繁中語言能力，避開純理工科目。

---

## Step 7：LoRA 微調（主線）

**做什麼** — 這週的核心交付物。

**跑什麼**

```bash
# 先確認 max_seq_length 已依 Step 4 的結果改好
mlx_lm.lora --config configs/lora_gemma4_e4b.yaml

# 中斷後恢復
mlx_lm.lora --config configs/lora_gemma4_e4b.yaml \
            --resume-adapter-file out/lora-e4b/adapters.safetensors

# 訓練完融合成完整權重（評測要用）
mlx_lm.fuse --model mlx-community/gemma-4-e4b-it-4bit \
            --adapter-path out/lora-e4b \
            --save-path out/gemma4-e4b-tw
```

**讀哪裡** — ch08 的三步驟決策流程（Step 2 已讀）。

**驗收** — train loss 下降且 val loss 沒有反轉；`out/lora-e4b/adapters.safetensors` 產出。

### 開跑前，先跑 30 步確認記憶體與速度

```bash
mlx_lm.lora --config configs/lora_gemma4_e4b.yaml --iters 30 --steps-per-report 5
```

看那行 `Peak mem X.XXX GB`，和 `predict_memory_gemma.py` 算的 **8.5 GiB** 比對
（注意 mlx 印的是 GB 十進位，8.5 GiB = 9.1 GB）。
差太多先查清楚再跑滿 1000 步 —— 這也順便完成了 H1 在**訓練**情境下的驗證
（Step 5 驗的是推論情境）。

從 30 步的 `It/sec` 推估 1000 步要多久。E4B 只有 4B 級的計算量，在 M4 Pro 上
應該比 12B 快得多，1000 步估 10–25 分鐘。

---

## Step 8：消融實驗（這週最有價值的部分）

**做什麼** — 把 ch01 的四條假設從「預測」變成「實測 vs 預測的落差表」。

**跑什麼**

```bash
# H3 梯度檢查點的取捨
python scripts/run_ablation.py --suite checkpoint

# H4 logits 隨 seq 線性成長
python scripts/run_ablation.py --suite seqlen

# H5 活化隨 batch size 的成長律
python scripts/run_ablation.py --suite batch

# H6 LoRA 掛的層數
python scripts/run_ablation.py --suite lora

# 一次全跑（約 30–60 分鐘）
python scripts/run_ablation.py --suite all --iters 40
```

**讀哪裡** — **ch08 的三步驟決策流程就是這一步的劇本**，跑之前務必已經讀完。

**驗收** — `reports/ablation_*.md` 四份，每份都有「預測 vs 實測」的落差。

### 這四條假設分別在驗什麼

| # | 假設 | 預測值（來自 Step 3） | 怎麼判定 |
|---|---|---|---|
| **H3** | 開 full checkpointing，活化下降 >85%，step time 增加 30–40% | seq2048：8.34 → 0.63 GiB（**−92%**） | `--suite checkpoint` 的兩列相減 |
| **H4** | logits 隨 seq 線性成長，是被低估的大戶 | seq 512→4096：0.75 → 6.00 GiB | `--suite seqlen` 的「logits 理論增量」欄 |
| **H5** | 活化隨 bs 線性；隨 seq **接近線性**而非 ch01 說的平方 | — | `--suite batch` + `--suite seqlen` 兩張表 |
| **H6** | 可訓練參數隨掛載層數線性；優化器記憶體相對權重仍是雜訊 | 4/16/42 層 → 約 0.9M/3.5M/9.1M | `--suite lora` |

> **換成 E4B 之後，H3 可以在正式訓練的同一個 seq 上量**：seq=2048 時
> 不開 16.2 GiB / 開 8.5 GiB，兩端都跑得起來。
> 原本用 12B 時「不開檢查點」在 seq=2048 是 25.3 GiB 直接 OOM，
> 只能壓到 seq=512 去量，消融數字和訓練設定對不上。
> **設計消融時先確認兩端都跑得起來** —— 這正是 ch08 三步驟第一步「先塞進記憶體」
> 在實務上的意思，而換模型剛好讓這一步變輕鬆。
>
> 跑消融前先放寬上限（「不開檢查點」那格是 16.2 GiB，超過預設的 ~16 GiB）：
> `sudo sysctl iogpu.wired_limit_mb=20480`

**H5 是最值得寫的一條**。ch01 的公式有個 $\frac{5 n_{heads} \cdot seq}{h}$ 平方項，
但 E4B 有 35/42 層是視窗只有 512 的滑動注意力，加上 Flash Attention 不具現化
S/P 矩陣 —— 所以實測應該接近線性。**「課本公式在我的架構上不成立，而且我知道為什麼」**
比「我驗證了課本公式」有價值得多。

### 順便驗一條 Week 1 沒有的：dense vs MoE 的活化差異

Step 3 算出 E4B dense 的活化（8.34 GiB）比 26B MoE（6.11 GiB）大 36%，儘管 MoE 的總參數是它的 3.4 倍。
如果你在租卡時也跑了 26B 的消融，這一組對照就完整了。跑不了就用預測值，
在報告裡明確標示「26B 這一欄是預測值，未實測」。

---

## Step 9：微調後評測 + 路由偏移

**做什麼** — 補完 slide 9 的表格，並拿到 E4／E5 這兩條「超出 Playbook 範圍」的結果。

**跑什麼**

```bash
# 9-1 微調後評測（server 換成融合後的權重）
mlx_lm.server --model out/gemma4-e4b-tw --host 127.0.0.1 --port 1234
twinkle-eval --config configs/eval_gemma4_e4b_tuned.yaml

# 9-2 MoE 路由分析（26B-A4B）
#     第一次一定先看結構，確認 router 模組叫什麼
python scripts/inspect_router_mlx.py --dump-modules

#     E3 負載均衡 + E4 中英路由差異
python scripts/inspect_router_mlx.py --compare-lang --save reports/router_before.json

#     E5 微調前後偏移（需要 26B 的 adapter，跑不到就留給 Step 10）
python scripts/inspect_router_mlx.py --adapter out/lora-26b \
    --save reports/router_after.json --compare-with reports/router_before.json
```

**驗收**

- 微調前後的準確率差值（**兩份 config 除了 `model.name` 以外必須完全一樣**，
  否則差值就不是微調造成的）
- `reports/router_before.json` 產出，E3 判定有結論

### E4 是這個專案最有研究價值的一條

E4 問的是：**同樣語意、不同語言的 prompt，在 MoE 裡走的專家是不是不一樣？**

如果平均 KL 散度顯著大於 0，就代表「語言」在 MoE 內部是一個**可辨識的路由特徵**。
這直接連到提案 slide 10 的延伸研究方向 B（語言變體作為可控制屬性）——
建議在報告最後單獨給一頁。

`--dump-modules` 是為了保險：我沒辦法在這裡驗證 mlx-lm 對 gemma4 MoE 的模組命名，
所以腳本會自動找名字含 `router`/`gate` 的模組；找不到就用 `--router-name <子字串>` 指定。

---

## Step 10：租卡 2–4 小時（補 CUDA-only 的項目）

**做什麼** — MLX 做不到的事情，用一張 48GB 的卡一次補完。

**租什麼** — RunPod 或 Vast.ai 的 **A6000 / L40S / A40（48GB）**，約 US$0.4–0.8/hr。
2–4 小時、總計約 US$2–3。

**讀哪裡** — `appb.md`（Step 2 已讀），上卡前再翻一次 profiler 那節。

**清單**（依優先序，時間不夠就砍後面）

| # | 項目 | 為什麼非 CUDA 不可 | 對應假設 |
|---|---|---|---|
| 1 | `torch.cuda.max_memory_allocated()` + PyTorch profiler 量 E4B LoRA 的記憶體組成 | MLX 只給總峰值，給不出「權重／活化／優化器」的**逐項拆解** | H1–H4 的右半邊 |
| 2 | CUDA 版的梯度檢查點消融，和 Step 8 的 Metal 版並排 | 證明結論跨後端成立，不是 MLX 的特例 | H3 |
| 3 | 開 / 關 Flash Attention 的直接對照 | MLX 沒有這個開關 | **P2** |
| 4 | 26B-A4B 的 LoRA（本機跑不動的那條） | 17.2 GiB 太貼邊 | E5 的前置 |
| 5 | 若租到 2 卡：ZeRO-1/2/3 與 DP 實測 | 單機不可能 | **D1–D5** |

> **原本排第 2 的「bf16 vs 4-bit 準確率對照」已經移回本機**（Step 5 + Step 7）——
> E4B 的 bf16 只要 13.9 GiB，訓練預算 18.7 GiB，24GB 機器放寬 wired limit 就跑得動。
> 這是換成 E4B 最實際的好處：租卡清單上最重要、也最花時間的那一項不必付錢了。

> **可比性務必寫進報告**：準確率跨硬體可比；吞吐與峰值記憶體**不可比**。
> Metal 的數字放一欄、CUDA 的數字放另一欄，不要混在同一欄取平均。

**驗收** — 至少完成 1 和 2。
至於主管最可能問的「bf16 vs 4-bit 掉幾分」，Step 7 已經在本機做完了，可以先答。

---

## Step 11：收尾與對齊

### 11-1 改寫章節筆記的「🧪 對照我的實驗」小節

`notes/ch01.md`、`ch02.md`、`ch06.md`、`ch10.md` 的**上半（課本內容整理）不用動**，
只有下半的實驗對照要換成 Gemma 的數字。對照表：

| 檔案 | 要換掉的 | 換成 |
|---|---|---|
| `ch01.md` | 已改成 Gemma 4 版 | 把 dense 那半從 12B 換成 **E4B**：7.46B/3.97B、4-bit 3.7 GiB、活化 8.34/0.63 GiB、LoRA 9.1M |
| `ch06.md` | 已改成 Gemma 4 版 | 只需把 dense 對照組的數字從 12B 換成 E4B |
| `ch10.md` | 已改成 Gemma 4 版 | 12B 欄換成 E4B：bf16 13.9 / 4-bit 3.7 GiB；P1/P4 標註改成「**本機可做**」 |
| `ch02.md` | 已改成 Gemma 4 版 | Ψ 從 11.91B 換成 **7.46B**，LoRA 可訓練參數 21.3M → **9.1M**，通訊量重算 |
| `ch08.md` | （新增） | 三步驟決策流程 → 你的變因清單 |

**保留 GPT-OSS 的數字作為對照是加分的**，不必全刪。
「同樣的公式，套到兩個不同的 MoE 上」本身就是公式通用性的證據。

### 11-2 產出 Week 2 執行總結

比照 `week1_執行總結.md` 的結構寫 `week2_執行總結.md`，至少包含：

- 換模型的理由與影響範圍（哪些 Week 1 結論還成立、哪些作廢）
- Gemma 4 記憶體預測表（Step 3）
- 資料管線的三個一致性數字 + 新的 token 長度分布 + `max_seq_len` 決策
- H1/E2 實測（Step 5）
- 微調前後 TMMLU+ 對照（Step 6 + 9）
- 四組消融的「預測 vs 實測」落差（Step 8）
- **dense vs MoE 對照表**（這是本週的新東西）
- E3/E4 路由分析（Step 9）
- 租卡補上的 CUDA-only 結果（Step 10）
- 誠實界定範圍：EP 的通訊與擴展性未實測

### 11-3 要跟主管對齊的問題清單

1. **E4B 主線 + 26B 對照**這個拆法認可嗎？（原本要用的 12B 因為 mlx-lm 不支援
   `gemma4_unified` 而換掉；E4B 的好處是每 token 計算量和 26B 的 active 對齊。）
2. `max_seq_len` 換 tokenizer 後變成 N，和 Week 1 定案的 2048 不同 —— 接受嗎？
3. bf16 vs 4-bit 的掉點幅度（現在本機就能做）如果超過 X%，要不要改走 bf16 路線？
4. E4（中英路由差異）若成立，Week 3 要不要擴大做成獨立的一節？
5. 租卡預算 US$2–3 是否核准？（清單已縮短，bf16 對照移回本機）
   要不要一次租 2 卡把 D1–D5 也做掉？

---

## 一頁時間表

| 日 | 上午 | 下午 |
|---|---|---|
| Day 1 | Step 0 清理 + Step 1 環境與下載（掛著跑） | Step 2 讀 ch08 |
| Day 2 | Step 2 讀 appb | Step 3 記憶體預測 + Step 4 資料管線 |
| Day 3 | Step 5 載入驗證 + roofline（兩個模型） | Step 6 baseline 評測 |
| Day 4 | Step 7 LoRA 微調（30 步試跑 → 跑滿） | Step 8 消融（checkpoint + seqlen） |
| Day 5 | Step 8 消融（batch + lora） | Step 9 微調後評測 + 路由分析 |
| 週末 | Step 10 租卡 2–4 小時 | Step 11 收尾、寫總結、與主管對齊 |

---

## 假設狀態總表（Week 1 → Week 2）

| 來源 | 編號 | 主題 | Week 1 狀態 | Week 2 怎麼處理 |
|---|---|---|---|---|
| ch01 | H1 | 權重記憶體 | ✅（GPT-OSS） | Step 5 + 7 在 Gemma 上重驗 |
| ch01 | H2 | LoRA 優化器可忽略 | 待驗 | Step 8 `--suite lora` |
| ch01 | H3 | 梯度檢查點取捨 | 待驗 | **Step 8 `--suite checkpoint`** |
| ch01 | H4 | logits 是隱藏大戶 | 待驗 | **Step 8 `--suite seqlen`**（Gemma 262K vocab 更明顯） |
| ch01 | H5 | 活化成長律 | 待驗 | Step 8 `--suite batch` + `seqlen` |
| ch01 | H6 | LoRA 掛 expert 的膨脹 | 待驗 | Step 3 算術（MoE 667M vs 11.5M＝**58×**；E4B dense 只有 3.8×）+ Step 8 |
| ch10 | P1 | bf16 vs 量化記憶體 | 待驗 | **Step 5 本機**（E4B bf16 只要 13.9 GiB） |
| ch10 | P2 | Flash Attention 效益 | 待驗 | Step 10 租卡 |
| ch10 | P3 | 平方項來源 | 待驗 | Step 8 `--suite seqlen` 間接驗 |
| ch10 | P4 | 量化掉點幅度 | 待驗 | **Step 7 + 9 本機**（4-bit 與 bf16 各微調一次再評測） |
| ch02 | D1–D5 | DP / ZeRO | 待租 2 卡 | Step 10 選配；租不到就標記「理論分析，未實測」 |
| ch06 | E1 | active 比例 | ✅（GPT-OSS） | Step 3 在 Gemma 上重算（3.82/25.23 = **15.1%**） |
| ch06 | E2 | 推論吞吐 roofline | ✅（GPT-OSS） | **Step 5 重驗，且這次有 dense 對照組** |
| ch06 | E3 | 負載均衡 | 待驗 | Step 9 `inspect_router_mlx.py` |
| ch06 | E4 | 中英路由差異 | 待驗 | **Step 9 `--compare-lang`（最有研究價值）** |
| ch06 | E5 | 微調後路由偏移 | 待驗 | Step 9（需 26B adapter，可能落到 Step 10） |
| ch06 | E6 | EP／MoE 划不划算 | 待驗 | **Step 3 + 5 + 8 的 dense vs MoE 對照表** |

---

## 常見坑（先知道就不會踩）

1. **Xet 下載失敗（`CAS Client Error: ... error decoding response body`）**：
   下載大模型時最常見的一個。加 `HF_HUB_DISABLE_XET=1` 退回一般 HTTPS 重跑即可，
   會續傳。Week 1 下載 GPT-OSS 時就踩過同一個坑。

2. **用了 Gemma 4 12B**：`ValueError: Model type gemma4_unified not supported.`
   mlx-lm 0.31.3 不支援 12B Unified（[issue #1481](https://github.com/ml-explore/mlx-lm/issues/1481)）。
   用 E4B / 26B-A4B / 31B，它們的 model_type 是 `gemma4`。
   另注意 mlx-community 的 E4B repo 名稱是**小寫** `gemma-4-e4b-it-4bit`。

3. **下載噴 `SSL: UNEXPECTED_EOF_WHILE_READING`**：這是 proxy 切斷 TLS，不是 Xet
   （看 traceback 有沒有 `http_proxy.py`）。降到 `--max-workers 1`、拉長
   `HF_HUB_DOWNLOAD_TIMEOUT`，並包 `until ... done` 迴圈續傳。詳見 Step 1 的說明。

4. **評測 config 的 `dataset_paths` 指到單一 `.parquet` 檔**：會噴
   `Dataset path is not a directory`。Twinkle Eval 只吃目錄 ——
   先跑 `python scripts/make_eval_subset.py` 做出科目子集資料夾。

5. **`transformers` 版本不夠新**：Gemma 4 的 chat template 需要 v5+。
   症狀是 `apply_chat_template` 不認得 `enable_thinking` 或 `reasoning` 欄位。
   `prepare_data_gemma.py` 的探針會直接中止並告訴你。

6. **`enable_thinking` 訓練與評測不一致**：這是 Week 2 版的頭號地雷。
   訓練資料開了、評測沒開（或反過來），微調前後的差值就不可信。
   兩份評測 config 已經寫死 `true`，不要改。

7. **`max_seq_length` 沿用 Week 1 的 2048**：換 tokenizer 後 token 長度分布會變，
   必須用 Step 4 印出來的涵蓋率重新決定。

8. **26B-A4B 沒調 wired limit 就跑**：症狀是載入到一半當掉或 Metal 報
   `Insufficient Memory`。先 `sudo sysctl iogpu.wired_limit_mb=20480`。

9. **LoRA 不小心掛到 MoE expert 上**：26B 的可訓練參數會從 11.5M 跳到 **667M**
   （優化器記憶體 0.17 → 9.94 GiB，58 倍），24GB 直接爆。
   `configs/lora_gemma4_26b_moe.yaml` 的 `keys` 只列 attention，不要加 `mlp.*`。

10. **兩份評測 config 不小心改到別的欄位**：`temperature` / `max_tokens` /
   `shuffle_options` / 科目清單都必須一致，只能差 `model.name` 那一行。

11. **`_to_delete/` 太早刪**：等 Step 4 跑完拿到新的 `data/train` 再刪。

12. **消融跑批會覆寫 adapter**：`run_ablation.py` 寫到 `out/_ablation_tmp`，
   不會動到 `out/lora-e4b`。但**不要在跑正式訓練的同時跑消融** —— 記憶體會打架。

13. **`mlx_lm.server` 的 model id**：`curl /v1/models` 看到什麼就在 config 的
   `model.name` 填什麼，不一致 Twinkle Eval 會 404。

---

## 附：這週新增／變動的檔案

```
ultrascale-lab/
├── scripts/
│   ├── cleanup_gptoss.sh           【新】Step 0 清理（HF cache / LM Studio）
│   ├── verify_env_week2.py         【新】Week 2 環境驗收
│   ├── predict_memory_gemma.py     【新】取代 predict_memory.py，雙模型
│   ├── prepare_data_gemma.py       【新】取代 prepare_data.py，Gemma template
│   ├── verify_load_mlx.py          【新】取代 verify_load.py，MLX/Metal
│   ├── inspect_router_mlx.py       【新】取代 inspect_router.py，MLX
│   ├── run_ablation.py             【新】消融跑批（H3–H6）
│   ├── make_eval_subset.py         【新】產生 Twinkle Eval 的科目子集目錄
│   └── verify_env.py               （沿用）
├── configs/
│   ├── lora_gemma4_e4b.yaml        【新】主線訓練設定（E4B）
│   ├── lora_gemma4_26b_moe.yaml    【新】MoE 訓練設定（stretch goal）
│   ├── eval_gemma4_e4b_base.yaml   【新】微調前評測
│   └── eval_gemma4_e4b_tuned.yaml  【新】微調後評測
├── reports/                        （Step 3–9 會逐步填滿）
├── datasets/
│   ├── ikala__tmmluplus/           66 科原始檔（Week 1 已下載）
│   └── subsets/                    【新】make_eval_subset.py 產生的科目子集
├── data/
│   ├── train/, val/                Step 4 重新產生
│   └── mlx/train.jsonl, valid.jsonl【新】MLX 訓練格式
├── out/                            【新】adapter 與融合後權重
├── _to_delete/gptoss_week1/        Step 0 已移入，確認後刪除
├── notes/                          Step 11 改寫 🧪 小節，另增 ch08.md
├── week1_執行總結.md                保留
└── week2_執行手冊.md                本檔
```

---

### Sources

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Gemma 4 發布公告](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) · [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it)（config.json 與 chat_template.jinja 已逐項核對）
- [Unsloth: Gemma 4 Fine-tuning Guide](https://unsloth.ai/docs/models/gemma-4/train)（VRAM 需求、thinking template 選擇）
- [mlx-community/gemma-4-26B-A4B-it 量化版](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit) · [MLX LoRA 微調文件](https://mlx-optiq.com/docs/finetune)
- [Twinkle Eval](https://github.com/ai-twinkle/Eval) · [ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus) · [twinkle-ai/tw-reasoning-instruct-50k](https://huggingface.co/datasets/twinkle-ai/tw-reasoning-instruct-50k)
