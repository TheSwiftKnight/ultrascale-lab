# Gemma 4 記憶體預測（Playbook ch01 + ch10 公式推算）

> 取代 Week 1 的 `memory_prediction.md`（GPT-OSS 20B 版）。
> 本檔是**事前預測**；本機實測由 `verify_load_mlx.py` 回填，
> CUDA 實測由租卡時的 torch profiler 回填。


---

# Gemma 4 26B-A4B（MoE）

設定：seq=2048, batch=1, LoRA rank=16, target=all


## 一、參數量驗算

| 項目 | 參數量 | 備註 |
|---|---:|---|
| sliding 層 attention × 25 | 34,603,008 | head_dim=256，KV 8 頭，K/V **分開兩組** |
| full 層 attention × 5 | 49,020,928 | head_dim=512，KV 2 頭，K/V **共用一組** |
| 每層 MoE（128 路由專家 + 1 共享） | 779,468,800 | 佔單層 95.7% |
| 每層 MoE（只算 active，top-8） | 65,781,760 | = 全部的 8.4% |
| embedding（tied，不另計 lm_head） | 738,197,504 | v=262,144 × h=2,816，佔全模型 2.9% |
| **總參數** | **25.23B** | 官方標示 25.20B，誤差 0.1% |
| **active 參數** | **3.82B** | 官方標示 3.80B，誤差 0.6% |

> 殘差來自視覺／音訊塔（26B 的視覺編碼器約 550M）與 per-layer embedding，本表只算語言主幹。誤差 <5% 即視為公式正確。


每 token 只走 **8/128 = 6.2%** 的路由專家。全部 expert 權重共 22.84B，佔全模型 **90.5%**。


## 二、Q1：載入權重要多少？

| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |
|---|---:|---:|---|
| bf16 全精度 | 2.000 | **47.0 GiB** | ❌ 放不下 |
| 8-bit（MLX q8） | 1.062 | **25.0 GiB** | ❌ 放不下 |
| 4-bit body + 8-bit embed | 0.547 | **12.8 GiB** | ⚠️ 貼邊 |
| 4-bit 全量（MLX 預設） | 0.531 | **12.5 GiB** | ⚠️ 貼邊 |

> **和 GPT-OSS 的關鍵差異**：GPT-OSS 原生 MXFP4 只量化 expert，attention／router／embed／lm_head 都留 bf16，所以 20.9B 壓到 12.8 GiB 就到底了。Gemma 4 沒有官方量化權重，走 MLX 的通用量化——**所有線性層一起壓**，加上 91% 的參數本來就在 expert 裡，結果 25.2B 反而只要 12.5 GiB，比 GPT-OSS 還小。


## 三、Q2：LoRA 的優化器狀態

| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |
|---|---:|---:|---:|
| 全參數微調 | 25.23B | 100% | **376 GiB** |
| LoRA r=16（attention only） | 667.4M | 2.6449% | **9.945 GiB** |
| LoRA r=16（含 FFN/expert） | 667.4M | 2.6449% | **9.945 GiB** |

差距 1 倍。
對 MoE 而言掛到 expert 上等於要處理 30×128 = 3,840 組矩陣，這是 **H6 / E6 的量測點**。


## 四、Q3：活化記憶體

| 情境 | 活化記憶體 | 說明 |
|---|---:|---|
| ch01 原始公式直接代入 | 14.9 GiB | 高估，架構假設不符 |
| 修正後，**無** Flash Attention | 10.49 GiB | S/P 矩陣要落地 |
| 修正後，**有** Flash Attention | 6.11 GiB | ch10：不具現化 S/P |
| 修正後，Flash + full checkpointing | 0.54 GiB | ch01：以算換記憶體 |

修正的三處：K/V 共用投影且只有 8 頭（GQA）；25/30 層的注意力視窗只有 1024 個 token；
MoE 每 token 只過 8/128 個專家，**活化跟著 active 3.8B 走，不跟著 25.2B 走**。


## 五、24GB 統一記憶體的預算表（seq=2048, bs=1）

> ⚠️ **logits 是本週最大的陷阱**：Gemma 4 的 vocab=262,144，比 GPT-OSS 的 201,088 大 30%。seq=2048 時光 logits 就要 3.00 GiB，且隨 seq 線性成長。這是 H4 的量測點。

| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-bit + Flash，不開 checkpointing | 12.5 | 9.94 | 6.11 | 3.00 | 1.0 | **32.5 GiB** | ❌ OOM |
| 4-bit + Flash + full checkpointing | 12.5 | 9.94 | 0.54 | 3.00 | 1.0 | **27.0 GiB** | ❌ OOM |
| bf16 + Flash + full checkpointing | 47.0 | 9.94 | 0.54 | 3.00 | 1.0 | **61.5 GiB** | ❌ OOM |