# Gemma 4 記憶體預測（Playbook ch01 + ch10 公式推算）

> 取代 Week 1 的 `memory_prediction.md`（GPT-OSS 20B 版）。
> 本檔是**事前預測**；本機實測由 `verify_load_mlx.py` 回填，
> CUDA 實測由租卡時的 torch profiler 回填。


---

# Gemma 4 12B Unified（dense）

設定：seq=1024, batch=1, LoRA rank=16, target=attn


## 一、參數量驗算

| 項目 | 參數量 | 備註 |
|---|---:|---|
| sliding 層 attention × 40 | 47,185,920 | head_dim=256，KV 8 頭，K/V **分開兩組** |
| full 層 attention × 8 | 64,880,640 | head_dim=512，KV 1 頭，K/V **共用一組** |
| 每層 FFN（dense, inter=15,360） | 176,947,200 | 佔單層 78.9% |
| embedding（tied，不另計 lm_head） | 1,006,632,960 | v=262,144 × h=3,840，佔全模型 8.5% |
| **總參數** | **11.91B** | 官方標示 11.95B，誤差 0.4% |
| **active 參數** | **11.91B** | 官方標示 11.95B，誤差 0.4% |

> 殘差來自視覺／音訊塔（26B 的視覺編碼器約 550M）與 per-layer embedding，本表只算語言主幹。誤差 <5% 即視為公式正確。


## 二、Q1：載入權重要多少？

| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |
|---|---:|---:|---|
| bf16 全精度 | 2.000 | **22.2 GiB** | ❌ 放不下 |
| 8-bit（MLX q8） | 1.062 | **11.8 GiB** | ⚠️ 貼邊 |
| 4-bit body + 8-bit embed | 0.576 | **6.4 GiB** | ✅ 有餘裕 |
| 4-bit 全量（MLX 預設） | 0.531 | **5.9 GiB** | ✅ 有餘裕 |


## 三、Q2：LoRA 的優化器狀態

| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |
|---|---:|---:|---:|
| 全參數微調 | 11.91B | 100% | **177 GiB** |
| LoRA r=16（attention only） | 21.3M | 0.1792% | **0.318 GiB** |
| LoRA r=16（含 FFN/expert） | 65.6M | 0.5507% | **0.977 GiB** |

差距 3 倍。
dense 模型的 FFN 只有 L 組矩陣，膨脹幅度遠小於 MoE —— 這正是 dense vs MoE 對照要呈現的差別之一。


## 四、Q3：活化記憶體

| 情境 | 活化記憶體 | 說明 |
|---|---:|---|
| ch01 原始公式直接代入 | 9.7 GiB | 高估，架構假設不符 |
| 修正後，**無** Flash Attention | 10.52 GiB | S/P 矩陣要落地 |
| 修正後，**有** Flash Attention | 7.52 GiB | ch10：不具現化 S/P |
| 修正後，Flash + full checkpointing | 0.52 GiB | ch01：以算換記憶體 |

修正的三處：K/V 共用投影且只有 8 頭（GQA）；40/48 層的注意力視窗只有 1024 個 token；
dense FFN 的 intermediate=15,360（= 4h）。


## 五、24GB 統一記憶體的預算表（seq=1024, bs=1）

> ⚠️ **logits 是本週最大的陷阱**：Gemma 4 的 vocab=262,144，比 GPT-OSS 的 201,088 大 30%。seq=1024 時光 logits 就要 1.50 GiB，且隨 seq 線性成長。這是 H4 的量測點。

| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-bit + Flash，不開 checkpointing | 5.9 | 0.32 | 7.52 | 1.50 | 1.0 | **16.2 GiB** | ⚠️ 需調高 wired limit |
| 4-bit + Flash + full checkpointing | 5.9 | 0.32 | 0.52 | 1.50 | 1.0 | **9.2 GiB** | ✅ 安全 |
| bf16 + Flash + full checkpointing | 22.2 | 0.32 | 0.52 | 1.50 | 1.0 | **25.5 GiB** | ❌ OOM |

---

# Gemma 4 26B-A4B（MoE）

設定：seq=1024, batch=1, LoRA rank=16, target=attn


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
| LoRA r=16（attention only） | 11.5M | 0.0455% | **0.171 GiB** |
| LoRA r=16（含 FFN/expert） | 667.4M | 2.6449% | **9.945 GiB** |

差距 58 倍。
對 MoE 而言掛到 expert 上等於要處理 30×128 = 3,840 組矩陣，這是 **H6 / E6 的量測點**。


## 四、Q3：活化記憶體

| 情境 | 活化記憶體 | 說明 |
|---|---:|---|
| ch01 原始公式直接代入 | 5.1 GiB | 高估，架構假設不符 |
| 修正後，**無** Flash Attention | 4.93 GiB | S/P 矩陣要落地 |
| 修正後，**有** Flash Attention | 3.06 GiB | ch10：不具現化 S/P |
| 修正後，Flash + full checkpointing | 0.27 GiB | ch01：以算換記憶體 |

修正的三處：K/V 共用投影且只有 8 頭（GQA）；25/30 層的注意力視窗只有 1024 個 token；
MoE 每 token 只過 8/128 個專家，**活化跟著 active 3.8B 走，不跟著 25.2B 走**。


## 五、24GB 統一記憶體的預算表（seq=1024, bs=1）

> ⚠️ **logits 是本週最大的陷阱**：Gemma 4 的 vocab=262,144，比 GPT-OSS 的 201,088 大 30%。seq=1024 時光 logits 就要 1.50 GiB，且隨 seq 線性成長。這是 H4 的量測點。

| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-bit + Flash，不開 checkpointing | 12.5 | 0.17 | 3.06 | 1.50 | 1.0 | **18.2 GiB** | ⚠️ 需調高 wired limit |
| 4-bit + Flash + full checkpointing | 12.5 | 0.17 | 0.27 | 1.50 | 1.0 | **15.4 GiB** | ⚠️ 需調高 wired limit |
| bf16 + Flash + full checkpointing | 47.0 | 0.17 | 0.27 | 1.50 | 1.0 | **49.9 GiB** | ❌ OOM |

---

# 六、dense vs MoE 對照（Week 2 的核心論點）

| 指標 | 12B dense | 26B-A4B MoE | 誰贏 |
|---|---:|---:|---|
| 總參數 | 11.91B | 25.23B | MoE 大 2.1× |
| active 參數 | 11.91B | 3.82B | MoE 只有 dense 的 32% |
| 4-bit 權重 | 5.9 GiB | 12.5 GiB | dense 小 2.1× |
| 活化（Flash, 無 ckpt） | 7.52 GiB | 3.06 GiB | MoE 小 |
| LoRA(attn) 可訓練參數 | 21.3M | 11.5M | — |
| LoRA(全掛) 可訓練參數 | 65.6M | 667.4M | MoE 膨脹 18.9× 更兇 |

**這張表就是 Week 2 要交付的核心對照**：MoE 用 2.2 倍的總參數與 1.4 倍的權重記憶體，換到只有 dense 三分之一的 active 參數（＝三分之一的每 token 計算量與記憶體頻寬）。值不值得，用微調後的 TMMLU+ 準確率與實測吞吐來回答。
