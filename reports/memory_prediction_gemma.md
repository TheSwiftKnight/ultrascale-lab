# Gemma 4 記憶體預測（Playbook ch01 + ch10 公式推算）

> 本檔是**事前預測**；本機實測由 `verify_load_mlx.py` 回填，
> CUDA 實測由租卡時的 torch profiler 回填。

> ⚠️ 主線模型是 **E4B** 不是 12B —— Gemma 4 12B Unified 的 `model_type` 是
> `gemma4_unified`，mlx-lm 0.31.3 不支援（issue #1481）。
> E4B / 26B-A4B / 31B 用的是 `gemma4`，都能跑。


---

# Gemma 4 E4B（dense）

MLX 權重：`mlx-community/gemma-4-e4b-it-4bit`　bf16：`mlx-community/gemma-4-e4b-it-bf16`

設定：seq=2048, batch=1, LoRA rank=16, target=attn


## 一、參數量驗算

| 項目 | 參數量 | 備註 |
|---|---:|---|
| 所有層的 attention | 587,202,560 | 35 sliding(hd=256) + 7 full(hd=512)；24/42 層有自己的 K/V |
| 每層 FFN（dense, inter=10,240 = 4h） | 78,643,200 | 全部層合計 3,303,014,400 |
| embedding（tied） | 671,088,640 | v=262,144 × h=2,560 |
| **Per-Layer Embeddings** | 2,818,572,288 | 262,144 × (42 層 × 256)，佔全模型 **38%** |
| PLE 的投影層 | 82,575,360 | 每層一組 gate + projection |
| **總參數** | **7.46B** | 官方標示 4.5B effective / 8B with embeddings；誤差 6.7% |
| **active / 非嵌入參數** | **3.97B** | 官方 4.5B，誤差 11.7% |

> 殘差來自視覺／音訊塔（E4B 原生支援影像與音訊）與少數 norm 項 —— 本表只算**語言主幹**。
> 官方標的是含多模態的完整模型，所以本表會系統性偏低約 5–12%，方向一致即視為公式正確。


## 二、Q1：載入權重要多少？

| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |
|---|---:|---:|---|
| bf16 全精度 | 2.000 | **13.9 GiB** | ⚠️ 貼邊 |
| 8-bit（MLX q8） | 1.062 | **7.4 GiB** | ✅ 有餘裕 |
| 4-bit body + 8-bit embed | 0.780 | **5.4 GiB** | ✅ 有餘裕 |
| 4-bit 全量（MLX 預設，group_size=64） | 0.531 | **3.7 GiB** | ✅ 有餘裕 |

> 交叉驗證：HF repo 上 4-bit 版實際 5.2 GB （= 4.8 GiB）、bf16 版 15.9 GB。比預測略高，差額是視覺／音訊塔（本表未計）。


## 三、Q2：LoRA 的優化器狀態

| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |
|---|---:|---:|---:|
| 全參數微調 | 7.46B | 100% | **111 GiB** |
| LoRA r=16（attention only） | 9.1M | 0.1216% | **0.135 GiB** |
| LoRA r=16（含 FFN/expert） | 34.9M | 0.4674% | **0.520 GiB** |

掛到 FFN 上會膨脹 **3.8 倍**。
dense 只有 42 組 FFN，膨脹幅度遠小於 MoE —— 這正是 dense vs MoE 對照要呈現的差別之一。


## 四、Q3：活化記憶體

| 情境 | 活化記憶體 | 說明 |
|---|---:|---|
| ch01 原始公式直接代入 | 13.5 GiB | 高估，架構假設不符 |
| 修正後，**無** Flash Attention | 10.31 GiB | S/P 矩陣要落地 |
| 修正後，**有** Flash Attention | 8.34 GiB | ch10：不具現化 S/P |
| 修正後，Flash + full checkpointing | 0.63 GiB | ch01：以算換記憶體 |

修正的地方：35/42 層的注意力視窗只有 512 個 token；KV 只有 2 頭（GQA）
；且後 18 層共用前面的 K/V，連投影都沒有
.


## 五、24GB 統一記憶體的預算表（seq=2048, bs=1）

> ⚠️ **logits 是最容易被忽略的一項**：vocab=262,144，seq=2048 時光 logits 就要 3.00 GiB，隨 seq 與 bs 線性成長。這是 H4 的量測點。

| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-bit + Flash，不開 checkpointing | 3.7 | 0.14 | 8.34 | 3.00 | 1.0 | **16.2 GiB** | ⚠️ 需調高 wired limit |
| 4-bit + Flash + full checkpointing | 3.7 | 0.14 | 0.63 | 3.00 | 1.0 | **8.5 GiB** | ✅ 安全 |
| bf16 + Flash + full checkpointing | 13.9 | 0.14 | 0.63 | 3.00 | 1.0 | **18.7 GiB** | ⚠️ 需調高 wired limit |

---

# Gemma 4 26B-A4B（MoE）

MLX 權重：`mlx-community/gemma-4-26B-A4B-it-4bit`

設定：seq=2048, batch=1, LoRA rank=16, target=attn


## 一、參數量驗算

| 項目 | 參數量 | 備註 |
|---|---:|---|
| 所有層的 attention | 1,110,179,840 | 25 sliding(hd=256) + 5 full(hd=512)；30/30 層有自己的 K/V |
| 每層 MoE（128 路由專家 + 1 共享 + router） | 779,468,800 | active 只有 65,781,760（8.4%） |
| embedding（tied） | 738,197,504 | v=262,144 × h=2,816 |
| **總參數** | **25.23B** | 官方標示 25.2B total / 3.8B active；誤差 0.1% |
| **active / 非嵌入參數** | **3.82B** | 官方 3.8B，誤差 0.6% |

> 殘差來自視覺／音訊塔（E4B 原生支援影像與音訊）與少數 norm 項 —— 本表只算**語言主幹**。
> 官方標的是含多模態的完整模型，所以本表會系統性偏低約 5–12%，方向一致即視為公式正確。

每 token 只走 **8/128 = 6.2%** 的路由專家。全部 expert 權重共 22.84B，佔全模型 **90.5%**。


## 二、Q1：載入權重要多少？

| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |
|---|---:|---:|---|
| bf16 全精度 | 2.000 | **47.0 GiB** | ❌ 放不下 |
| 8-bit（MLX q8） | 1.062 | **25.0 GiB** | ❌ 放不下 |
| 4-bit body + 8-bit embed | 0.547 | **12.8 GiB** | ⚠️ 貼邊 |
| 4-bit 全量（MLX 預設，group_size=64） | 0.531 | **12.5 GiB** | ⚠️ 貼邊 |

## 三、Q2：LoRA 的優化器狀態

| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |
|---|---:|---:|---:|
| 全參數微調 | 25.23B | 100% | **376 GiB** |
| LoRA r=16（attention only） | 11.5M | 0.0455% | **0.171 GiB** |
| LoRA r=16（含 FFN/expert） | 667.4M | 2.6449% | **9.945 GiB** |

掛到 FFN 上會膨脹 **58.1 倍**。
對 MoE 而言等於要處理 30×128 = 3,840 組矩陣，這是 **H6 / E6 的量測點**。


## 四、Q3：活化記憶體

| 情境 | 活化記憶體 | 說明 |
|---|---:|---|
| ch01 原始公式直接代入 | 14.9 GiB | 高估，架構假設不符 |
| 修正後，**無** Flash Attention | 10.49 GiB | S/P 矩陣要落地 |
| 修正後，**有** Flash Attention | 6.11 GiB | ch10：不具現化 S/P |
| 修正後，Flash + full checkpointing | 0.54 GiB | ch01：以算換記憶體 |

修正的地方：25/30 層的注意力視窗只有 1024 個 token；KV 只有 8 頭（GQA）
；MoE 每 token 只過 8/128 個專家，**活化跟著 active 3.8B 走**
.


## 五、24GB 統一記憶體的預算表（seq=2048, bs=1）

> ⚠️ **logits 是最容易被忽略的一項**：vocab=262,144，seq=2048 時光 logits 就要 3.00 GiB，隨 seq 與 bs 線性成長。這是 H4 的量測點。

| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| 4-bit + Flash，不開 checkpointing | 12.5 | 0.17 | 6.11 | 3.00 | 1.0 | **22.8 GiB** | ❌ OOM |
| 4-bit + Flash + full checkpointing | 12.5 | 0.17 | 0.54 | 3.00 | 1.0 | **17.2 GiB** | ⚠️ 需調高 wired limit |
| bf16 + Flash + full checkpointing | 47.0 | 0.17 | 0.54 | 3.00 | 1.0 | **51.7 GiB** | ❌ OOM |

---

# 六、dense vs MoE 對照（Week 2 的核心論點）

**這組配對的關鍵**：E4B 的非嵌入參數與 26B-A4B 的 active 參數幾乎一樣，也就是**每 token 的計算量相當**。差別只在總參數與記憶體佔用。所以「MoE 到底買到了什麼」可以被乾淨地量出來。

| 指標 | E4B dense | 26B-A4B MoE | 差異 |
|---|---:|---:|---|
| 總參數 | 7.46B | 25.23B | MoE 大 3.4× |
| 每 token 實際用到 | 3.97B（非嵌入） | 3.82B（active） | **幾乎相同** |
| 4-bit 權重 | 3.7 GiB | 12.5 GiB | dense 小 3.4× |
| bf16 權重 | 13.9 GiB | 47.0 GiB | E4B **本機塞得下**，26B 不行 |
| 活化（Flash, 無 ckpt） | 8.34 GiB | 6.11 GiB | MoE 小 |
| LoRA(attn) 可訓練參數 | 9.1M | 11.5M | — |
| LoRA 掛到 FFN 的膨脹 | 3.8× | 58.1× | MoE 兇 15 倍 |

**要回答的問題**：在每 token 計算量相同的前提下，MoE 多花 8.8 GiB 的記憶體養 25B 參數，換到多少準確率？用微調前後的 TMMLU+ 與實測吞吐回答。
