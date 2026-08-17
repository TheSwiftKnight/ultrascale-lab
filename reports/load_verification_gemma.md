# 本機載入驗證與 roofline 分析（Gemma 4 / MLX / Metal）

硬體假設：統一記憶體頻寬 273 GB/s（用 `--bandwidth` 改）

> ⚠️ **可比性**：這裡量到的吞吐與峰值記憶體是 Metal 上的數字，不能和租用 CUDA 卡的數字並列。準確率才是跨硬體可比的。


## H1 — 權重記憶體預測 vs 實測

| 模型 | 預測（4-bit） | 載入後 | 生成峰值 | 誤差 | 判定 |
|---|---:|---:|---:|---:|---|
| Gemma 4 E4B（dense, 4-bit） | 3.91 GiB | 3.91 GiB | 4.02 GiB | −0.0% | ✅ |
| Gemma 4 26B-A4B（MoE, 4-bit） | 13.22 GiB | 13.23 GiB | 13.37 GiB | −0.0% | ✅ |

## E2 — 記憶體頻寬 roofline

解碼是記憶體頻寬受限的：每產生一個 token，該次用到的權重就必須從統一記憶體讀一次。所以 `理論上限 = 頻寬 ÷ 每 token 讀取量`。MoE 只讀 active 那部分，dense 要讀全部 —— 這是「MoE 真的只算 active 參數」的物理證據，比「跑起來很快」有力得多。

| 模型 | 每 token 讀取 | 理論上限 | 實測 | 達成率 |
|---|---:|---:|---:|---:|
| Gemma 4 E4B（dense, 4-bit） | 4.20 GB | 65.0 tok/s | 69.1 tok/s | 106% |
| Gemma 4 26B-A4B（MoE, 4-bit）（MoE 實際） | 2.84 GB | 96.1 tok/s | 53.6 tok/s | 56% |
| Gemma 4 26B-A4B（MoE, 4-bit）（假設 dense） | 14.20 GB | 19.2 tok/s | — | 實測是它的 2.79× |

## dense vs MoE 實測對照

> E4B 的非嵌入參數（3.97B）≈ 26B-A4B 的 active（3.82B）—— 每 token 計算量相當，所以吞吐差異直接反映架構本身。

| 指標 | E4B dense | 26B-A4B MoE |
|---|---:|---:|
| 權重實際佔用 | 3.91 GiB | 13.22 GiB |
| 生成峰值 | 4.02 GiB | 13.37 GiB |
| 載入耗時 | 6.8s | 10.3s |
| 生成吞吐 | 69.1 tok/s | 53.6 tok/s |

**MoE 用 3.4 倍的記憶體，換到 0.78 倍的吞吐。**這個比值加上微調後的 TMMLU+ 準確率，就是「MoE 在單機情境下划不划算」的答案。


## 原始數據

```json
[
  {
    "model": "mlx-community/gemma-4-e4b-it-4bit",
    "load_s": 6.8,
    "weight_bytes": 4198750292,
    "weight_gib": 3.910390932112932,
    "param_elems": 1166568490,
    "after_load_gib": 3.9129392616450787,
    "peak_gib": 4.021695364266634,
    "tok_per_s": 69.1,
    "label": "Gemma 4 E4B（dense, 4-bit）",
    "is_moe": false,
    "expected_weight_gib": 3.91,
    "h1_error_pct": -0.0,
    "h1_pass": true,
    "roofline": {
      "read_per_token_gb": 4.198750292,
      "ceiling": 65.01934647558221,
      "measured": 69.1,
      "pct_of_ceiling": 106.27606050446887
    }
  },
  {
    "model": "mlx-community/gemma-4-26B-A4B-it-4bit",
    "load_s": 10.3,
    "weight_bytes": 14200055868,
    "weight_gib": 13.224832590669394,
    "param_elems": 3944621086,
    "after_load_gib": 13.227440992370248,
    "peak_gib": 13.37234104424715,
    "tok_per_s": 53.6,
    "label": "Gemma 4 26B-A4B（MoE, 4-bit）",
    "is_moe": true,
    "expected_weight_gib": 13.22,
    "h1_error_pct": -0.0,
    "h1_pass": true,
    "roofline": {
      "active_frac": 0.2,
      "read_per_token_moe_gb": 2.8400111736,
      "read_per_token_dense_gb": 14.200055868,
      "ceiling_moe": 96.12638236699082,
      "ceiling_dense": 19.225276473398168,
      "measured": 53.6,
      "pct_of_moe_ceiling": 55.759926338813194,
      "speedup_vs_dense_ceiling": 2.7879963169406596
    }
  }
]
```