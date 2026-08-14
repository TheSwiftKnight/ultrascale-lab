# 消融實驗：batch

**假設** — H5：活化與 logits 隨 batch size 線性成長；因為有 Flash 與滑動視窗，隨 seq 也接近線性而非 ch01 說的平方

設定：base config `configs/lora_gemma4_e4b.yaml`，每個變因跑 30 步（取後半平均，跳過暖機）
釘死的條件（變因以外都不變，且保證兩端都跑得起來）：`{'max_seq_length': 512, 'grad_checkpoint': True}`

| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |
|---|---:|---:|---:|---:|---|
| bs=1 | 5.52 GiB | 1.889 | 271 | 1.9247 | ✅ |
| bs=2 | 6.76 GiB | 3.759 | 263 | 1.7150 | ✅ |
| bs=4 | 9.01 GiB | 7.481 | 255 | 1.7390 | ✅ |
| bs=8 | 13.56 GiB | 15.075 | 257 | 1.7813 | ✅ |

**bs=1 → bs=8**：峰值記憶體 +145.7%，每步耗時 +698.0%。
