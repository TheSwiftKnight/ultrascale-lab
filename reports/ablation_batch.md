# 消融實驗：batch

**假設** — H5：活化與 logits 隨 batch size 線性成長；因為有 Flash 與滑動視窗，隨 seq 也接近線性而非 ch01 說的平方

設定：base config `configs/lora_gemma4_12b.yaml`，每個變因跑 30 步（取後半平均，跳過暖機）
釘死的條件（變因以外都不變，且保證兩端都跑得起來）：`{'max_seq_length': 512, 'grad_checkpoint': True}`

| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |
|---|---:|---:|---:|---:|---|
| bs=1 | — | — | — | — | — dry-run |
| bs=2 | — | — | — | — | — dry-run |
| bs=4 | — | — | — | — | — dry-run |
| bs=8 | — | — | — | — | — dry-run |