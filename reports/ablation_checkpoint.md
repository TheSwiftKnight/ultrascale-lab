# 消融實驗：checkpoint

**假設** — H3：開 full checkpointing 後活化記憶體應下降 >85%（seq512 預測 3.76 → 0.26 GiB），step time 增加約 30–40%

設定：base config `configs/lora_gemma4_12b.yaml`，每個變因跑 30 步（取後半平均，跳過暖機）
釘死的條件（變因以外都不變，且保證兩端都跑得起來）：`{'max_seq_length': 512, 'batch_size': 1}`

| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |
|---|---:|---:|---:|---:|---|
| 不開梯度檢查點 | — | — | — | — | — dry-run |
| 開梯度檢查點 | — | — | — | — | — dry-run |