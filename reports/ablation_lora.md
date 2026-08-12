# 消融實驗：lora

**假設** — H6：可訓練參數隨掛載層數線性成長（4/16/48 層 ≈ 1.8M/7.1M/21.3M），但優化器記憶體相對 5.9 GiB 的權重仍是雜訊

設定：base config `configs/lora_gemma4_12b.yaml`，每個變因跑 30 步（取後半平均，跳過暖機）
釘死的條件（變因以外都不變，且保證兩端都跑得起來）：`{'max_seq_length': 1024, 'batch_size': 1, 'grad_checkpoint': True}`

| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |
|---|---:|---:|---:|---:|---|
| LoRA 4 層 | — | — | — | — | — dry-run |
| LoRA 16 層 | — | — | — | — | — dry-run |
| LoRA 全部層 | — | — | — | — | — dry-run |