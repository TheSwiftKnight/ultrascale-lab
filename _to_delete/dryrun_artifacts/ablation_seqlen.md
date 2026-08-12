# 消融實驗：seqlen

**假設** — H4：logits（seq × 262,144 × 6 bytes）隨 seq 線性成長，是峰值記憶體裡被低估的大戶（512→4096：0.75 → 6.00 GiB）

設定：base config `configs/lora_gemma4_e4b.yaml`，每個變因跑 30 步（取後半平均，跳過暖機）
釘死的條件（變因以外都不變，且保證兩端都跑得起來）：`{'batch_size': 1, 'grad_checkpoint': True}`

| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |
|---|---:|---:|---:|---:|---|
| seq=512 | — | — | — | — | — dry-run |
| seq=1024 | — | — | — | — | — dry-run |
| seq=2048 | — | — | — | — | — dry-run |
| seq=4096 | — | — | — | — | — dry-run |