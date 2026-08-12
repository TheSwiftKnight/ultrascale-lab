# ultrascale-lab

繁體中文語言模型訓練工程實證 — 實驗程式碼庫
（對應 `ultrascale-proposal_1.pptx`、`week1_執行總結.md`、`week2_執行手冊.md`）

> **Week 2 起改用 Gemma 4，訓練改在本機 M4 Pro（MLX）進行。**
> GPT-OSS 相關的程式碼與產物已移到 `_to_delete/gptoss_week1/`；
> Week 1 的**結論與方法論**保留在 `week1_執行總結.md`，仍是最終報告的依據。

---

## 兩條實驗線

機器是 M4 Pro / 24GB 統一記憶體，這個數字決定了模型配置：

| | Gemma 4 12B Unified | Gemma 4 26B-A4B |
|---|---|---|
| 架構 | dense，48 層 | MoE，30 層，128 專家取 8 |
| 總參數 / active | 11.91B / 11.91B | 25.23B / **3.82B** |
| 4-bit 權重 | 5.9 GiB | 12.5 GiB |
| 微調預算（seq2048, bs1, ckpt） | **11.2 GiB ✅** | 17.2 GiB ⚠️ |
| 這個專案的角色 | **微調主線**（訓練 + 消融 + 評測） | **MoE 對照組**（推論 + roofline + 路由分析） |

26B-A4B 是 GPT-OSS 20B（20.9B/3.6B）的規格替身，所以 Week 1 的 ch06 分析、
roofline 論證與 E1–E6 假設可以整套沿用；12B 則負責保證這週交付得出微調結果。
兩者並置多出一組原計畫沒有的 **dense vs MoE 對照**。

---

## 目錄結構

```
ultrascale-lab/
├── scripts/
│   ├── cleanup_gptoss.sh          Step 0：清 HF cache / LM Studio 的 gpt-oss
│   ├── verify_env_week2.py        環境驗收，缺什麼直接告訴你怎麼補
│   ├── predict_memory_gemma.py    ch01/ch10 公式 → 兩個 Gemma 4 的記憶體預測
│   ├── prepare_data_gemma.py      資料管線（Gemma 4 template，輸出 HF + MLX 兩種格式）
│   ├── verify_load_mlx.py         載入驗證 + 記憶體頻寬 roofline（H1 / E2）
│   ├── inspect_router_mlx.py      MoE 路由分布（E3 / E4 / E5）
│   └── run_ablation.py            消融跑批（H3 / H4 / H5 / H6）
├── configs/
│   ├── lora_gemma4_12b.yaml       主線 LoRA 訓練設定
│   ├── lora_gemma4_26b_moe.yaml   MoE LoRA（stretch goal，24GB 貼邊）
│   ├── eval_gemma4_12b_base.yaml  Twinkle Eval 微調前
│   └── eval_gemma4_12b_tuned.yaml Twinkle Eval 微調後（只差 model.name 一行）
├── reports/                       所有可放進簡報的產物
├── data/
│   ├── train/, val/               HF dataset
│   └── mlx/train.jsonl, valid.jsonl
├── out/                           adapter 與融合後的權重
├── notes/                         Playbook 章節筆記
├── datasets/ikala__tmmluplus/     TMMLU+ 66 科
└── Eval/                          Twinkle Eval 原始碼
```

---

## 快速開始

```bash
source .venv/bin/activate

# 0. 清掉 GPT-OSS 的舊資源（先 dry-run 看要刪什麼）
bash scripts/cleanup_gptoss.sh
bash scripts/cleanup_gptoss.sh --yes

# 1. 套件與環境驗收
uv pip install -U mlx mlx-lm "transformers>=5.0" "datasets>=2.19" \
                  datasketch matplotlib pyyaml twinkle-eval
python scripts/verify_env_week2.py --check-models

# 2. 下載模型（12B 約 7GB、26B 約 16GB）
hf download mlx-community/gemma-4-12B-it-4bit
hf download mlx-community/gemma-4-26B-A4B-it-4bit

# 3. 記憶體預測（不需 GPU，30 秒；--verify-config 會上網核對 config.json）
python scripts/predict_memory_gemma.py --verify-config

# 4. 資料管線（約 3–5 分鐘）
#    ⚠️ 跑完把印出來的建議 max_seq_len 填進 configs/lora_gemma4_12b.yaml
python scripts/prepare_data_gemma.py

# 5. 載入驗證 + roofline（兩個模型）
sudo sysctl iogpu.wired_limit_mb=20480     # 26B 需要
python scripts/verify_load_mlx.py --both

# 6. Baseline 評測（另開一個終端機起 server）
mlx_lm.server --model mlx-community/gemma-4-12B-it-4bit --host 127.0.0.1 --port 1234
twinkle-eval --config configs/eval_gemma4_12b_base.yaml

# 7. LoRA 微調（先跑 30 步確認記憶體與速度）
mlx_lm.lora --config configs/lora_gemma4_12b.yaml --iters 30 --steps-per-report 5
mlx_lm.lora --config configs/lora_gemma4_12b.yaml
mlx_lm.fuse --model mlx-community/gemma-4-12B-it-4bit \
            --adapter-path out/lora-12b --save-path out/gemma4-12b-tw

# 8. 消融
python scripts/run_ablation.py --suite all --iters 40

# 9. 微調後評測 + 路由分析
mlx_lm.server --model out/gemma4-12b-tw --host 127.0.0.1 --port 1234
twinkle-eval --config configs/eval_gemma4_12b_tuned.yaml
python scripts/inspect_router_mlx.py --dump-modules
python scripts/inspect_router_mlx.py --compare-lang --save reports/router_before.json
```

完整的逐步說明、讀書計畫、驗收條件與常見坑，見 **`week2_執行手冊.md`**。

---

## 三個換模型後最重要的數字

1. **26B-A4B 的 4-bit 權重（12.5 GiB）比 GPT-OSS 的 MXFP4（12.8 GiB）還小**，
   雖然總參數多 20%。因為 GPT-OSS 只量化 expert，Gemma 走 MLX 通用量化全部一起壓。
   → 「量化省多少取決於涵蓋範圍，不只取決於位元數」。

2. **logits 變成第二大戶**。Gemma 4 vocab = 262,144，比 GPT-OSS 大 30%；
   seq=2048 時光 logits 就要 3.00 GiB。這是 24GB 機器上最容易 OOM 的地方。

3. **dense 的活化比 MoE 大 2.5 倍**（15.05 vs 6.11 GiB @ seq2048）。
   參數多的那個活化反而小 —— 因為 MoE 每 token 只過 8/128 個專家。

---

## 一致性規則（違反就整週白做）

| 規則 | 為什麼 |
|---|---|
| `enable_thinking` 訓練與評測必須同開或同關 | Gemma 4 版的「reasoning_effort 要同檔位」；不一致則微調前後對比失真 |
| 兩份評測 config 只能差 `model.name` 一行 | 否則準確率差值不是微調造成的 |
| `max_seq_len` 用 Step 4 印出的涵蓋率重新決定 | 換 tokenizer 後分布會變，不能沿用 Week 1 的 2048 |
| `seed=42` 貫穿 shuffle / split / sampling | 可複現是提案的交付承諾 |
| 準確率跨硬體可比；吞吐與峰值記憶體不可比 | Metal 與 CUDA 的數字要分欄放 |
