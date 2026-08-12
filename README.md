# ultrascale-lab

繁體中文語言模型訓練工程實證 — 實驗程式碼庫
（對應 `ultrascale-proposal_1.pptx`、`week1_執行總結.md`、`week2_執行手冊.md`）

> **Week 2 起改用 Gemma 4，訓練改在本機 M4 Pro（MLX）進行。**
> GPT-OSS 相關的程式碼與產物已移到 `_to_delete/gptoss_week1/`；
> Week 1 的**結論與方法論**保留在 `week1_執行總結.md`，仍是最終報告的依據。

---

## 兩條實驗線

機器是 M4 Pro / 24GB 統一記憶體，這個數字決定了模型配置：

| | Gemma 4 E4B | Gemma 4 26B-A4B |
|---|---|---|
| 架構 | dense，42 層 | MoE，30 層，128 專家取 8 |
| 總參數 | 7.46B | 25.23B |
| **每 token 實際用到** | **3.97B**（非嵌入） | **3.82B**（active） |
| 4-bit / bf16 權重 | 3.7 / **13.9** GiB | 12.5 / 47.0 GiB |
| 微調預算（seq2048, bs1, ckpt） | **8.5 GiB ✅** | 17.2 GiB ⚠️ |
| 這個專案的角色 | **微調主線**（訓練 + 消融 + 評測 + bf16 對照） | **MoE 對照組**（推論 + roofline + 路由分析） |

⚠️ **不要用 Gemma 4 12B** —— 它的 `model_type` 是 `gemma4_unified`，
mlx-lm 0.31.3 不支援（[issue #1481](https://github.com/ml-explore/mlx-lm/issues/1481)）。
E4B / 26B-A4B / 31B 用的是 `gemma4`，都能跑。

26B-A4B 是 GPT-OSS 20B（20.9B/3.6B）的規格替身，Week 1 的 ch06 分析、
roofline 論證與 E1–E6 假設可以整套沿用。
**E4B 的非嵌入參數（3.97B）幾乎等於 26B-A4B 的 active（3.82B）** ——
每 token 計算量對齊、總參數差 3.4 倍，這是最乾淨的 dense vs MoE 對照。
E4B 的 bf16 只要 13.9 GiB，所以 P1/P4（bf16 vs 4-bit 掉點）本機就能做完。

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
│   ├── lora_gemma4_e4b.yaml       主線 LoRA 訓練設定
│   ├── lora_gemma4_26b_moe.yaml   MoE LoRA（stretch goal，24GB 貼邊）
│   ├── eval_gemma4_e4b_base.yaml  Twinkle Eval 微調前
│   └── eval_gemma4_e4b_tuned.yaml Twinkle Eval 微調後（只差 model.name 一行）
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

# 2. 下載模型（注意 e4b 小寫；HF_HUB_DISABLE_XET=1 避免 Xet 重組失敗）
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-e4b-it-4bit --max-workers 4       # 5.2 GB
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-26B-A4B-it-4bit --max-workers 4   # 15.4 GB
HF_HUB_DISABLE_XET=1 hf download mlx-community/gemma-4-e4b-it-bf16 --max-workers 4       # 15.9 GB

# 3. 記憶體預測（不需 GPU，30 秒；--verify-config 會上網核對 config.json）
python scripts/predict_memory_gemma.py --verify-config

# 4. 資料管線（約 3–5 分鐘）
#    ⚠️ 跑完把印出來的建議 max_seq_len 填進 configs/lora_gemma4_e4b.yaml
python scripts/prepare_data_gemma.py

# 5. 載入驗證 + roofline（兩個模型）
sudo sysctl iogpu.wired_limit_mb=20480     # 26B 需要
python scripts/verify_load_mlx.py --both

# 6. Baseline 評測（另開一個終端機起 server）
mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --host 127.0.0.1 --port 1234
twinkle-eval --config configs/eval_gemma4_e4b_base.yaml

# 7. LoRA 微調（先跑 30 步確認記憶體與速度）
mlx_lm.lora --config configs/lora_gemma4_e4b.yaml --iters 30 --steps-per-report 5
mlx_lm.lora --config configs/lora_gemma4_e4b.yaml
mlx_lm.fuse --model mlx-community/gemma-4-e4b-it-4bit \
            --adapter-path out/lora-e4b --save-path out/gemma4-e4b-tw

# 8. 消融
python scripts/run_ablation.py --suite all --iters 40

# 9. 微調後評測 + 路由分析
mlx_lm.server --model out/gemma4-e4b-tw --host 127.0.0.1 --port 1234
twinkle-eval --config configs/eval_gemma4_e4b_tuned.yaml
python scripts/inspect_router_mlx.py --dump-modules
python scripts/inspect_router_mlx.py --compare-lang --save reports/router_before.json
```

完整的逐步說明、讀書計畫、驗收條件與常見坑，見 **`week2_執行手冊.md`**。

---

## 三個換模型後最重要的數字

1. **E4B 有 38% 的參數在 Per-Layer Embeddings**（2.82B / 7.46B）。
   官方標的「4.5B effective / 8B with embeddings」兩個數字都對，只是算的東西不同 ——
   報告裡要講清楚用的是哪一個。

2. **logits 是第二大戶**。Gemma 4 vocab = 262,144；seq=2048 時 logits 要 3.00 GiB，
   對 E4B 而言幾乎和權重（3.7 GiB）一樣大，比活化（0.63 GiB）大 5 倍。

3. **dense 的活化比 MoE 大**（8.34 vs 6.11 GiB @ seq2048），
   儘管 MoE 的總參數是它的 3.4 倍 —— 因為 MoE 每 token 只過 8/128 個專家。

---

## 一致性規則（違反就整週白做）

| 規則 | 為什麼 |
|---|---|
| `enable_thinking` 訓練與評測必須同開或同關 | Gemma 4 版的「reasoning_effort 要同檔位」；不一致則微調前後對比失真 |
| 兩份評測 config 只能差 `model.name` 一行 | 否則準確率差值不是微調造成的 |
| `max_seq_len` 用 Step 4 印出的涵蓋率重新決定 | 換 tokenizer 後分布會變，不能沿用 Week 1 的 2048 |
| `seed=42` 貫穿 shuffle / split / sampling | 可複現是提案的交付承諾 |
| 準確率跨硬體可比；吞吐與峰值記憶體不可比 | Metal 與 CUDA 的數字要分欄放 |
