# Week 3 執行手冊

> **這週在做什麼**：Week 2 微調把模型搞崩了（−17.4 pt）。Week 3 換到 Gemma 3 + Colab T4，
> 回答三個問題：**Q1** 記憶體優化框架省在哪、數學有沒有變？**Q2** Week 2 的崩潰是不是超參數開太大？
> **Q4** Shadow-FT（在 base 上訓、把 delta 搬到 instruct）能不能保住輸出格式？
>
> 全流程在 `notebooks/week3_colab.ipynb`，原始數據在 `results/`、`reports/`。
> 配套：`week3_0_上週提問回覆.md`、`moe_routing_分析.md`。

---

## 前提：Week 2 交接的五個事實

1. **mlx-lm 的 LoRA `scale` 是直接乘數**（沒有除以 rank）→ Week 2 的有效 scaling 是 20，業界常規的 10 倍。
2. **`k_proj`/`v_proj` 沒訓到**：E4B 後 18 層共用 KV，adapter 被靜默略過，實際只訓了 q 和 o。
3. **E4（中英路由差異）訊噪比 6.6×**：雜訊底線要用 bootstrap 實測（0.1063），解析式會低估 3.5 倍。
4. **準確率要 macro＋micro 並列**：科目平均、逐題平均。
5. **MLX 4-bit 實測 4.50 bit/參數**（group 64）；bitsandbytes nf4 要另量，不能沿用。

---

## Step 1：本機收尾

**在做什麼**：把 Week 2 訓練留下的 5 個 checkpoint（200–1000 步）各評一次，看崩潰（無法解析）是「什麼時候」發生的；並重跑路由分析確認數字可重現。

- [x] 五個 checkpoint 都有結果（`results/sweep/ckpt-*.json`，2026-08-18 跑完）
- [x] 無法解析率隨步數上升：**23.7% → 11.5% → 32.6% → 39.7% → 40.4%**。
      大趨勢上升、只有 400 步一個非單調點（允許範圍內）
- [x] 200 步時 23.7%，未達 40% → Stage A 用 200 步的設定**不需要**下調
- [x] 路由分析可重現：訊噪比 **6.63× / 7.12× / 4.35×**（`reports/router_lang_analysis.json`）

---

## Step 2–3：環境與資料

**在做什麼**：Colab T4 上裝套件、掛 Drive；把 Week 2 同一批 8,000 筆抽樣改用 Gemma 3 的 chat template 重新渲染。Gemma 3 沒有 thinking channel，CoT 改成 `<think>…</think>` 內嵌（`inline`），另備 `drop`、`inline_mixed` 兩版。

- [x] T4 到手、九個套件就緒、Drive 掛載
- [x] `torch.cuda.is_bf16_supported()` = False（T4 特性，預期內，這是主線用 Unsloth 的理由）
- [x] 6 個 jsonl 產生；樣本開頭無雙 `<bos>`（notebook 已剝掉 template 加的那個）
- [x] 資料洩漏檢查通過（**全量、題目全文比對**；評測三科題目不在訓練資料中，
      含 TMMLU+ 跨科目重複題的題目層級排除）

---

## Step 4：框架對照（答 Q1）

**在做什麼**：同一組 LoRA 訓練（r=16、30 步）在三個框架各跑一次，量峰值記憶體、速度、loss。
重點對帳：**優化框架的正當性建立在 loss 等價上**——省記憶體但 loss 對不上，就是換了數學，不是優化。

| 框架 | 峰值 GiB | s/step | 末 loss | 結果 |
|---|---:|---:|---:|---|
| HF+peft fp16 | 6.39 | 1.81 | 0.0 | ❌ 溢位崩潰（loss 歸零＝NaN 判定），**如預測** |
| HF+peft fp32 | 8.00 | 7.52 | 1.5702 | ✅ 能跑但慢 |
| Unsloth | 7.50 | 2.02 | 1.8211 | ✅ 快 **3.7×**|

- [x] 三組都有數字（fp16 的失敗是有效紀錄；它的 1.81 s/step 是在算溢位垃圾，不可引用）
- [x] Unsloth 峰值 7.50 < HF fp32 8.00
- [x] loss 對帳：末值差 16%（超過 10% 門檻），但單步 loss 雜訊大；**後半段平均差 6.9%**，過門檻。
      結論寫「大致等價、不完全等價」，差異可能來自 optimizer（adamw_torch vs adamw_8bit）與精度路徑
- [x] **加答**：Unsloth 只比 HF fp32 省 0.50 GiB ≪ 1.50 GiB（logits+CE 的大小）→
      **fused CE 在 Gemma 3 上沒生效**，省的是 Triton kernel／梯度檢查點

---

## Step 5：LoRA 參數掃描（答 Q2）

**在做什麼**：固定其他條件，只動一個變因，把 Week 2「超參數開太大」的成分量出來。
Stage A 掃 scaling（0.5→20）、Stage B 掃 rank（8→64）、Stage C 比 target module。
每組訓 200 步後用快速評測（每科 100 題）排序。

**Stage A 實測**：

| | A1 (0.5) | A2 (1.0) | A3 (2.0 常規) | A4 (4.0) | A5 (20 = Week 2) |
|---|---:|---:|---:|---:|---:|
| 無法解析 | 18.3% | 21.7% | 17.3% | 16.0% | **12.7%** |
| 正確率 | 31.0% | 29.7% | 29.7% | 33.0% | 33.7% |

- [x] 每組 `trainable_params` 合理（r=16 all-linear = 29.8M，符合預測）
- [ ] ❌ 無法解析率**不單調**，方向甚至相反：A5 沒有重現 Week 2 的 ~40% 崩潰，A3 也沒 < 5%

**判讀（重要）**：落在事前寫好的第三分支——**「全都不崩」→ Week 2 的崩潰主因不是超參數**，
另有原因（候選：Gemma 4 的 thinking-channel 模板、訓練步數、twinkle-eval 無種子洗牌的雜訊）。

**Stage B（rank）**：29.0% → 29.7% → 32.3% → 33.0%（r=8→64），輕微隨 rank 上升，
與 Shadow-FT 論文「常規 LoRA rank 越大越傷」**相反**，但差距多在 300 題的雜訊內。
**Stage C（target）**：attn-only 30.0% vs all-linear 29.7%，無可辨差異。

---

## Step 6：Shadow-FT（答 Q4）

**在做什麼**：Shadow-FT（arXiv 2505.12716）主張「在 base（`-pt`）上訓 adapter，原封搬到 instruct（`-it`）」
能避免破壞 instruct 的輸出格式。
先驗前提：論文立論要求 base 和 instruct 權重幾乎一樣（σ < 0.05）。

- [ ] ❌ **前提不成立**：σ（僅 language_model）= **0.1075**，全張量 0.0942，都是論文門檻的 ~2 倍。
      差最大的是多層 `mlp.down_proj`（σ 0.30–0.35）。明細在 `reports/shadow_ft_sigma_per_tensor.json`
- [x] `shadow_graft` 逐張量相加的 assert 通過（掛載數 = adapter 模組數）
- [x] 融合改為掛 adapter 對稱比較（免費版不融合，量化處理兩邊一致）

**三條假設全數不成立**（完整評測 1,036 題）：

| 假設 | 判準 | 實測 | 判定 |
|---|---|---|---|
| S1 格式保住 | D1 無法解析 < 5% 且明顯低於 A3 | D1 **19.4%** vs A3 18.3% | ❌ |
| S2 知識增益 | D1 正確率 ≥ 未微調 base_it | 34.1% < **38.0%** | ❌ |
| S3 對超參數不敏感 | \|D1−D2\| ≪ \|A3−A5\| | **11.0 pt** vs 2.5 pt，方向相反 | ❌ |

**判讀**：前提驗證失敗 → 方法效果差，兩者互相印證。這是一個完整、可報告的負結果：
**Shadow-FT 不適用於 gemma-3-4b 的 pt↔it 這一對**（論文沒測過這對；我們先驗了 σ 所以不意外）。

---

## Step 7：評測

- [x] 未微調 `-it` 無法解析率：完整版 **4.2%** < 5%（快速版 6.0% 略超，僅用於排序）
- [x] 題數正確：快速 300、完整 1,036
- [x] 快速與完整評測**排序一致**：base(38.0) > A5(34.3) ≈ D1(34.1) > A3(31.8) > D2(23.1)
- [x] 開跑前 3 題防呆通過（`results/probe_base_it.json`）
- [x] 抽查無法解析樣本：不是 512 token 截斷，是 `\box{}` 裡填了內容而非選項字母 → 數字真實

**完整評測主表**：

| run | 嚴格正確率 (micro) | 無法解析 |
|---|---:|---:|
| base_it（未微調） | **38.0%** | 4.2% |
| A5（scaling 20） | 34.3% | 11.2% |
| D1（Shadow-FT, scaling 2） | 34.1% | 19.4% |
| A3（scaling 2 常規） | 31.8% | 18.3% |
| D2（Shadow-FT, scaling 20） | 23.1% | 45.7% |

**跨週比較守則**：Week 2（Gemma 4 E4B、thinking、max_tokens 2048）和 Week 3（Gemma 3 4B、max_tokens 512）
**分表呈現，不並列**；只有同一份 Week 3 config 下的組間比較才有因果意義。
快速評測只用來排序，結論一律以完整評測為準。

---

## Step 8–9：待辦

1. **Step 8 MoE 前置**（不吃 GPU）：逐 prompt 存 counts、prompt 層級 permutation test、**繁簡對照**
   （分辨語言專家 vs 字符集專家——繁簡 KL 應顯著小於中英 KL，否則 `moe_routing_分析.md` §5 要重寫立論）。
   注意：專家權重是融合 3-D 張量，沒有 `experts.N.gate_proj` 路徑可指，先用遮罩法驗證。
2. **Step 9 總結**：產出 `week3_執行總結.md`（兩句話總結 → 五個事實 → Q1 框架表 → Q2 Stage A 判讀 →
   Q4 σ + S1–S3 → MoE 摘要 → 下週規劃），並把 Drive 的 `results/`、`reports/` 同步回 repo。

---

## 假設狀態總表（更新）

| 編號 | 主題 | 狀態 |
|---|---|---|
| H1/H3/H4/H5/E1–E5 | 記憶體與 MoE 各假設 | ✅ Week 1/2 已證實（見舊版手冊） |
| H6 | LoRA 掛哪些模組 | ⚠️ Stage C 補了下游指標：attn-only ≈ all-linear，差異在雜訊內 |
| P2 | Flash Attention 效益 | ⏳ T4 不支援 FA2，需 Colab Pro（L4/A100） |
| P1 | 4-bit vs bf16 準確率 | ⏳ 同上，需 bf16 基準線 |
| **新增 Q1** | Unsloth 省的是不是 logits | ✅ **證實不是**：差 0.50 ≪ 1.50 GiB，fused CE 未生效 |
| **S1** | Shadow-FT 保住格式 | ❌ 不成立（19.4% vs 判準 <5%） |
| **S2** | Shadow-FT 知識增益 | ❌ 不成立（34.1% < 38.0%） |
| **S3** | Shadow-FT 對超參數不敏感 | ❌ 不成立（方向相反） |
| **新增** | Week 2 崩潰主因 = 超參數 | ❌ **被推翻**：scaling 掃到 20 也不崩，另有原因（待查） |

---

## Sources

- Shadow-FT（arXiv 2505.12716，本地 `shadow-ft.pdf`）
- [Fine-tune Gemma 3 with Unsloth](https://unsloth.ai/blog/gemma3)（fp16 溢位成因）
- [google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt) / [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)
