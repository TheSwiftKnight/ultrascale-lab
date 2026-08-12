#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data_gemma.py — Week 2 資料管線（Gemma 4 版，取代 Week 1 的 prepare_data.py）

清理／去重的邏輯與 Week 1 完全相同（同樣的 SEED、門檻、MinHash 參數），
所以「原始 50,000 → 清理 → 去重」那幾個數字應該和 Week 1 一模一樣 ——
**這本身就是一個驗收點**：如果不一樣，代表管線被改壞了。

真正變的是 Step 4 之後：

    Week 1（GPT-OSS）                     Week 2（Gemma 4）
    ─────────────────────────────────────────────────────────────
    Harmony template                      Gemma 4 canonical template
    <|channel|>analysis / final           <|channel>thought ... <channel|>
    reasoning_effort="medium"             enable_thinking=True
    assistant 用 "thinking" 欄位           assistant 用 "reasoning" 欄位
    vocab 201,088                         vocab 262,144
    輸出 HF dataset                        HF dataset + MLX jsonl

⚠️ **train/eval 一致性規則（Gemma 4 版的「reasoning_effort 要同檔位」）**：
   `enable_thinking` 控制的是系統回合裡那個 `<|think|>` token。
   assistant 的 `reasoning` 欄位**不管開不開都會被渲染**，
   所以如果訓練資料用 enable_thinking=True、評測時卻沒開，
   模型看到的前綴就不一樣，比較會失真。本檔預設 True，評測設定檔也必須開。

用法：
    python scripts/prepare_data_gemma.py                    # 預設 8000 筆
    python scripts/prepare_data_gemma.py --n-sample 5000
    python scripts/prepare_data_gemma.py --no-minhash       # 只做精確去重（快很多）
    python scripts/prepare_data_gemma.py --no-xet           # 連線不穩時用
    python scripts/prepare_data_gemma.py --no-thinking      # 消融組：不訓練思考通道

輸出：
    data/train/, data/val/              HF dataset（給 CUDA / HF Trainer 路線）
    data/mlx/train.jsonl, valid.jsonl   MLX 格式（給 mlx_lm.lora，本機主線）
    reports/data_stats_gemma.json       每一步的筆數與 token 長度統計
    reports/sample_rendered_gemma.txt   一筆完整 Gemma 4 格式樣本
    reports/token_length_hist_gemma.png Token 長度分布圖（需 matplotlib）
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

# ⚠️ 這些環境變數必須在 import huggingface_hub / datasets **之前**設定
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)      # huggingface_hub 1.0+ 已棄用
if "--no-xet" in sys.argv:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

from datasets import load_dataset

# ------------------------------------------------------------------ 常數
SEED = 42
TOKENIZER_ID = "google/gemma-4-E4B-it"   # E4B 與 26B-A4B 共用同一組 tokenizer；
                                         # 兩者的 chat template 在 enable_thinking=True
                                         # 這條路徑上輸出完全相同（已逐字比對過）
DATASET_ID = "twinkle-ai/tw-reasoning-instruct-50k"
MIN_CHARS, MAX_CHARS = 20, 8000          # 與 Week 1 相同，確保前三步數字可對照
MINHASH_THRESHOLD = 0.85
NUM_PERM = 64

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SIMPLIFIED_MARKERS = (
    "习县这来对开关门时东说话记为业产没" "个们么样车马鸟鱼见贝页风飞"
    "长张问间闻阳陈邓刘赵孙韩汉军农写让证识语读"
)

# Gemma 4 canonical template 的標記（用來驗證 template 真的套上了）
MARK_THINK_SYS = "<|think|>"
MARK_THOUGHT_OPEN = "<|channel>thought"
MARK_THOUGHT_CLOSE = "<channel|>"
MARK_TURN_MODEL = "<|turn>model"


def count_simplified(s: str) -> int:
    return sum(1 for ch in s if ch in SIMPLIFIED_MARKERS)


def load_with_retry(dataset_id: str, split: str, retries: int = 3):
    """下載資料集，對付不穩定連線；Xet 持續失敗時自動退回一般 HTTPS 重啟。"""
    xet_on = not os.environ.get("HF_HUB_DISABLE_XET")
    for attempt in range(1, retries + 1):
        try:
            return load_dataset(dataset_id, split=split)
        except Exception as e:
            msg = str(e)
            print(f"  ⚠️ 第 {attempt}/{retries} 次下載失敗：{type(e).__name__}: {msg[:160]}")
            transport = any(k in msg for k in (
                "CAS", "reconstruct", "decoding response body",
                "Connection", "Timeout", "timed out", "IncompleteRead"))
            if attempt < retries and transport:
                time.sleep(5 * attempt)
                continue
            if xet_on and transport:
                print("\n  ❌ Xet 傳輸持續失敗，改用一般 HTTPS 重新啟動…\n")
                os.execve(sys.executable, [sys.executable] + sys.argv + ["--no-xet"],
                          {**os.environ, "HF_HUB_DISABLE_XET": "1"})
            raise


def get_user_prompt(ex: dict) -> str:
    if ex.get("input"):
        return ex["input"].strip()
    for c in ex.get("conversations") or []:
        if c.get("from") in ("human", "user"):
            return (c.get("value") or "").strip()
    return ""


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=8000)
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--no-minhash", action="store_true")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--no-xet", action="store_true")
    ap.add_argument("--no-thinking", action="store_true",
                    help="消融組：丟掉 think 欄位，只訓練最終答案")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    use_thinking = not args.no_thinking
    DATA.mkdir(exist_ok=True)
    (DATA / "mlx").mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    stats = {"config": {
        "seed": SEED, "dataset": DATASET_ID, "tokenizer": TOKENIZER_ID,
        "enable_thinking": use_thinking, "n_sample": args.n_sample,
        "val_ratio": args.val_ratio, "minhash": not args.no_minhash,
        "min_chars": MIN_CHARS, "max_chars": args.max_chars,
    }}

    # ---------------------------------------------------- Step 1 載入與確認
    print("\n[Step 1] 載入與確認")
    ds = load_with_retry(DATASET_ID, "train")
    print(f"  欄位：{ds.column_names}")
    print(f"  筆數：{len(ds):,}")
    stats["step1_loaded"] = len(ds)
    stats["columns"] = ds.column_names

    # ---------------------------------------------------- Step 2 清理
    print("\n[Step 2] 清理（門檻與 Week 1 相同 → 數字應完全一致）")
    reasons = Counter()

    def keep(ex):
        prompt = get_user_prompt(ex)
        think = (ex.get("think") or "").strip()
        output = (ex.get("output") or "").strip()
        if not prompt:
            reasons["缺 user prompt"] += 1; return False
        if not think:
            reasons["think 空白"] += 1; return False
        if not output:
            reasons["output 空白"] += 1; return False
        total = len(prompt) + len(think) + len(output)
        if total < MIN_CHARS:
            reasons["過短"] += 1; return False
        if total > args.max_chars:
            reasons["過長"] += 1; return False
        return True

    ds_clean = ds.filter(keep, desc="cleaning", load_from_cache_file=False)
    print(f"  {len(ds):,} → {len(ds_clean):,}（刪除 {len(ds)-len(ds_clean):,}）")
    for r, cnt in reasons.most_common():
        print(f"    - {r}: {cnt:,}")
    stats["step2_clean"] = len(ds_clean)
    stats["step2_drop_reasons"] = dict(reasons)

    print("  簡體殘留掃描…")
    simp = [(i, n) for i, ex in enumerate(ds_clean)
            if (n := count_simplified(get_user_prompt(ex) + (ex.get("output") or ""))) >= 3]
    print(f"    疑似簡體樣本：{len(simp):,} 筆（僅標記，未刪除）")
    stats["step2_suspect_simplified"] = len(simp)
    stats["step2_suspect_simplified_examples"] = simp[:20]

    # ---------------------------------------------------- Step 3 去重
    print("\n[Step 3] 去重")
    seen, keep_idx = set(), []
    for i, ex in enumerate(ds_clean):
        k = get_user_prompt(ex)
        if k not in seen:
            seen.add(k); keep_idx.append(i)
    ds_exact = ds_clean.select(keep_idx)
    print(f"  精確去重：{len(ds_clean):,} → {len(ds_exact):,}")
    stats["step3_exact"] = len(ds_exact)

    if args.no_minhash:
        ds_dedup = ds_exact
        stats["step3_minhash"] = None
    else:
        from datasketch import MinHash, MinHashLSH

        def shingles(text, k=5):
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\s+", "", text)[:400]
            return {text[i:i + k] for i in range(max(1, len(text) - k + 1))}

        lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=NUM_PERM)
        dedup_idx, dropped = [], 0
        for i, ex in enumerate(ds_exact):
            m = MinHash(num_perm=NUM_PERM)
            for sh in shingles(get_user_prompt(ex)):
                m.update(sh.encode("utf-8"))
            if lsh.query(m):
                dropped += 1; continue
            lsh.insert(str(i), m); dedup_idx.append(i)
        ds_dedup = ds_exact.select(dedup_idx)
        print(f"  近似去重：{len(ds_exact):,} → {len(ds_dedup):,}（刪除 {dropped:,}）")
        stats["step3_minhash"] = len(ds_dedup)
    stats["step3_dedup"] = len(ds_dedup)

    # ---------------------------------------------------- Step 4 套 chat template
    print(f"\n[Step 4] 套 Gemma 4 chat template（enable_thinking={use_thinking}）")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

    # 先用一筆探針確認 template 行為，再跑全量（避免跑完兩分鐘才發現格式錯）
    probe = [{"role": "user", "content": "測試"},
             {"role": "assistant", "reasoning": "思考內容", "content": "答案"}]
    probe_txt = tok.apply_chat_template(probe, tokenize=False,
                                        enable_thinking=use_thinking)
    print("  探針輸出：", repr(probe_txt[:120]))
    assert MARK_TURN_MODEL in probe_txt, "template 沒有 model 回合標記，版本可能不對"
    if use_thinking:
        assert MARK_THOUGHT_OPEN in probe_txt and MARK_THOUGHT_CLOSE in probe_txt, \
            "reasoning 欄位沒有被渲染成 thought 通道 —— 檢查 transformers/tokenizer 版本"
        assert MARK_THINK_SYS in probe_txt, \
            "系統回合缺少 <|think|> —— enable_thinking 沒有生效"
    print("  ✅ template 行為符合預期")

    def to_record(ex):
        assistant = {"role": "assistant", "content": (ex.get("output") or "").strip()}
        if use_thinking:
            assistant["reasoning"] = (ex.get("think") or "").strip()
        messages = [{"role": "user", "content": get_user_prompt(ex)}, assistant]
        text = tok.apply_chat_template(messages, tokenize=False,
                                       enable_thinking=use_thinking)
        return {"messages": messages, "text": text}

    ds_text = ds_dedup.map(to_record, desc="templating",
                           remove_columns=ds_dedup.column_names)

    sample = ds_text[0]["text"]
    (REPORTS / "sample_rendered_gemma.txt").write_text(sample, encoding="utf-8")
    print(f"  樣本已寫入 {REPORTS/'sample_rendered_gemma.txt'}")

    # token 長度統計 —— 決定 Week 2 的 max_seq_len
    print("  統計 token 長度（Gemma tokenizer，vocab 262,144）…")
    texts = list(ds_text["text"])
    lens = []
    for i in range(0, len(texts), 1000):
        enc = tok(texts[i:i + 1000], add_special_tokens=False)["input_ids"]
        lens.extend(len(x) for x in enc)
    ls = sorted(lens)
    pct = lambda p: ls[min(int(len(ls) * p), len(ls) - 1)]
    tok_stats = {
        "mean": round(sum(lens) / len(lens), 1),
        "p50": pct(.50), "p90": pct(.90), "p95": pct(.95), "p99": pct(.99),
        "max": ls[-1],
        "over_1024": sum(1 for x in lens if x > 1024),
        "over_1536": sum(1 for x in lens if x > 1536),
        "over_2048": sum(1 for x in lens if x > 2048),
        "over_4096": sum(1 for x in lens if x > 4096),
        "n": len(lens),
    }
    for k, v in tok_stats.items():
        print(f"    {k}: {v}")
    stats["step4_token_length"] = tok_stats

    # 給 max_seq_len 的建議 —— 這是 Week 2 唯一必須「重新決定」的超參數
    for cand in (1024, 1536, 2048, 3072, 4096):
        cover = 1 - sum(1 for x in lens if x > cand) / len(lens)
        mark = " ← 建議" if cover >= 0.99 else ""
        print(f"    max_seq_len={cand:>5}：涵蓋 {cover*100:.2f}%"
              f"（截斷 {sum(1 for x in lens if x > cand):,} 筆）{mark}")
        if cover >= 0.99 and "recommended_max_seq_len" not in stats:
            stats["recommended_max_seq_len"] = cand
    print(f"\n  → 建議 max_seq_len = {stats.get('recommended_max_seq_len')}"
          f"（Week 1 在 GPT-OSS tokenizer 下是 2048，換 tokenizer 後必須重新決定）")

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.hist(lens, bins=80, color="#4C6EF5", edgecolor="none")
            for q, lab in ((tok_stats["p50"], "p50"), (tok_stats["p95"], "p95"),
                           (tok_stats["p99"], "p99")):
                ax.axvline(q, ls="--", lw=1, color="#E8590C")
                ax.text(q, ax.get_ylim()[1] * .9, f" {lab}={q}", fontsize=8, color="#E8590C")
            ax.set_xlabel("tokens (Gemma 4 tokenizer)")
            ax.set_ylabel("count")
            ax.set_title("Token length distribution")
            fig.tight_layout()
            fig.savefig(REPORTS / "token_length_hist_gemma.png", dpi=150)
            print(f"  分布圖：{REPORTS/'token_length_hist_gemma.png'}")
        except Exception as e:
            print(f"  ⚠️ 畫圖略過（{type(e).__name__}: {e}）")

    # ---------------------------------------------------- Step 5 抽樣與切分
    print("\n[Step 5] 抽樣與切分")
    n = min(args.n_sample, len(ds_text))
    small = ds_text.shuffle(seed=SEED).select(range(n))
    split = small.train_test_split(test_size=args.val_ratio, seed=SEED)
    split["train"].save_to_disk(str(DATA / "train"))
    split["test"].save_to_disk(str(DATA / "val"))
    print(f"  抽樣 {n:,} → train {len(split['train']):,} / val {len(split['test']):,}")
    stats.update(step5_sample=n, step5_train=len(split["train"]),
                 step5_val=len(split["test"]))

    # ---------------------------------------------------- Step 5b MLX 格式
    # mlx_lm.lora 吃 train.jsonl / valid.jsonl，每行一個 JSON 物件。
    # 用 {"text": ...} 而非 {"messages": ...}：因為 template 已經套好了，
    # 直接餵 text 可以保證訓練看到的字串和上面那份樣本 100% 一致。
    print("\n[Step 5b] 輸出 MLX 格式")
    for name, part in (("train", split["train"]), ("valid", split["test"])):
        path = DATA / "mlx" / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in part:
                f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
        print(f"  {path}  ({len(part):,} 行)")

    # ---------------------------------------------------- 收尾
    (REPORTS / "data_stats_gemma.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 完成。統計 → {REPORTS/'data_stats_gemma.json'}")
    print("\n驗收點：step1/step2/step3 的數字應與 Week 1 完全相同")
    print("        （50,000 → 49,984 → 49,968 → 49,965）；")
    print("        若不同，代表管線被改動了，先查清楚再往下走。")


if __name__ == "__main__":
    main()
