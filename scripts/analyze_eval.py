#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""analyze_eval.py — 把兩輪 TMMLU+ 評測拆成「格式」與「知識」兩個維度。

用法：
    python scripts/analyze_eval.py \
        results/eval_results_20260814_0015_run0.jsonl \
        results/eval_results_20260814_0637_run0.jsonl

為什麼需要這支：
    twinkle-eval 只給「平均正確率」和「無法解析數」。但 8/14 那兩輪的差距
    （53.98% → 36.57%）幾乎全部來自無法解析率（0.5% → 39.7%），
    直接比正確率會得到「模型變笨了」這個錯誤結論。
    這支程式回答三個 twinkle-eval 不回答的問題：

      1. 只看「有照格式輸出」的題目，正確率是多少？（→ 會有選擇性偏誤）
      2. 把兩輪逐題對齊，只比**同一批題目**，正確率是多少？（→ 排掉偏誤）
      3. 用寬鬆解析把散文裡的答案撈回來，總分會變多少？（→ 上界）

注意：兩輪的 shuffle_options 各自獨立洗牌，所以同一題的正解字母不一定相同。
      這對「答對與否」沒有影響，但不是逐字元相同的重跑。
"""
import json
import re
import sys
from collections import Counter

# smoke 子集的評測順序與題數（twinkle-eval 依 os.walk 順序逐檔評測後append）
SEGMENTS = [("three_principles_of_people", 139), ("taiwanese_hokkien", 129), ("geography_of_taiwan", 768)]

LENIENT_PATTERNS = [
    re.compile(r"box\{\s*([ABCD])\s*\}"),
    re.compile(r"(?:答案是|答案為|正確答案是|應該是|選項)\s*[:：]?\s*([ABCD])"),
]


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def segment(rows):
    out, i = {}, 0
    for name, n in SEGMENTS:
        out[name] = {r["question_id"]: r for r in rows[i:i + n]}
        i += n
    assert i == len(rows), f"題數對不上：預期 {i}，實際 {len(rows)}"
    return out


def lenient(text):
    """只用高信心的規則，寧可少救也不要誤判。"""
    if not text:
        return None
    for pat in LENIENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1)
    s = text.strip()
    if len(s) <= 3 and s and s[0] in "ABCD":
        return s[0]
    return None


def describe(tag, rows):
    n = len(rows)
    unparsed = [r for r in rows if not r.get("predicted_answer")]
    correct = sum(1 for r in rows if r.get("is_correct"))
    parsed = n - len(unparsed)
    print(f"\n=== {tag}  n={n}")
    print(f"  嚴格正確率(微觀)   {correct / n:.4f}")
    print(f"  無法解析          {len(unparsed)} ({len(unparsed) / n:.1%})")
    print(f"  解析成功題上的正確率 {correct / parsed:.4f}   ← 有選擇性偏誤，不能直接比")

    kinds = Counter()
    for r in unparsed:
        out = r.get("llm_output") or ""
        if r.get("usage_completion_tokens", 0) >= 2040:
            kinds["截斷（達 max_tokens）"] += 1
        elif not out.strip():
            kinds["空輸出"] += 1
        elif "box" in out:
            kinds["有 box 但內容不是字母"] += 1
        else:
            kinds["有內容但沒有 box"] += 1
    if kinds:
        print("  無法解析的成因：", dict(kinds))

    rec = [(r, lenient(r.get("llm_output"))) for r in unparsed]
    rec_ok = [(r, g) for r, g in rec if g]
    rec_hit = sum(1 for r, g in rec_ok if g == r["correct_answer"])
    print(f"  寬鬆解析救回      {len(rec_ok)}/{len(unparsed)}，其中答對 {rec_hit}")
    print(f"  寬鬆正確率(微觀)   {(correct + rec_hit) / n:.4f}")

    toks = [r.get("usage_completion_tokens", 0) for r in rows]
    toks.sort()
    print(f"  完成 token  mean={sum(toks) / n:.0f}  p50={toks[n // 2]}  p90={toks[int(.9 * n)]}  "
          f"撞上限={sum(1 for t in toks if t >= 2040)}")
    return unparsed


def per_subject(tag, seg):
    print(f"\n--- {tag} 分科")
    for name, n in SEGMENTS:
        rows = list(seg[name].values())
        unp = sum(1 for r in rows if not r.get("predicted_answer"))
        cor = sum(1 for r in rows if r.get("is_correct"))
        print(f"  {name:32s} n={n:4d}  正確率={cor / n:.4f}  無法解析={unp:3d} ({unp / n:.1%})  "
              f"解析成功題上={cor / max(1, n - unp):.4f}")


def main():
    base_path, tuned_path = sys.argv[1], sys.argv[2]
    base, tuned = load(base_path), load(tuned_path)
    describe("BASE  " + base_path, base)
    describe("TUNED " + tuned_path, tuned)

    B, T = segment(base), segment(tuned)
    per_subject("BASE", B)
    per_subject("TUNED", T)

    # ── 控制對照：逐題對齊，只比同一批題目 ──────────────────────────
    pairs = []
    for name, _ in SEGMENTS:
        for qid in set(B[name]) & set(T[name]):
            pairs.append((B[name][qid], T[name][qid]))
    print(f"\n=== 控制對照（逐題對齊 {len(pairs)} 題）")
    ok = [(b, t) for b, t in pairs if t.get("predicted_answer")]
    bad = [(b, t) for b, t in pairs if not t.get("predicted_answer")]
    print(f"  微調後格式正常的 {len(ok)} 題：")
    print(f"    base  {sum(1 for b, _ in ok if b['is_correct']) / len(ok):.4f}")
    print(f"    tuned {sum(1 for _, t in ok if t['is_correct']) / len(ok):.4f}   ← 差值才是真正的知識增益")
    print(f"  微調後格式壞掉的 {len(bad)} 題，base 的正確率："
          f"{sum(1 for b, _ in bad if b['is_correct']) / len(bad):.4f}   ← 明顯較低 = 這些題本來就比較難")


if __name__ == "__main__":
    main()
