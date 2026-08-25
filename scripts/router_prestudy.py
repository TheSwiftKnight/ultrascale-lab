#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router_prestudy.py — MoE 路由前置作業（moe_routing_分析.md §5.1 / week3 手冊 Step 8）

一次做完三件事：
  1. 逐 prompt 存 counts（inspect_router_mlx.py 只存加總，做不了 prompt 層級檢定）
  2. prompt 層級 permutation test —— 10 組配對 → 2^10 = 1024 種標籤交換**窮舉**，
     精確 p 值。這直接解決 moe_routing_分析.md §3.4 限制 2（token 不獨立、
     bootstrap p 偏樂觀）。
  3. 繁簡對照 —— 分辨「中文專家」是**語言**專家還是**字符集／tokenization**專家。
     驗收：繁簡 KL 應顯著小於中英 KL；若兩者同量級，§5 的提案立論要重寫。

用法（兩段式）：
    # 在 Mac 上（需要 mlx_lm 與模型，約 5–10 分鐘，只做 forward 不生成）
    .venv/bin/python scripts/router_prestudy.py

    # 只重算分析（不載模型，任何機器可跑）
    python scripts/router_prestudy.py --analyze-only

產出：
    reports/router_prestudy_counts.json     原始逐 prompt counts
    reports/router_prestudy_analysis.json   檢定結果（本篇表格的來源數字）

【設計注解：簡體 prompt 是「逐字轉換、用詞不變」】
    刻意不把「程式碼」改成大陸慣用的「代码」——那是詞彙變異，不是字符集變異。
    只做字級簡化（碼→码、檔→档），中英對照裡的「主題配對」邏輯才能沿用：
    繁簡兩版連用詞都相同，唯一變因就是字符集（以及它引起的 tokenization 差異）。
"""

import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"

# 與 inspect_router_mlx.PROMPT_PAIRS 同序的三語版本（trad 必須逐字對應 simp）
PROMPTS = {
    "trad": [
        "請解釋什麼是遞迴，並舉一個例子。",
        "台灣的健保制度如何運作？",
        "計算 15 的階乘，並說明過程。",
        "寫一段 Python 程式碼讀取 CSV 檔。",
        "為什麼天空是藍色的？",
        "比較資本主義與社會主義的差異。",
        "如何煮出一碗好吃的牛肉麵？",
        "解釋量子糾纏的基本概念。",
        "說明台灣的地形對氣候的影響。",
        "什麼是通貨膨脹？對民生有何影響？",
    ],
    "simp": [
        "请解释什么是递回，并举一个例子。",
        "台湾的健保制度如何运作？",
        "计算 15 的阶乘，并说明过程。",
        "写一段 Python 程式码读取 CSV 档。",
        "为什么天空是蓝色的？",
        "比较资本主义与社会主义的差异。",
        "如何煮出一碗好吃的牛肉面？",
        "解释量子纠缠的基本概念。",
        "说明台湾的地形对气候的影响。",
        "什么是通货膨胀？对民生有何影响？",
    ],
    "en": [
        "Explain what recursion is, with an example.",
        "How does Taiwan's national health insurance work?",
        "Compute the factorial of 15 and show your work.",
        "Write Python code to read a CSV file.",
        "Why is the sky blue?",
        "Compare capitalism and socialism.",
        "How do you cook a good bowl of beef noodle soup?",
        "Explain the basic concept of quantum entanglement.",
        "Explain how Taiwan's terrain affects its climate.",
        "What is inflation and how does it affect daily life?",
    ],
}

N_PAIRS = 10


# ---------------- 收集（需要模型；邏輯沿用 inspect_router_mlx 的攔截） ----------------

def collect_per_prompt(model, tokenizer, prompts, name_filter, top_k, n_experts,
                       thinking=True):
    """逐 prompt 存 counts：同一個 tap 連續跑，每個 prompt 結束後取累計差分。"""
    import mlx.core as mx
    from inspect_router_mlx import RouterTap

    tap = RouterTap(model, name_filter, top_k, n_experts)
    if not tap.names:
        raise RuntimeError("找不到 router 模組，先跑 inspect_router_mlx.py --dump-modules")

    per_prompt, n_tokens = [], []
    prev = [dict() for _ in tap.names]
    with tap:
        for p in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False, enable_thinking=thinking)
            ids = mx.array([tokenizer.encode(text)])
            out = model(ids)
            mx.eval(out)
            n_tok = ids.shape[-1]
            n_tokens.append(n_tok)
            # 差分 = 這個 prompt 的 counts
            snap = [dict(c) for c in tap.counts]
            delta = []
            for l in range(len(snap)):
                d = {k: snap[l].get(k, 0) - prev[l].get(k, 0) for k in snap[l]}
                delta.append({str(k): v for k, v in d.items() if v > 0})
            per_prompt.append(delta)
            prev = snap
            # 對帳：這個 prompt 的選擇次數必須 == token 數 × top_k × 層數
            got = sum(sum(d.values()) for d in delta)
            want = n_tok * top_k * len(tap.names)
            assert got == want, f"對帳失敗：攔到 {got}，理論 {want}（prompt={p[:20]}…）"
    return {"per_prompt_counts": per_prompt, "n_tokens": n_tokens,
            "layer_names": tap.names}


def run_collect(args):
    from mlx_lm import load
    print(f"載入 {args.model} …")
    model, tokenizer = load(args.model)

    payload = {"model": args.model, "n_experts": args.n_experts, "top_k": args.top_k,
               "prompts": PROMPTS, "sets": {}}
    for key, label in (("trad", "繁體"), ("simp", "簡體"), ("en", "英文")):
        print(f"\n跑{label} prompt（{len(PROMPTS[key])} 個，只 forward 不生成）…")
        r = collect_per_prompt(model, tokenizer, PROMPTS[key], args.router_name,
                               args.top_k, args.n_experts, not args.no_thinking)
        payload["sets"][key] = r
        print(f"  {label}: {sum(r['n_tokens'])} tokens，{len(r['layer_names'])} 層")
    payload["n_layers"] = len(payload["sets"]["trad"]["layer_names"])

    out = ROOT / "reports" / "router_prestudy_counts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 已寫入 {out}")
    return payload


# ---------------- 分析（不需要模型） ----------------

def to_vec(c, n_experts):
    return [int(c.get(str(e), c.get(e, 0))) for e in range(n_experts)]


def add_vec(a, b):
    return [x + y for x, y in zip(a, b)]


def kl_kt(ca, cb, alpha=0.5):
    """add-0.5（Krichevsky–Trofimov）平滑 KL(a‖b)，與 analyze_router_lang.py 一致。"""
    k = len(ca)
    ta, tb = sum(ca) + alpha * k, sum(cb) + alpha * k
    s = 0.0
    for e in range(k):
        p = (ca[e] + alpha) / ta
        q = (cb[e] + alpha) / tb
        s += p * math.log(p / q)
    return s


def jsd(ca, cb):
    ta, tb = sum(ca) or 1, sum(cb) or 1
    p = [x / ta for x in ca]
    q = [x / tb for x in cb]
    m = [(p[e] + q[e]) / 2 for e in range(len(p))]

    def half(a):
        return sum(a[e] * math.log(a[e] / m[e]) for e in range(len(a))
                   if a[e] > 0 and m[e] > 0)

    return 0.5 * half(p) + 0.5 * half(q)


def layer_totals(per_prompt, layer, n_experts, which=None):
    """把（選中的）prompt 的第 layer 層 counts 加總成向量。which=None 表示全部。"""
    idx = range(len(per_prompt)) if which is None else which
    tot = [0] * n_experts
    for i in idx:
        tot = add_vec(tot, to_vec(per_prompt[i][layer], n_experts))
    return tot


def permutation_test(set_a, set_b, n_layers, n_experts):
    """配對標籤交換的窮舉 permutation test（2^10 = 1024 種指派，含恆等）。

    H0：對每一組主題配對，兩個版本（a_i, b_i）可交換 —— 「語言/字符集」這個
    標籤與路由無關。統計量對「a 全體 vs b 全體」計算，所以 token 不獨立
    不影響檢定的有效性（交換的單位是整個 prompt）。
    """
    a_pp, b_pp = set_a["per_prompt_counts"], set_b["per_prompt_counts"]
    n = len(a_pp)
    assert n == len(b_pp) == N_PAIRS

    # 每層預先把 per-prompt 向量算好，內圈只做加法
    A = [[to_vec(a_pp[i][l], n_experts) for i in range(n)] for l in range(n_layers)]
    B = [[to_vec(b_pp[i][l], n_experts) for i in range(n)] for l in range(n_layers)]

    per_layer = []
    obs_means = None
    null_layer = [[] for _ in range(n_layers)]
    null_mean_stat = []
    masks = list(itertools.product((0, 1), repeat=n))   # 1 = 這組交換
    for mask in masks:
        stats_kl, stats_jsd = [], []
        for l in range(n_layers):
            ca, cb = [0] * n_experts, [0] * n_experts
            for i in range(n):
                if mask[i]:
                    ca, cb = add_vec(ca, B[l][i]), add_vec(cb, A[l][i])
                else:
                    ca, cb = add_vec(ca, A[l][i]), add_vec(cb, B[l][i])
            stats_kl.append(kl_kt(ca, cb))
            stats_jsd.append(jsd(ca, cb))
        if all(m == 0 for m in mask):
            obs_kl, obs_jsd = stats_kl, stats_jsd
        for l in range(n_layers):
            null_layer[l].append((stats_kl[l], stats_jsd[l]))
        null_mean_stat.append(sum(stats_kl) / n_layers)

    obs_mean = sum(obs_kl) / n_layers
    p_global = sum(1 for s in null_mean_stat if s >= obs_mean) / len(masks)
    null_mean = (sum(null_mean_stat) - obs_mean) / (len(masks) - 1)

    for l in range(n_layers):
        kls = [x[0] for x in null_layer[l]]
        jss = [x[1] for x in null_layer[l]]
        per_layer.append({
            "layer": l,
            "kl_kt": obs_kl[l],
            "jsd": obs_jsd[l],
            "kl_kt_null_mean": (sum(kls) - obs_kl[l]) / (len(masks) - 1),
            "p_kl_kt": sum(1 for x in kls if x >= obs_kl[l]) / len(masks),
            "p_jsd": sum(1 for x in jss if x >= obs_jsd[l]) / len(masks),
        })

    return {
        "n_permutations": len(masks),
        "mean_kl_kt": obs_mean,
        "mean_jsd": sum(obs_jsd) / n_layers,
        "null_mean_kl_kt": null_mean,
        "snr": obs_mean / null_mean if null_mean else float("inf"),
        "p_global": p_global,
        "n_layers_p_lt_0.01": sum(1 for r in per_layer if r["p_kl_kt"] < 0.01),
        "n_layers_p_lt_0.05": sum(1 for r in per_layer if r["p_kl_kt"] < 0.05),
        "per_layer": per_layer,
    }


def top_contrib(ca, cb, n_experts, k=3, alpha=0.5):
    """KT 平滑下 KL 的專家逐點貢獻，取前 k 名。"""
    ta, tb = sum(ca) + alpha * n_experts, sum(cb) + alpha * n_experts
    contrib = []
    for e in range(n_experts):
        p = (ca[e] + alpha) / ta
        q = (cb[e] + alpha) / tb
        contrib.append(p * math.log(p / q))
    return sorted(range(n_experts), key=lambda e: -contrib[e])[:k]


def expert_overlap(sets, n_layers, n_experts, k=3):
    """繁 vs 英、簡 vs 英 各選 top-k 貢獻專家，看兩份名單的重疊。
    若是「語言專家」，換字符集不應換人；重疊率低就是 tokenization 專家的徵兆。"""
    rows = []
    for l in range(n_layers):
        t = layer_totals(sets["trad"]["per_prompt_counts"], l, n_experts)
        s = layer_totals(sets["simp"]["per_prompt_counts"], l, n_experts)
        e = layer_totals(sets["en"]["per_prompt_counts"], l, n_experts)
        top_t = top_contrib(t, e, n_experts, k)
        top_s = top_contrib(s, e, n_experts, k)
        inter = len(set(top_t) & set(top_s))
        rows.append({"layer": l, "top_trad_vs_en": top_t, "top_simp_vs_en": top_s,
                     "overlap": inter / k})
    return {"k": k, "mean_overlap": sum(r["overlap"] for r in rows) / n_layers,
            "per_layer": rows}


def split_half_stability(set_zh, set_en, n_layers, n_experts, k=3, n_splits=50, seed=42):
    """§5.3 Step 2 的預演：把 10 組配對切成 5/5 兩半，各選 top-k，看重疊率。
    重疊率低 → 10-prompt 材料選出的專家不穩定，方案 C 的挑選要更多材料。"""
    rng = random.Random(seed)
    overlaps = []
    for _ in range(n_splits):
        idx = list(range(N_PAIRS))
        rng.shuffle(idx)
        h1, h2 = idx[:N_PAIRS // 2], idx[N_PAIRS // 2:]
        for l in range(n_layers):
            t1 = top_contrib(layer_totals(set_zh["per_prompt_counts"], l, n_experts, h1),
                             layer_totals(set_en["per_prompt_counts"], l, n_experts, h1),
                             n_experts, k)
            t2 = top_contrib(layer_totals(set_zh["per_prompt_counts"], l, n_experts, h2),
                             layer_totals(set_en["per_prompt_counts"], l, n_experts, h2),
                             n_experts, k)
            overlaps.append(len(set(t1) & set(t2)) / k)
    return {"k": k, "n_splits": n_splits, "mean_overlap": sum(overlaps) / len(overlaps)}


def run_analyze(counts):
    n_l, n_e = counts["n_layers"], counts["n_experts"]
    sets = counts["sets"]

    print("=" * 78)
    print("MoE 前置作業分析（prompt 層級 permutation，2^10 = 1024 窮舉）")
    print("=" * 78)

    res = {"n_layers": n_l, "n_experts": n_e,
           "n_tokens": {k: sum(v["n_tokens"]) for k, v in sets.items()}}

    comparisons = {}
    for key, (a, b, label) in {
        "zh_en": ("trad", "en", "中英（繁體 vs 英文）"),
        "trad_simp": ("trad", "simp", "繁簡（繁體 vs 簡體）"),
        "simp_en": ("simp", "en", "簡英（簡體 vs 英文）"),
    }.items():
        r = permutation_test(sets[a], sets[b], n_l, n_e)
        comparisons[key] = r
        print(f"\n【{label}】")
        print(f"  平均 KL(KT) = {r['mean_kl_kt']:.4f}｜H0 平均 = {r['null_mean_kl_kt']:.4f}"
              f"｜訊噪比 {r['snr']:.1f}×")
        print(f"  全域 p = {r['p_global']:.4f}（最小可得 {1/r['n_permutations']:.4f}）"
              f"｜p<0.01 的層 {r['n_layers_p_lt_0.01']}/{n_l}"
              f"｜p<0.05 的層 {r['n_layers_p_lt_0.05']}/{n_l}")
    res["comparisons"] = comparisons

    ov = expert_overlap(sets, n_l, n_e)
    res["expert_overlap_trad_simp"] = ov
    print(f"\n【top-3 貢獻專家重疊（繁vs英 對照 簡vs英）】平均 {ov['mean_overlap']:.0%}")

    sh = split_half_stability(sets["trad"], sets["en"], n_l, n_e)
    res["split_half_stability"] = sh
    print(f"【split-half 穩定性（§5.3 Step 2 預演）】top-3 重疊平均 {sh['mean_overlap']:.0%}"
          f"（<60% 代表 10-prompt 材料不夠選專家）")

    # ---- 驗收判定 ----
    ze, ts = comparisons["zh_en"], comparisons["trad_simp"]
    ratio = ts["mean_kl_kt"] / ze["mean_kl_kt"] if ze["mean_kl_kt"] else float("nan")
    print("\n" + "=" * 78)
    print(f"驗收：繁簡 KL 應顯著小於中英 KL")
    print(f"  繁簡/中英 KL 比值 = {ratio:.2f}")
    if ts["p_global"] >= 0.05 and ze["p_global"] < 0.01:
        verdict = ("charset_irrelevant：繁簡路由差異與 H0 不可分、中英顯著 → "
                   "「中文專家」是語言/內容專家，不是字符集專家。§5 立論成立")
    elif ratio < 0.5 and ze["p_global"] < 0.01:
        verdict = ("mostly_language：繁簡差異存在但明顯小於中英 → 主要是語言專家，"
                   "夾雜字符集成分。§5 立論大致成立，挑專家時建議用繁簡都穩定的交集")
    elif ze["p_global"] >= 0.05:
        verdict = ("zh_en_not_significant：中英差異在 prompt 層級檢定下不顯著 → "
                   "Week 2 的 E4 結論被 token 不獨立灌水，§3 要重寫")
    else:
        verdict = ("charset_expert：繁簡 KL 與中英同量級 → 找到的是字符集/tokenization "
                   "專家，不是語言專家。§5 的四個提案立論要重寫")
    print(f"  判定：{verdict}")
    res["verdict"] = verdict
    res["trad_simp_over_zh_en_ratio"] = ratio

    out = ROOT / "reports" / "router_prestudy_analysis.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已寫入 {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--router-name", default=None)
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--analyze-only", action="store_true",
                    help="只重算分析（讀 reports/router_prestudy_counts.json，不載模型）")
    args = ap.parse_args()

    counts_path = ROOT / "reports" / "router_prestudy_counts.json"
    if args.analyze_only:
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
    else:
        counts = run_collect(args)
    run_analyze(counts)


if __name__ == "__main__":
    main()
