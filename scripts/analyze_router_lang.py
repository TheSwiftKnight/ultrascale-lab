#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_router_lang.py — 對 inspect_router_mlx.py 存下來的路由統計做二次分析

做四件 inspect_router_mlx.py 沒做的事：

1. 【修正雜訊底線】原本用解析式 (K-1)/(2N) 當 plug-in KL 的偏差估計。
   那個式子是 N >> K 的漸近展開，本例 N/K = 2072/128 ≈ 16，而且 40% 的專家
   幾乎沒被選到，漸近條件不成立。改用參數化 bootstrap：在 H0（中英同分布）
   下從合併分布重抽同樣大小的兩份樣本，直接量 plug-in KL 的經驗分布。
   → 實測 null ≈ 0.106，是解析式 0.0306 的 3.5 倍。用解析式會得到 23×，
     用 bootstrap 是 6.6× —— 差 3.5 倍，結論方向相同但強度不能誇大。

2. 【換兩個更穩健的統計量】
   - KL_KT：add-0.5（Krichevsky–Trofimov）平滑，避免 p_en = 0 的格子被
     eps=1e-9 放大成天文數字。
   - JSD：對稱且有界（0 ~ ln2），不會被單一零格主導。

3. 【逐層 p-value】每層做一次 bootstrap 檢定。

4. 【定位「中文專家」】計算每個專家對 KL 的逐點貢獻
   c_e = p_zh(e) · log(p_zh(e)/p_en(e))，排序後看要幾個專家才涵蓋 80/90% 的 KL。
   這決定了「只微調中文相關專家」這個提案的參數預算。

用法：
    python scripts/analyze_router_lang.py \
        --before reports/router_before.json \
        --after  reports/router_after.json \
        --out    reports/router_lang_analysis
"""

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------- 基本工具 ----------

def counts_to_list(c, n_experts):
    """inspect_router_mlx 存的是 Counter 轉成的 dict，key 可能是 str 也可能是 int。"""
    return [int(c.get(str(e), c.get(e, 0))) for e in range(n_experts)]


def normalize(c):
    t = sum(c)
    if t == 0:
        return [0.0] * len(c)
    return [x / t for x in c]


def kl_plugin(ca, cb, eps=1e-9):
    """原本 inspect_router_mlx.py 用的估計式，保留以便對帳。"""
    p, q = normalize(ca), normalize(cb)
    return sum((p[e] + eps) * math.log((p[e] + eps) / (q[e] + eps)) for e in range(len(p)))


def kl_smoothed(ca, cb, alpha=0.5):
    """add-alpha（alpha=0.5 即 Krichevsky–Trofimov）平滑後的 KL(zh ‖ en)。"""
    k = len(ca)
    ta, tb = sum(ca) + alpha * k, sum(cb) + alpha * k
    p = [(ca[e] + alpha) / ta for e in range(k)]
    q = [(cb[e] + alpha) / tb for e in range(k)]
    return sum(p[e] * math.log(p[e] / q[e]) for e in range(k))


def jsd(ca, cb):
    """Jensen-Shannon divergence，對稱、上界 ln2。"""
    p, q = normalize(ca), normalize(cb)
    m = [(p[e] + q[e]) / 2 for e in range(len(p))]

    def half(a):
        return sum(a[e] * math.log(a[e] / m[e]) for e in range(len(a)) if a[e] > 0 and m[e] > 0)

    return 0.5 * half(p) + 0.5 * half(q)


def multinomial(probs, n, rng):
    """用累積分布 + 二分搜尋抽 n 次，避免依賴 numpy。"""
    k = len(probs)
    cum, s = [], 0.0
    for p in probs:
        s += p
        cum.append(s)
    out = [0] * k
    for _ in range(n):
        r = rng.random() * s
        lo, hi = 0, k - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        out[lo] += 1
    return out


# ---------- 主分析 ----------

def analyze_language(before, n_boot, seed):
    rng = random.Random(seed)
    ne = before["zh"]["n_experts"]
    nl = before["zh"]["n_layers"]

    per_layer = []
    for l in range(nl):
        cz = counts_to_list(before["zh"]["counts"][l], ne)
        ce = counts_to_list(before["en"]["counts"][l], ne)
        nz, nen = sum(cz), sum(ce)

        obs = {
            "kl_plugin": kl_plugin(cz, ce),
            "kl_kt": kl_smoothed(cz, ce),
            "jsd": jsd(cz, ce),
        }

        pooled = normalize([cz[e] + ce[e] for e in range(ne)])
        null = {k: [] for k in obs}
        for _ in range(n_boot):
            a = multinomial(pooled, nz, rng)
            b = multinomial(pooled, nen, rng)
            null["kl_plugin"].append(kl_plugin(a, b))
            null["kl_kt"].append(kl_smoothed(a, b))
            null["jsd"].append(jsd(a, b))

        row = {"layer": l, "n_zh": nz, "n_en": nen}
        for k, v in obs.items():
            nl_ = null[k]
            row[k] = v
            row[f"{k}_null_mean"] = sum(nl_) / len(nl_)
            row[f"{k}_snr"] = v / (sum(nl_) / len(nl_))
            row[f"{k}_p"] = (1 + sum(1 for x in nl_ if x >= v)) / (n_boot + 1)

        # 專家層級：KL 的逐點貢獻
        eps = 1e-9
        p, q = normalize(cz), normalize(ce)
        contrib = [(p[e] + eps) * math.log((p[e] + eps) / (q[e] + eps)) for e in range(ne)]
        total = sum(contrib)
        order = sorted(range(ne), key=lambda e: -contrib[e])
        cum, n80, n90 = 0.0, None, None
        for i, e in enumerate(order, 1):
            cum += contrib[e]
            if n80 is None and cum >= 0.8 * total:
                n80 = i
            if n90 is None and cum >= 0.9 * total:
                n90 = i
                break
        row["n_experts_for_80pct_kl"] = n80
        row["n_experts_for_90pct_kl"] = n90
        row["top_experts"] = [
            {
                "expert": e,
                "p_zh": p[e],
                "p_en": q[e],
                "kl_contribution": contrib[e],
            }
            for e in order[:5]
        ]
        # 中文傾向專家：使用率高於平均，且中文是英文的兩倍以上
        ideal = 1.0 / ne
        row["zh_leaning"] = [e for e in range(ne) if p[e] >= 2 * q[e] and p[e] >= ideal]
        row["zh_exclusive"] = [e for e in range(ne) if q[e] == 0 and p[e] >= ideal]

        per_layer.append(row)

    summary = {}
    for k in ("kl_plugin", "kl_kt", "jsd"):
        summary[f"{k}_mean"] = sum(r[k] for r in per_layer) / nl
        summary[f"{k}_null_mean"] = sum(r[f"{k}_null_mean"] for r in per_layer) / nl
        summary[f"{k}_snr"] = summary[f"{k}_mean"] / summary[f"{k}_null_mean"]
        summary[f"{k}_n_layers_p_lt_0.01"] = sum(1 for r in per_layer if r[f"{k}_p"] < 0.01)
    summary["analytic_floor_(K-1)/(2N)"] = (ne - 1) / (2 * per_layer[0]["n_zh"])
    summary["n_layers"] = nl
    summary["n_experts"] = ne
    summary["n_bootstrap"] = n_boot
    summary["mean_n_experts_for_80pct_kl"] = sum(r["n_experts_for_80pct_kl"] for r in per_layer) / nl
    summary["mean_n_experts_for_90pct_kl"] = sum(r["n_experts_for_90pct_kl"] for r in per_layer) / nl
    summary["total_zh_leaning"] = sum(len(r["zh_leaning"]) for r in per_layer)
    summary["total_zh_exclusive"] = sum(len(r["zh_exclusive"]) for r in per_layer)

    return {"summary": summary, "per_layer": per_layer}


def lora_budget(h, moe_inter, n_layers, rank, per_layer_experts):
    """一個專家有 gate/up/down 三個矩陣；回傳可訓練參數與 Adam 狀態大小。"""
    per_expert = 2 * (h * rank + rank * moe_inter) + (moe_inter * rank + rank * h)
    n = n_layers * per_layer_experts
    params = n * per_expert
    return {
        "experts_selected": n,
        "params_per_expert": per_expert,
        "trainable_params": params,
        "optimizer_gib_at_16B": params * 16 / 2 ** 30,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="reports/router_before.json")
    ap.add_argument("--after", default="reports/router_after.json")
    ap.add_argument("--out", default="reports/router_lang_analysis")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    # 26B-A4B 的架構常數，與 predict_memory_gemma.py 一致
    ap.add_argument("--hidden", type=int, default=2816)
    ap.add_argument("--moe-inter", type=int, default=704)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    before = json.loads((ROOT / args.before).read_text())
    res = analyze_language(before, args.n_boot, args.seed)
    s = res["summary"]

    # LoRA 預算表
    budgets = {}
    for k in (3, 4, 8, 16, 128):
        budgets[f"top{k}_per_layer"] = lora_budget(
            args.hidden, args.moe_inter, s["n_layers"], args.rank, k
        )
    router_params = s["n_layers"] * args.hidden * s["n_experts"]
    budgets["router_only_full_ft"] = {
        "trainable_params": router_params,
        "optimizer_gib_at_16B": router_params * 16 / 2 ** 30,
    }
    res["lora_budget"] = budgets

    after_path = ROOT / args.after
    if after_path.exists():
        after = json.loads(after_path.read_text())
        res["finetune_kl"] = after.get("finetune_kl")
        res["finetune_kl_avg"] = after.get("finetune_kl_avg")
        res["after_adapter"] = after.get("adapter")

    out_json = ROOT / f"{args.out}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, ensure_ascii=False, indent=2))

    # 主控台摘要
    print("=" * 78)
    print("E4 中英路由差異 —— 用 bootstrap 重估雜訊底線")
    print("=" * 78)
    print(f"{'統計量':<12}{'觀測平均':>10}{'H0 平均':>10}{'訊噪比':>8}{'p<0.01 的層數':>14}")
    for k, label in (("kl_plugin", "KL(plug-in)"), ("kl_kt", "KL(add-0.5)"), ("jsd", "JSD")):
        print(f"{label:<12}{s[k+'_mean']:>10.4f}{s[k+'_null_mean']:>10.4f}"
              f"{s[k+'_snr']:>8.1f}x{s[k+'_n_layers_p_lt_0.01']:>10}/{s['n_layers']}")
    print()
    print(f"解析式雜訊底線 (K-1)/(2N) = {s['analytic_floor_(K-1)/(2N)']:.4f}"
          f"  → 低估了 {s['kl_plugin_null_mean']/s['analytic_floor_(K-1)/(2N)']:.1f} 倍")
    print()
    print(f"平均需要 {s['mean_n_experts_for_80pct_kl']:.1f} 個專家涵蓋 80% 的 KL"
          f"（90%：{s['mean_n_experts_for_90pct_kl']:.1f}）")
    print(f"中文傾向專家共 {s['total_zh_leaning']} 個 / {s['n_layers']*s['n_experts']}"
          f"，其中英文完全沒用到的 {s['total_zh_exclusive']} 個")
    print()
    print("LoRA r=%d 預算：" % args.rank)
    for k, v in budgets.items():
        if "trainable_params" in v:
            print(f"  {k:<22} {v['trainable_params']/1e6:>7.1f}M  優化器 {v['optimizer_gib_at_16B']:.2f} GiB")
    print()
    print(f"寫出：{out_json}")


if __name__ == "__main__":
    main()
