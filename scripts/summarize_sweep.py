#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_sweep.py — 把 sweep_checkpoints.sh 的結果彙整成一張表 + 一張圖

主指標是**無法解析率**（格式保留率的反面），不是準確率。
Week 2 的結論是「掉的分數幾乎全部來自格式」，所以這條曲線在問的是
「微調到第幾步，模型開始不照 \\box{} 格式回答」。

同時報 macro（科目平均，可和 Week 2 的數字對照）與 micro（逐題加權）。
twinkle-eval 只算 macro，但三科題數是 768/139/129，兩者差很多。

用法：
    python3 scripts/summarize_sweep.py
    python3 scripts/summarize_sweep.py --no-plot
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def subject_sizes():
    """從 parquet 讀每科題數，用來算 micro 平均。"""
    try:
        import pandas as pd
    except ImportError:
        return {}
    sizes = {}
    for f in (ROOT / "datasets" / "ikala__tmmluplus").glob("*.parquet"):
        try:
            sizes[f.stem] = len(pd.read_parquet(f))
        except Exception:
            pass
    return sizes


def load_one(path, sizes):
    d = json.loads(Path(path).read_text())
    ds = d["dataset_results"]
    key = next(iter(ds))
    rows = ds[key]["results"]

    per = []
    for r in rows:
        stem = Path(r["file"]).stem
        n = sizes.get(stem)
        if n is None:                       # parquet 讀不到就用 unparsed 反推
            n = round(r["unparsed_count"] / r["unparsed_rate"]) if r["unparsed_rate"] else 0
        per.append({
            "subject": stem, "n": n,
            "acc": r["accuracy_mean"],
            "unparsed": r["unparsed_rate"],
            "unparsed_count": r["unparsed_count"],
        })

    tot = sum(p["n"] for p in per) or 1
    return {
        "model": d["config"]["model"]["name"],
        "n_questions": tot,
        "macro_acc": ds[key]["average_accuracy"],
        "macro_unparsed": ds[key]["average_unparsed_rate"],
        "micro_acc": sum(p["acc"] * p["n"] for p in per) / tot,
        "micro_unparsed": sum(p["unparsed_count"] for p in per) / tot,
        "minutes": d.get("duration_seconds", 0) / 60,
        "per_subject": per,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", default="results/sweep")
    ap.add_argument("--base", default="results/results_20260814_0015.json",
                    help="未微調的 baseline，畫成水平參考線")
    ap.add_argument("--out", default="reports/checkpoint_sweep.md")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    sizes = subject_sizes()
    files = sorted((ROOT / args.sweep_dir).glob("ckpt-*.json"))
    if not files:
        raise SystemExit(f"{args.sweep_dir} 底下沒有 ckpt-*.json\n"
                         f"   先跑：bash scripts/sweep_checkpoints.sh")

    rows = []
    for f in files:
        r = load_one(f, sizes)
        r["step"] = int(re.search(r"ckpt-(\d+)", f.name).group(1))
        rows.append(r)
    rows.sort(key=lambda x: x["step"])

    base = None
    bp = ROOT / args.base
    if bp.exists():
        base = load_one(bp, sizes)

    # ---- 對帳：這一段是為了擋住「五份結果其實是同一個模型」 ----
    sig = {(round(r["micro_acc"], 6), round(r["micro_unparsed"], 6)) for r in rows}
    if len(rows) > 1 and len(sig) == 1:
        raise SystemExit(
            "❌ 所有 checkpoint 的結果完全相同。\n"
            "   最可能的原因是評測時 server 上跑的一直是同一個模型。\n"
            "   檢查 results/sweep/*.json 裡的 config.model.name 是否各不相同。")
    names = [r["model"] for r in rows]
    if len(set(names)) != len(names):
        raise SystemExit(f"❌ 有結果檔的 model.name 重複，代表評測到同一個模型：\n   {names}")

    # ---- 表 ----
    head = (f"{'步數':>6} {'題數':>6} {'嚴格micro':>11} {'嚴格macro':>11} "
            f"{'無法解析micro':>14} {'無法解析macro':>14} {'分鐘':>6}")
    lines = [head, "-" * 80]
    if base:
        lines.append(f"{0:>6} {base['n_questions']:>6} {base['micro_acc']:>11.4f} "
                     f"{base['macro_acc']:>11.4f} {base['micro_unparsed']:>14.4f} "
                     f"{base['macro_unparsed']:>14.4f} {base['minutes']:>6.0f}   ← 未微調")
    for r in rows:
        lines.append(f"{r['step']:>6} {r['n_questions']:>6} {r['micro_acc']:>11.4f} "
                     f"{r['macro_acc']:>11.4f} {r['micro_unparsed']:>14.4f} "
                     f"{r['macro_unparsed']:>14.4f} {r['minutes']:>6.0f}")
    table = "\n".join(lines)
    print(table)

    # ---- 判讀 ----
    u = [r["micro_unparsed"] for r in rows]
    inversions = sum(1 for a, b in zip(u, u[1:]) if b < a - 0.01)
    print()
    if inversions == 0:
        print("▶ 無法解析率隨步數單調上升 —— 格式崩潰是漸進的，可以找到一個安全的步數。")
    else:
        print(f"▶ 有 {inversions} 個非單調點。只有一個可能是雜訊；兩個以上代表單次評測"
              f"變異太大，要開 repeat_runs。")
    if base:
        first_bad = next((r["step"] for r in rows
                          if r["micro_unparsed"] > base["micro_unparsed"] + 0.05), None)
        if first_bad:
            print(f"▶ 無法解析率首次超過 baseline +5pt：第 {first_bad} 步。")
        else:
            print("▶ 所有 checkpoint 的無法解析率都沒有明顯高過 baseline。")

    # ---- 圖 ----
    png = ROOT / "reports" / "checkpoint_sweep.png"
    plotted = False
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            matplotlib.rcParams["axes.unicode_minus"] = False
            import matplotlib.pyplot as plt
            steps = [r["step"] for r in rows]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(steps, u, "o-", color="crimson")
            if base:
                ax[0].axhline(base["micro_unparsed"], ls="--", c="gray", label="base (no FT)")
                ax[0].legend()
            ax[0].set_xlabel("training steps"); ax[0].set_ylabel("unparsed rate")
            ax[0].set_title("format collapse vs training steps"); ax[0].grid(alpha=.3)
            ax[1].plot(steps, [r["micro_acc"] for r in rows], "o-", label="micro")
            ax[1].plot(steps, [r["macro_acc"] for r in rows], "s--", label="macro")
            if base:
                ax[1].axhline(base["micro_acc"], ls="--", c="gray", label="base micro")
            ax[1].set_xlabel("training steps"); ax[1].set_ylabel("strict accuracy")
            ax[1].set_title("accuracy vs training steps"); ax[1].legend(); ax[1].grid(alpha=.3)
            plt.tight_layout(); plt.savefig(png, dpi=140)
            plotted = True
            print(f"\n圖：{png}")
        except Exception as e:
            print(f"\n（畫圖略過：{type(e).__name__}: {e}）")

    # ---- 報告 ----
    out = ROOT / args.out
    md = ["# checkpoint 掃描：訓練步數 vs 格式保留率", "",
          "由 `scripts/sweep_checkpoints.sh` + `scripts/summarize_sweep.py` 產生。", "",
          "```", table, "```", ""]
    if plotted:
        md += [f"![](checkpoint_sweep.png)", ""]
    md += ["## 每個 checkpoint 實際評測的模型（對帳用）", "",
           "| 步數 | model.name |", "|---:|---|"]
    md += [f"| {r['step']} | `{r['model']}` |" for r in rows]
    md.append("")
    out.write_text("\n".join(md))
    print(f"報告：{out}")


if __name__ == "__main__":
    main()
