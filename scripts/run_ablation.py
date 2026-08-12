#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation.py — 消融實驗跑批器（ch01 H3/H4/H5/H6 + ch08 的三步驟決策流程）

mlx_lm.lora 每 N 步會印一行像這樣的東西：
    Iter 10: Train loss 2.316, Learning Rate 1.000e-04, It/sec 3.198,
             Tokens/sec 655.263, Trained Tokens 20480, Peak mem 10.482 GB
這支程式把同一組設定用不同變因各跑幾十步，把 Peak mem / It/sec / loss 抓下來排成表 ——
這張表就是「實測值 vs 書中公式落差」的右半邊。

⚠️ 實作上不是用 CLI flag 覆寫，而是**每個變因產生一份暫時的 YAML config**。
   原因：`--grad-checkpoint` 是 store_true，沒加就是 False，
   但 mlx_lm.lora 會讓 config 檔的值蓋過「沒被指定」的 CLI 參數 ——
   結果 base config 裡的 `grad_checkpoint: true` 會讓「不開檢查點」那組其實也開著，
   整個消融就白跑了。改寫 config 檔可以完全避開這類優先序問題。

用法：
    python scripts/run_ablation.py --suite checkpoint   # H3 梯度檢查點取捨
    python scripts/run_ablation.py --suite seqlen       # H4 logits 隨 seq 成長
    python scripts/run_ablation.py --suite batch        # H5 活化隨 bs 成長
    python scripts/run_ablation.py --suite lora         # H6 LoRA 掛載範圍
    python scripts/run_ablation.py --suite all --iters 40
    python scripts/run_ablation.py --suite checkpoint --dry-run   # 只印會跑什麼

輸出：reports/ablation_<suite>.md + reports/ablation_<suite>.json
"""

import argparse
import copy
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TMP = ROOT / "out" / "_ablation_tmp"
BASE_CONFIG = "configs/lora_gemma4_12b.yaml"

LINE_RE = re.compile(
    r"Iter\s+(\d+):.*?Train loss\s+([\d.]+).*?It/sec\s+([\d.]+).*?"
    r"Tokens/sec\s+([\d.]+).*?Peak mem\s+([\d.]+)\s*GB", re.S)

# 每個 suite = (釘死的條件, [(標籤, 變因)])
# 覆寫的 key 直接用 YAML 欄位名。
# 釘死的數值是用 predict_memory_gemma.py 的 24GB 預算表挑的，
# 目的是讓**變因的兩端都跑得起來** —— 一端 OOM 就沒有對照可言。
SUITES = {
    "checkpoint": (
        # seq 壓到 512：不開 11.7 GiB / 開 8.2 GiB，兩端在**預設** wired limit 下都活。
        # seq=1024 不開是 16.3 GiB，會頂到 24GB 機器預設的 ~16 GiB 上限；
        # seq=2048 不開是 25.3 GiB，直接 OOM。
        {"max_seq_length": 512, "batch_size": 1},
        [("不開梯度檢查點", {"grad_checkpoint": False}),
         ("開梯度檢查點", {"grad_checkpoint": True})],
    ),
    "seqlen": (
        {"batch_size": 1, "grad_checkpoint": True},
        [("seq=512", {"max_seq_length": 512}),
         ("seq=1024", {"max_seq_length": 1024}),
         ("seq=2048", {"max_seq_length": 2048}),
         ("seq=4096", {"max_seq_length": 4096})],      # 預估 15.0 GiB，貼邊
    ),
    "batch": (
        # seq 壓到 512，讓 bs 一路開到 8 都不 OOM（bs=8 預估 15.0 GiB）
        {"max_seq_length": 512, "grad_checkpoint": True},
        [("bs=1", {"batch_size": 1}),
         ("bs=2", {"batch_size": 2}),
         ("bs=4", {"batch_size": 4}),
         ("bs=8", {"batch_size": 8})],
    ),
    "lora": (
        {"max_seq_length": 1024, "batch_size": 1, "grad_checkpoint": True},   # 9.7 GiB
        [("LoRA 4 層", {"num_layers": 4}),
         ("LoRA 16 層", {"num_layers": 16}),
         ("LoRA 全部層", {"num_layers": -1})],
    ),
}

HYPOTHESIS = {
    "checkpoint": "H3：開 full checkpointing 後活化記憶體應下降 >85%"
                  "（seq512 預測 3.76 → 0.26 GiB），step time 增加約 30–40%",
    "seqlen": "H4：logits（seq × 262,144 × 6 bytes）隨 seq 線性成長，"
              "是峰值記憶體裡被低估的大戶（512→4096：0.75 → 6.00 GiB）",
    "batch": "H5：活化與 logits 隨 batch size 線性成長；因為有 Flash 與滑動視窗，"
             "隨 seq 也接近線性而非 ch01 說的平方",
    "lora": "H6：可訓練參數隨掛載層數線性成長（4/16/48 層 ≈ 1.8M/7.1M/21.3M），"
            "但優化器記憶體相對 5.9 GiB 的權重仍是雜訊",
}


def build_config(base, pin, overrides, label_slug, iters, report_every):
    cfg = copy.deepcopy(base)
    cfg.update(pin)
    cfg.update(overrides)
    cfg["iters"] = iters
    cfg["steps_per_report"] = report_every
    cfg["adapter_path"] = str(TMP / label_slug)
    # 消融只看記憶體與速度，不需要中途評估與存檔 —— 省時間也避免污染峰值
    cfg["steps_per_eval"] = iters + 1
    cfg["save_every"] = iters + 1
    cfg["resume_adapter_file"] = None
    return cfg


def run_one(slug, label, cfg, args):
    TMP.mkdir(parents=True, exist_ok=True)
    cfg_path = TMP / f"cfg_{slug}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    cmd = [sys.executable, "-m", "mlx_lm.lora", "--config", str(cfg_path)]

    key = {k: cfg[k] for k in
           ("max_seq_length", "batch_size", "grad_checkpoint", "num_layers")
           if k in cfg}
    print(f"\n{'─'*66}\n▶ {label}   {key}")
    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return {"label": label, "status": "dry_run", **key}

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                              timeout=args.timeout)
    except subprocess.TimeoutExpired:
        print("  ⏱ 逾時，跳過")
        return {"label": label, "status": "timeout", **key}
    out = (proc.stdout or "") + (proc.stderr or "")
    dur = time.time() - t0

    rows = LINE_RE.findall(out)
    if not rows:
        tail = "\n".join(out.strip().splitlines()[-12:])
        oom = any(s in out.lower() for s in
                  ("out of memory", "insufficient memory", "failed to allocate"))
        print(f"  ❌ {'OOM' if oom else '抓不到 Iter 行'}。尾端輸出：\n{tail}")
        return {"label": label, "status": "oom" if oom else "parse_error",
                "tail": tail, "seconds": round(dur, 1), **key}

    keep = rows[len(rows) // 2:] or rows          # 跳過暖機，取後半平均
    f = lambda i: sum(float(r[i]) for r in keep) / len(keep)
    its = f(2)
    res = {
        "label": label, "status": "ok", **key,
        "loss": round(f(1), 4),
        "it_per_s": round(its, 3),
        "s_per_step": round(1 / its, 3) if its else None,
        "tokens_per_s": round(f(3), 1),
        "peak_mem_gb": round(max(float(r[4]) for r in rows), 3),
        "peak_mem_gib": round(max(float(r[4]) for r in rows) * 1e9 / 1024 ** 3, 3),
        "seconds": round(dur, 1), "n_reports": len(rows),
    }
    print(f"  ✅ peak {res['peak_mem_gb']} GB ({res['peak_mem_gib']} GiB) | "
          f"{res['s_per_step']} s/step | {res['tokens_per_s']} tok/s | "
          f"loss {res['loss']}")
    return res


def report(suite, results, args, pin):
    L, A = [], None
    A = L.append
    A(f"# 消融實驗：{suite}\n")
    A(f"**假設** — {HYPOTHESIS.get(suite, '')}\n")
    A(f"設定：base config `{args.config}`，每個變因跑 {args.iters} 步"
      f"（取後半平均，跳過暖機）")
    A(f"釘死的條件（變因以外都不變，且保證兩端都跑得起來）："
      f"`{pin}`\n")
    A("| 變因 | 峰值記憶體 | s/step | tokens/s | train loss | 狀態 |")
    A("|---|---:|---:|---:|---:|---|")
    ok = [r for r in results if r.get("status") == "ok"]
    for r in results:
        if r.get("status") != "ok":
            tag = {"oom": "❌ OOM", "timeout": "⏱ 逾時",
                   "dry_run": "— dry-run"}.get(r["status"], "⚠️ " + r["status"])
            A(f"| {r['label']} | — | — | — | — | {tag} |")
        else:
            A(f"| {r['label']} | {r['peak_mem_gib']:.2f} GiB | {r['s_per_step']:.3f} | "
              f"{r['tokens_per_s']:.0f} | {r['loss']:.4f} | ✅ |")

    if len(ok) >= 2:
        a, b = ok[0], ok[-1]
        dmem = (b["peak_mem_gib"] - a["peak_mem_gib"]) / a["peak_mem_gib"] * 100
        dspd = (b["s_per_step"] - a["s_per_step"]) / a["s_per_step"] * 100
        A(f"\n**{a['label']} → {b['label']}**："
          f"峰值記憶體 {dmem:+.1f}%，每步耗時 {dspd:+.1f}%。\n")

    if suite == "seqlen" and len(ok) >= 2:
        A("\n### logits 線性度檢查（H4 的關鍵表）\n")
        A("| seq | 峰值記憶體 | 相對最短 seq 的增量 | logits 理論增量 | 差額（＝活化的貢獻） |")
        A("|---:|---:|---:|---:|---:|")
        base = ok[0]
        bseq = base["max_seq_length"]
        for r in ok:
            s = r["max_seq_length"]
            theo = (s - bseq) * 262144 * 6 / 1024 ** 3
            meas = r["peak_mem_gib"] - base["peak_mem_gib"]
            A(f"| {s} | {r['peak_mem_gib']:.2f} GiB | {meas:+.2f} GiB | "
              f"{theo:+.2f} GiB | {meas-theo:+.2f} GiB |")
        A("\n最後一欄若遠小於 logits 的理論增量，代表峰值成長主要來自 logits"
          "而不是活化 —— **H4 成立**，而且順便說明了為什麼 262K vocab 是這個"
          "實驗的隱形成本。\n")

    if suite == "checkpoint" and len(ok) == 2:
        no_ck, ck = ok[0], ok[1]
        A("\n### H3 判定\n")
        dmem = (ck["peak_mem_gib"] - no_ck["peak_mem_gib"]) / no_ck["peak_mem_gib"] * 100
        dspd = (ck["s_per_step"] - no_ck["s_per_step"]) / no_ck["s_per_step"] * 100
        A(f"- 峰值記憶體：{no_ck['peak_mem_gib']:.2f} → {ck['peak_mem_gib']:.2f} GiB"
          f"（{dmem:+.1f}%）")
        A(f"- 每步耗時：{no_ck['s_per_step']:.3f} → {ck['s_per_step']:.3f} s"
          f"（{dspd:+.1f}%）")
        A(f"\n注意峰值的降幅會**小於**活化本身的降幅（預測 −93%），"
          f"因為權重 5.7 GiB 與 logits 都不受檢查點影響 —— "
          f"這個差別本身就值得在報告裡說明：**梯度檢查點只動活化那一項**。\n")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"ablation_{suite}.md").write_text("\n".join(L), encoding="utf-8")
    (REPORTS / f"ablation_{suite}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(L))
    print(f"\n✅ 已寫入 reports/ablation_{suite}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES) + ["all"], default="checkpoint")
    ap.add_argument("--config", default=BASE_CONFIG)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--report-every", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true", help="只印會跑什麼，不真的跑")
    args = ap.parse_args()

    base_path = ROOT / args.config
    if not base_path.exists():
        sys.exit(f"❌ 找不到 base config：{base_path}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    for s in (list(SUITES) if args.suite == "all" else [args.suite]):
        pin, variants = SUITES[s]
        print(f"\n{'='*66}\n消融 suite：{s}\n{'='*66}")
        print(f"  假設：{HYPOTHESIS.get(s, '')}")
        print(f"  釘死：{pin}")
        print("  ⚠️ 跑之前先放寬 GPU wired limit，"
              "seq=4096 與 bs=8 這兩格會頂到 ~15 GiB：")
        print("     sudo sysctl iogpu.wired_limit_mb=20480")
        results = []
        for i, (lbl, ov) in enumerate(variants):
            slug = f"{s}_{i:02d}"
            cfg = build_config(base, pin, ov, slug,
                               args.iters, args.report_every)
            results.append(run_one(slug, lbl, cfg, args))
        report(s, results, args, pin)


if __name__ == "__main__":
    main()
