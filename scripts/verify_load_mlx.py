#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_load_mlx.py — 本機（Apple Silicon / MLX）載入驗證 + roofline 分析

取代 Week 1 的 verify_load.py。做三件事，全部在 24GB M4 Pro 上：

  H1  權重記憶體預測 vs Metal 實測峰值，誤差應 <10%
  E2  用記憶體頻寬 roofline 證明 MoE 每 token 只讀 active 參數
  P1  bf16 vs 4-bit 的權重記憶體對照（E4B 的 bf16 版本機塞得下，不必租卡）

用法：
    # E4B dense（本機主線模型）
    python scripts/verify_load_mlx.py --model mlx-community/gemma-4-e4b-it-4bit

    # 26B-A4B MoE（對照組；記憶體貼邊，先看 §0 的 wired limit 說明）
    python scripts/verify_load_mlx.py --model mlx-community/gemma-4-26B-A4B-it-4bit \\
        --moe --expected-weight-gib 12.5

    # 兩個 4-bit 模型都跑並寫出對照報告
    python scripts/verify_load_mlx.py --both

    # P1：E4B 的 bf16 對照（13.9 GiB，24GB 機器跑得起來）
    python scripts/verify_load_mlx.py --model mlx-community/gemma-4-e4b-it-bf16

輸出：reports/load_verification_gemma.md

§0 24GB 機器的 wired limit
    macOS 預設只讓 GPU 用約 2/3 的統一記憶體（24GB 機器約 16 GiB）。
    26B-A4B 4-bit 權重就要 12.5 GiB，加上 KV cache 與 buffer pool 會很貼邊。
    跑之前先放寬（重開機後失效，不會永久改動系統）：
        sudo sysctl iogpu.wired_limit_mb=20480
    跑完想還原：
        sudo sysctl iogpu.wired_limit_mb=0
    ⚠️ 不要調到超過總記憶體的 ~85%，否則系統會開始換頁，反而更慢甚至當掉。
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
GiB = 1024 ** 3

# M4 Pro 官方統一記憶體頻寬（GB/s，十進位）。
# 若你的機器是 M4 Max/M3 等，用 --bandwidth 覆寫。
DEFAULT_BANDWIDTH_GBPS = 273.0

PROMPTS = [
    "用繁體中文解釋什麼是專家平行（Expert Parallelism），並說明它和資料平行的差別。",
    "台灣的全民健保制度如何運作？請說明保費計算方式。",
    "解釋梯度檢查點（gradient checkpointing）為什麼能省記憶體，代價是什麼。",
]


# ------------------------------------------------------------------ MLX 記憶體 API 相容層
def _mem_api():
    """mlx 的記憶體 API 在 0.21 前後搬過家，這裡兩種都試。"""
    import mlx.core as mx
    for holder in (mx, getattr(mx, "metal", None)):
        if holder is None:
            continue
        peak = getattr(holder, "get_peak_memory", None)
        active = getattr(holder, "get_active_memory", None)
        reset = getattr(holder, "reset_peak_memory", None) or \
            getattr(holder, "clear_cache", None)
        if peak and active:
            return peak, active, (reset or (lambda: None))
    raise RuntimeError(
        "找不到 mlx 的記憶體 API（get_peak_memory / get_active_memory）。"
        "請 `uv pip install -U mlx mlx-lm` 後重試。")


def run_one(model_id, args):
    import mlx.core as mx
    from mlx_lm import load, generate

    get_peak, get_active, reset_peak = _mem_api()
    reset_peak()

    print(f"\n{'='*66}\n載入 {model_id}\n{'='*66}")
    t0 = time.time()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    load_s = time.time() - t0
    after_load = get_active()
    print(f"  載入耗時 {load_s:.1f}s，載入後 active memory {after_load/GiB:.2f} GiB")

    # ---- 參數量與實際 bytes（roofline 的分母就從這裡來）----
    def walk(tree, prefix=""):
        import mlx.core as mx
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                yield from walk(v, f"{prefix}.{i}")
        elif isinstance(tree, mx.array):
            yield prefix, tree

    params = list(walk(model.parameters()))
    total_bytes = sum(a.nbytes for _, a in params)
    total_elems = sum(a.size for _, a in params)
    print(f"  參數張量 {len(params):,} 個；實際佔用 {total_bytes/GiB:.2f} GiB")

    # ---- 生成，量峰值與吞吐 ----
    print(f"\n  生成 {args.max_tokens} tokens…")
    reset_peak()
    outs, tps = [], []
    for p in PROMPTS[:args.n_prompts]:
        msgs = [{"role": "user", "content": p}]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=not args.no_thinking)
        t = time.time()
        text = generate(model, tokenizer, prompt=prompt,
                        max_tokens=args.max_tokens, verbose=False)
        dt = time.time() - t
        n_out = len(tokenizer.encode(text))
        tps.append(n_out / dt)
        outs.append(text)
        print(f"    {n_out:>4} tokens / {dt:>5.1f}s = {n_out/dt:>5.1f} tok/s")
    peak = get_peak()
    thr = sum(tps) / len(tps)
    print(f"\n  平均吞吐 {thr:.1f} tok/s；生成期間峰值記憶體 {peak/GiB:.2f} GiB")

    return dict(
        model=model_id, load_s=round(load_s, 1),
        weight_bytes=total_bytes, weight_gib=total_bytes / GiB,
        param_elems=total_elems,
        after_load_gib=after_load / GiB, peak_gib=peak / GiB,
        tok_per_s=round(thr, 1),
        sample_output=outs[0][:600] if outs else "",
    )


def roofline(res, args, moe_cfg=None):
    """
    E2：解碼是記憶體頻寬受限的 —— 每產生一個 token，該次用到的權重都要
    從統一記憶體讀進運算單元一次。所以
        理論上限 tok/s = 頻寬(GB/s) ÷ 每 token 讀取量(GB)
    對 MoE，「用到的」只有 active 部分；對 dense 就是全部。
    """
    bw = args.bandwidth * 1e9
    total_b = res["weight_bytes"]
    if moe_cfg:
        # active 佔比 = (top_k 個路由專家 + 共享 + 非 FFN) / 全部
        frac = moe_cfg["active_frac"]
        read_moe = total_b * frac
        read_dense = total_b
        return {
            "active_frac": frac,
            "read_per_token_moe_gb": read_moe / 1e9,
            "read_per_token_dense_gb": read_dense / 1e9,
            "ceiling_moe": bw / read_moe,
            "ceiling_dense": bw / read_dense,
            "measured": res["tok_per_s"],
            "pct_of_moe_ceiling": res["tok_per_s"] / (bw / read_moe) * 100,
            "speedup_vs_dense_ceiling": res["tok_per_s"] / (bw / read_dense),
        }
    return {
        "read_per_token_gb": total_b / 1e9,
        "ceiling": bw / total_b,
        "measured": res["tok_per_s"],
        "pct_of_ceiling": res["tok_per_s"] / (bw / total_b) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/gemma-4-e4b-it-4bit")
    ap.add_argument("--both", action="store_true",
                    help="依序跑 E4B dense 與 26B-A4B MoE（皆 4-bit）並寫出對照表")
    ap.add_argument("--moe", action="store_true", help="這個模型是 MoE（走 MoE roofline）")
    ap.add_argument("--active-frac", type=float, default=None,
                    help="MoE 每 token 讀取的權重比例；不給就用 predict_memory 的預設 0.20")
    ap.add_argument("--expected-weight-gib", type=float, default=None,
                    help="predict_memory_gemma.py 算出的權重預測值，用來判定 H1")
    ap.add_argument("--bandwidth", type=float, default=DEFAULT_BANDWIDTH_GBPS,
                    help="統一記憶體頻寬 GB/s（M4 Pro=273, M4 Max=546, M4=120）")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--n-prompts", type=int, default=3)
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    # 兩個模型的預測值（來自 predict_memory_gemma.py，4-bit 全量）
    PRESET = {
        "mlx-community/gemma-4-e4b-it-4bit":
            dict(moe=False, expected=3.7, active_frac=1.0, label="Gemma 4 E4B（dense, 4-bit）"),
        "mlx-community/gemma-4-e4b-it-bf16":
            dict(moe=False, expected=13.9, active_frac=1.0, label="Gemma 4 E4B（dense, bf16）"),
        "mlx-community/gemma-4-26B-A4B-it-4bit":
            dict(moe=True, expected=12.5, active_frac=0.20, label="Gemma 4 26B-A4B（MoE, 4-bit）"),
    }

    # --both 只跑 4-bit 的兩個；bf16 版要另外指定（--model ...-bf16）
    targets = ["mlx-community/gemma-4-e4b-it-4bit",
               "mlx-community/gemma-4-26B-A4B-it-4bit"] if args.both else [args.model]
    results = []
    for m in targets:
        preset = PRESET.get(m, {})
        is_moe = preset.get("moe", args.moe)
        expected = args.expected_weight_gib or preset.get("expected")
        frac = args.active_frac or preset.get("active_frac", 0.20)

        r = run_one(m, args)
        r["label"] = preset.get("label", m)
        r["is_moe"] = is_moe
        r["expected_weight_gib"] = expected
        if expected:
            err = (r["peak_gib"] - expected) / expected * 100
            r["h1_error_pct"] = round(err, 1)
            r["h1_pass"] = abs(err) < 10
            print(f"\n  H1：預測 {expected:.2f} GiB / 實測峰值 {r['peak_gib']:.2f} GiB"
                  f" → 誤差 {err:+.1f}%  {'✅ 成立' if r['h1_pass'] else '⚠️ 超過 10%'}")
        r["roofline"] = roofline(r, args,
                                 {"active_frac": frac} if is_moe else None)
        rl = r["roofline"]
        print("\n  E2 roofline：")
        if is_moe:
            print(f"    每 token 讀取（MoE）  {rl['read_per_token_moe_gb']:.2f} GB"
                  f" → 理論上限 {rl['ceiling_moe']:.1f} tok/s")
            print(f"    假如是 dense（讀全部）{rl['read_per_token_dense_gb']:.2f} GB"
                  f" → 理論上限 {rl['ceiling_dense']:.1f} tok/s")
            print(f"    實測 {rl['measured']:.1f} tok/s"
                  f" = MoE 上限的 {rl['pct_of_moe_ceiling']:.0f}%，"
                  f"dense 上限的 {rl['speedup_vs_dense_ceiling']:.2f} 倍")
        else:
            print(f"    每 token 讀取 {rl['read_per_token_gb']:.2f} GB"
                  f" → 理論上限 {rl['ceiling']:.1f} tok/s")
            print(f"    實測 {rl['measured']:.1f} tok/s = 上限的 {rl['pct_of_ceiling']:.0f}%")
        results.append(r)

    # ---------------------------------------------------------------- 報告
    L = []
    A = L.append
    A("# 本機載入驗證與 roofline 分析（Gemma 4 / MLX / Metal）\n")
    A(f"硬體假設：統一記憶體頻寬 {args.bandwidth:.0f} GB/s"
      f"（用 `--bandwidth` 改）\n")
    A("> ⚠️ **可比性**：這裡量到的吞吐與峰值記憶體是 Metal 上的數字，"
      "不能和租用 CUDA 卡的數字並列。準確率才是跨硬體可比的。\n")

    A("\n## H1 — 權重記憶體預測 vs 實測\n")
    A("| 模型 | 預測（4-bit） | 載入後 | 生成峰值 | 誤差 | 判定 |")
    A("|---|---:|---:|---:|---:|---|")
    for r in results:
        e = r.get("expected_weight_gib")
        A(f"| {r['label']} | {e:.2f} GiB | {r['after_load_gib']:.2f} GiB | "
          f"{r['peak_gib']:.2f} GiB | {r.get('h1_error_pct', 0):+.1f}% | "
          f"{'✅' if r.get('h1_pass') else '⚠️'} |" if e else
          f"| {r['label']} | — | {r['after_load_gib']:.2f} | {r['peak_gib']:.2f} | — | — |")

    A("\n## E2 — 記憶體頻寬 roofline\n")
    A("解碼是記憶體頻寬受限的：每產生一個 token，該次用到的權重就必須從統一記憶體"
      "讀一次。所以 `理論上限 = 頻寬 ÷ 每 token 讀取量`。"
      "MoE 只讀 active 那部分，dense 要讀全部 —— 這是「MoE 真的只算 active 參數」"
      "的物理證據，比「跑起來很快」有力得多。\n")
    A("| 模型 | 每 token 讀取 | 理論上限 | 實測 | 達成率 |")
    A("|---|---:|---:|---:|---:|")
    for r in results:
        rl = r["roofline"]
        if r["is_moe"]:
            A(f"| {r['label']}（MoE 實際） | {rl['read_per_token_moe_gb']:.2f} GB | "
              f"{rl['ceiling_moe']:.1f} tok/s | {rl['measured']:.1f} tok/s | "
              f"{rl['pct_of_moe_ceiling']:.0f}% |")
            A(f"| {r['label']}（假設 dense） | {rl['read_per_token_dense_gb']:.2f} GB | "
              f"{rl['ceiling_dense']:.1f} tok/s | — | 實測是它的 "
              f"{rl['speedup_vs_dense_ceiling']:.2f}× |")
        else:
            A(f"| {r['label']} | {rl['read_per_token_gb']:.2f} GB | "
              f"{rl['ceiling']:.1f} tok/s | {rl['measured']:.1f} tok/s | "
              f"{rl['pct_of_ceiling']:.0f}% |")

    if len(results) == 2:
        a, b = results[0], results[1]
        A("\n## dense vs MoE 實測對照\n")
        A("> E4B 的非嵌入參數（3.97B）≈ 26B-A4B 的 active（3.82B）—— "
          "每 token 計算量相當，所以吞吐差異直接反映架構本身。\n")
        A("| 指標 | E4B dense | 26B-A4B MoE |")
        A("|---|---:|---:|")
        A(f"| 權重實際佔用 | {a['weight_gib']:.2f} GiB | {b['weight_gib']:.2f} GiB |")
        A(f"| 生成峰值 | {a['peak_gib']:.2f} GiB | {b['peak_gib']:.2f} GiB |")
        A(f"| 載入耗時 | {a['load_s']:.1f}s | {b['load_s']:.1f}s |")
        A(f"| 生成吞吐 | {a['tok_per_s']:.1f} tok/s | {b['tok_per_s']:.1f} tok/s |")
        A(f"\n**MoE 用 {b['weight_gib']/a['weight_gib']:.1f} 倍的記憶體，"
          f"換到 {b['tok_per_s']/a['tok_per_s']:.2f} 倍的吞吐。**"
          "這個比值加上微調後的 TMMLU+ 準確率，就是「MoE 在單機情境下划不划算」的答案。\n")

    A("\n## 原始數據\n\n```json")
    A(json.dumps([{k: v for k, v in r.items() if k != "sample_output"}
                  for r in results], ensure_ascii=False, indent=2))
    A("```")

    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / "load_verification_gemma.md"
    out.write_text("\n".join(L), encoding="utf-8")
    (REPORTS / "load_verification_gemma.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已寫入 {out}")


if __name__ == "__main__":
    main()
