#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_router_mlx.py — 觀察 Gemma 4 26B-A4B 的 MoE 路由行為（ch06 假設 E3 / E4 / E5）

取代 Week 1 的 inspect_router.py（那支走 HF transformers + CUDA，本機跑不動）。
本支用 MLX，在 24GB M4 Pro 上就能跑。

回答三個問題：
  E3  負載是否均衡？（理想值：每個 expert 被選中 8/128 = 6.25% 的 token）
  E4  繁中與英文 prompt 的路由分布是否不同？
  E5  微調前後路由分布是否偏移？

用法：
    # 0. 先看模型結構，確認 router 模組叫什麼（第一次一定先跑這個）
    python scripts/inspect_router_mlx.py --dump-modules

    # E3 負載均衡
    python scripts/inspect_router_mlx.py --save reports/router_before.json

    # E4 中英對照
    python scripts/inspect_router_mlx.py --compare-lang --save reports/router_before.json

    # E5 微調前後對照
    python scripts/inspect_router_mlx.py --adapter out/lora-26b \\
        --save reports/router_after.json --compare-with reports/router_before.json

⚠️ 這支腳本用「攔截 router 模組的輸出」實作，不依賴特定版本的 mlx-lm API。
   如果 --dump-modules 顯示的結構和預期不同，改 --router-name 指定即可。
"""

import argparse
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"

# 語意配對的中英 prompt（E4 用）：同樣的意思、不同語言。
# 路由若只看語意應該相似；若對「語言表面形式」敏感則會分歧。
PROMPT_PAIRS = [
    ("請解釋什麼是遞迴，並舉一個例子。", "Explain what recursion is, with an example."),
    ("台灣的健保制度如何運作？", "How does Taiwan's national health insurance work?"),
    ("計算 15 的階乘，並說明過程。", "Compute the factorial of 15 and show your work."),
    ("寫一段 Python 程式碼讀取 CSV 檔。", "Write Python code to read a CSV file."),
    ("為什麼天空是藍色的？", "Why is the sky blue?"),
    ("比較資本主義與社會主義的差異。", "Compare capitalism and socialism."),
    ("如何煮出一碗好吃的牛肉麵？", "How do you cook a good bowl of beef noodle soup?"),
    ("解釋量子糾纏的基本概念。", "Explain the basic concept of quantum entanglement."),
    ("說明台灣的地形對氣候的影響。", "Explain how Taiwan's terrain affects its climate."),
    ("什麼是通貨膨脹？對民生有何影響？", "What is inflation and how does it affect daily life?"),
]

# mlx-lm 0.31.3 的 models/gemma4_text.py 裡，MoE block 長這樣：
#     self.router  = Router(config)      # class Router，__call__ 回傳 (top_k_indices, top_k_weights)
#     self.experts = Experts(config)
# 所以：
#   1) 認 **class 名稱 == "Router"** 最可靠，比用名字比對安全
#      （用名字比對會誤抓 router.proj 這個 Linear，以及 dense MLP 的 gate_proj）
#   2) 回傳的第一個元素**已經是 top-k 索引**，不是 logits —— 直接數就好，不要再 argpartition
ROUTER_CLASS = "Router"
ROUTER_HINTS = ("router",)


# ------------------------------------------------------------------ 結構探索
def iter_named_modules(mod, prefix=""):
    """走遍 mlx.nn.Module 樹，產生 (name, module)。"""
    yield prefix, mod
    children = getattr(mod, "children", None)
    if callable(children):
        try:
            kids = children()
        except Exception:
            kids = {}
    else:
        kids = {}
    if isinstance(kids, dict):
        items = kids.items()
    else:
        items = enumerate(kids or [])
    for k, v in items:
        name = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            for k2, v2 in v.items():
                yield from iter_named_modules(v2, f"{name}.{k2}")
        elif isinstance(v, (list, tuple)):
            for i, v2 in enumerate(v):
                yield from iter_named_modules(v2, f"{name}.{i}")
        elif hasattr(v, "children") or hasattr(v, "parameters"):
            yield from iter_named_modules(v, name)


def is_router(name, mod, name_filter):
    """判定一個模組是不是 router。優先看 class 名稱，其次才看模組路徑。"""
    if name_filter:
        return name_filter.lower() in name.lower()
    if type(mod).__name__ == ROUTER_CLASS:
        return True
    # 後備：路徑以 .router 結尾（不要匹配 router.proj 這種子模組）
    return name.lower().rsplit(".", 1)[-1] in ROUTER_HINTS


def dump_modules(model, max_lines=160):
    print("\n=== 模型結構（找出 router 模組）===")
    hits = []
    shown = 0
    for name, m in iter_named_modules(model):
        cls = type(m).__name__
        star = " ★ ← router" if is_router(name, m, None) else ""
        if star or shown < max_lines:
            print(f"  {name:<66} {cls}{star}")
            shown += 1
        if star:
            hits.append((name, cls))
    print(f"\n判定為 router 的模組（{len(hits)} 個）：")
    for h, cls in hits[:12]:
        print(f"  {h}   [{cls}]")
    if not hits:
        print("  ⚠️ 沒找到。請看上面的結構，用 --router-name <子字串> 手動指定。")
    else:
        print(f"\n預期數量 = MoE 層數。26B-A4B 應該是 30 個。")
    return hits


# ------------------------------------------------------------------ 攔截
def _layer_sort_key(name):
    """依名稱裡的層號排序，讓 layers.2 排在 layers.10 前面。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


class RouterTap:
    """把每個 router 模組的 __call__ 包起來，記錄 top-k 選擇。

    mlx-lm 的 `Router.__call__` 回傳 `(top_k_indices, top_k_weights)`，
    第一個元素形狀是 (B, L, top_k) —— **已經是索引**。
    所以這裡直接數，不要再對它做 argpartition（那會得到完全錯的分布）。
    為了保險，若回傳的最後一維等於 n_experts（代表拿到的是 logits），
    才退回自己取 top-k。
    """

    def __init__(self, model, name_filter, top_k, n_experts):
        self.top_k, self.n_experts = top_k, n_experts
        self.layers = []                       # [[name, module, original_call]]
        self.mode = {}                         # idx -> "indices" | "logits"
        for name, m in iter_named_modules(model):
            if is_router(name, m, name_filter) and hasattr(m, "parameters"):
                self.layers.append([name, m, None])
        self.layers.sort(key=lambda x: _layer_sort_key(x[0]))
        self.counts = [Counter() for _ in self.layers]

    def __enter__(self):
        import mlx.core as mx

        for idx, entry in enumerate(self.layers):
            _, mod, _ = entry
            orig = mod.__call__
            entry[2] = orig

            def make(i, o):
                def wrapped(x, *a, **kw):
                    out = o(x, *a, **kw)
                    try:
                        arr = out[0] if isinstance(out, (tuple, list)) else out
                        last = arr.shape[-1]
                        if last == self.n_experts and last != self.top_k:
                            # 拿到的是 logits，自己取 top-k
                            self.mode[i] = "logits"
                            flat = arr.reshape(-1, last)
                            k = min(self.top_k, last)
                            sel = mx.argpartition(-flat, k - 1, axis=-1)[:, :k]
                        else:
                            # 已經是索引（mlx-lm gemma4 的情況）
                            self.mode[i] = "indices"
                            sel = arr.reshape(-1, last)
                        mx.eval(sel)
                        c = self.counts[i]
                        for row in sel.tolist():
                            for v in (row if isinstance(row, list) else [row]):
                                c[int(v)] += 1
                    except Exception as e:      # 不要讓觀測本身弄壞 forward
                        self.mode[i] = f"error:{type(e).__name__}"
                    return out
                return wrapped

            mod.__call__ = make(idx, orig)
        return self

    def __exit__(self, *exc):
        for _, mod, orig in self.layers:
            if orig is not None:
                try:
                    mod.__call__ = orig
                except Exception:
                    pass
        return False


def collect(model, tokenizer, prompts, name_filter, top_k, n_experts, thinking=True):
    import mlx.core as mx

    tap = RouterTap(model, name_filter, top_k, n_experts)
    if not tap.layers:
        raise RuntimeError(
            "找不到 router 模組。先跑 `--dump-modules` 看結構，"
            "再用 `--router-name <子字串>` 指定。")
    print(f"  攔截到 {len(tap.layers)} 個 router 模組")

    n_tok = 0
    with tap:
        for p in prompts:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True,
                tokenize=False, enable_thinking=thinking)
            ids = mx.array([tokenizer.encode(text)])
            out = model(ids)
            mx.eval(out)
            n_tok += ids.shape[-1]

    modes = set(tap.mode.values())
    print(f"  攔截模式：{modes or '（沒被呼叫到）'}")
    if any(str(m).startswith("error") for m in modes):
        print("  ⚠️ 有層攔截失敗，結果不完整 —— 先跑 --dump-modules 確認結構")
    total_sel = sum(sum(c.values()) for c in tap.counts)
    expected = n_tok * top_k * len(tap.layers)
    print(f"  收集到 {total_sel:,} 次專家選擇（理論值 {expected:,}，"
          f"比值 {total_sel/expected if expected else 0:.2f}）")
    if expected and not 0.8 < total_sel / expected < 1.25:
        print("  ⚠️ 比值離 1.0 太遠，代表攔到的可能不是 top-k 索引 —— 檢查 --dump-modules")

    return {
        "n_layers": len(tap.layers),
        "layer_names": [n for n, _, _ in tap.layers],
        "n_experts": n_experts,
        "top_k": top_k,
        "n_tokens": n_tok,
        "tap_modes": {str(k): v for k, v in tap.mode.items()},
        "counts": [dict(c) for c in tap.counts],
    }


# ------------------------------------------------------------------ 分析
def summarize(res, label=""):
    n_e, k = res["n_experts"], res["top_k"]
    ideal = k / n_e
    rows = []
    for layer, c in enumerate(res["counts"]):
        total = sum(c.values()) or 1
        shares = [c.get(e, 0) / total for e in range(n_e)]
        cv = statistics.pstdev(shares) / ideal if ideal else 0
        dead = sum(1 for s in shares if s < ideal * 0.1)
        rows.append({"layer": layer, "cv": cv, "dead": dead, "max_share": max(shares)})

    print(f"\n=== E3 路由負載均衡 {label} ===")
    print(f"理想每 expert 佔比 = {k}/{n_e} = {ideal:.2%}")
    print(f"{'層':>3}  {'變異係數':>8}  {'閒置 expert':>10}  {'最熱 expert':>12}")
    for r in rows:
        print(f"{r['layer']:>3}  {r['cv']:>8.2f}  {r['dead']:>10}  {r['max_share']:>11.2%}")
    avg_cv = statistics.mean(r["cv"] for r in rows) if rows else 0
    tot_dead = sum(r["dead"] for r in rows)
    print(f"\n平均變異係數 {avg_cv:.2f}（0 = 完美均衡）；閒置 expert 總數 {tot_dead}"
          f" / {len(rows)*n_e}")
    print("→ E3 判定：", "均衡 ✅" if avg_cv < 0.5 else "不均衡 ⚠️（有專家被冷落）")
    return {"avg_cv": avg_cv, "total_dead": tot_dead, "per_layer": rows, "ideal": ideal}


def kl(a, b, n_experts):
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    eps = 1e-9
    return sum((a.get(e, 0) / ta + eps) *
               math.log((a.get(e, 0) / ta + eps) / (b.get(e, 0) / tb + eps))
               for e in range(n_experts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None, help="LoRA adapter 路徑（E5 微調後用）")
    ap.add_argument("--dump-modules", action="store_true")
    ap.add_argument("--router-name", default=None,
                    help="router 模組名稱的子字串；不給就自動找 router/gate")
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--compare-lang", action="store_true", help="E4：中英對照")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare-with", default=None, help="E5：與先前結果比對")
    ap.add_argument("--no-thinking", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load

    print(f"載入 {args.model}"
          + (f"（+ adapter {args.adapter}）" if args.adapter else "") + " …")
    kwargs = {"adapter_path": args.adapter} if args.adapter else {}
    model, tokenizer = load(args.model, **kwargs)

    if args.dump_modules:
        dump_modules(model)
        return

    zh = [p[0] for p in PROMPT_PAIRS]
    en = [p[1] for p in PROMPT_PAIRS]

    print("\n跑繁中 prompt…")
    res_zh = collect(model, tokenizer, zh, args.router_name, args.top_k,
                     args.n_experts, not args.no_thinking)
    stats = summarize(res_zh, "（繁體中文）")
    payload = {"model": args.model, "adapter": args.adapter,
               "zh": res_zh, "zh_stats": stats}

    if args.compare_lang:
        print("\n跑英文 prompt…")
        res_en = collect(model, tokenizer, en, args.router_name, args.top_k,
                         args.n_experts, not args.no_thinking)
        summarize(res_en, "（英文）")
        kls = [kl(res_zh["counts"][i], res_en["counts"][i], args.n_experts)
               for i in range(res_zh["n_layers"])]
        print("\n=== E4：中英路由分布差異（KL 散度，越大越不同）===")
        for i, v in enumerate(kls):
            print(f"  層 {i:>2}: {v:.4f} {'█' * int(min(v, 1) * 40)}")
        avg = sum(kls) / len(kls)
        print(f"\n平均 KL = {avg:.4f}")
        print("→ E4 判定：",
              "中英路由明顯不同 ✅（支持「語言是可辨識的路由特徵」）"
              if avg > 0.05 else "差異不顯著 ❌")
        payload.update(en=res_en, lang_kl=kls, lang_kl_avg=avg)

    if args.compare_with:
        prev = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
        prev_counts = prev["zh"]["counts"]
        kls = [kl({int(k): v for k, v in prev_counts[i].items()},
                  {int(k): v for k, v in res_zh["counts"][i].items()},
                  args.n_experts) for i in range(min(len(prev_counts), res_zh["n_layers"]))]
        print("\n=== E5：微調前後路由偏移（KL 散度）===")
        for i, v in enumerate(kls):
            print(f"  層 {i:>2}: {v:.4f}")
        avg = sum(kls) / len(kls)
        print(f"\n平均 KL = {avg:.4f}")
        print("→ E5 判定：", "微調確實改變了路由 ✅" if avg > 0.01 else "路由基本未變")
        payload.update(finetune_kl=kls, finetune_kl_avg=avg)

    if args.save:
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 已寫入 {p}")


if __name__ == "__main__":
    main()
