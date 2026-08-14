#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspect_router_mlx.py — 觀察 Gemma 4 26B-A4B 的 MoE 路由行為（ch06 假設 E3 / E4 / E5）

用法：
    python scripts/inspect_router_mlx.py --dump-modules
    python scripts/inspect_router_mlx.py --compare-lang --save reports/router_before.json
    python scripts/inspect_router_mlx.py --adapter out/lora-26b \
        --save reports/router_after.json --compare-with reports/router_before.json

=====================================================================
⚠️ 這支程式踩過三個 MLX 特有的坑，都不會報錯、只會安靜地給錯答案。
   實作時千萬別「簡化」回去：

【坑 1】不要自己走 children() 找模組。
   mlx.nn.Module **是 dict 的子類別**。所以任何
       if isinstance(v, dict): ...
       elif isinstance(v, Module): ...
   的寫法，第一個分支會吃掉每一個 Module，程式就只看得到模組的「內容物」
   而看不到模組本身 —— 症狀是 dump 出來只有
       layers.0.router.proj   QuantizedLinear
       layers.0.router.scale  array
   卻永遠沒有 layers.0.router 這一行，於是「找到 0 個 router」。
   正解：用 mlx 自己的 named_modules() / apply_to_modules()。

【坑 2】不要用 module.__call__ = wrapped 攔截。
   Python 的 dunder method 在隱式呼叫時查的是 type 不是 instance：
       mod(x)  ->  type(mod).__call__(mod, x)
   指派到 instance 上完全不會生效，而且不報錯，你只會得到一份全是 0 的統計。
   正解：patch class 的 __call__，用 id(module) 對回是哪一層，離開時還原。

【坑 3】Router.__call__ 回傳的是 top-k 索引，不是 logits。
   mlx-lm 的 models/gemma4_text.py：
       return top_k_indices, top_k_weights
   直接數 top_k_indices 就好；再做一次 argpartition 會得到完全錯的分布。
=====================================================================
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

ROUTER_CLASS = "Router"
ROUTER_LEAF_NAME = "router"


def iter_modules(model):
    """回傳 [(name, module), ...]。用 mlx 自己的 API，不要手寫走訪（坑 1）。"""
    fn = getattr(model, "named_modules", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            pass
    fn = getattr(model, "apply_to_modules", None)
    if callable(fn):
        out = []
        try:
            fn(lambda n, m: out.append((n, m)))
            if out:
                return out
        except Exception:
            pass
    from mlx.nn import Module
    out, seen = [], set()

    def walk(m, prefix=""):
        if id(m) in seen:
            return
        seen.add(id(m))
        out.append((prefix, m))
        items = m.items() if isinstance(m, dict) else []
        for k, v in items:
            name = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Module):          # 必須在 dict 之前判斷
                walk(v, name)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, Module):
                        walk(v2, f"{name}.{k2}")
            elif isinstance(v, (list, tuple)):
                for i, v2 in enumerate(v):
                    if isinstance(v2, Module):
                        walk(v2, f"{name}.{i}")

    walk(model)
    return out


def is_router(name, mod, name_filter=None):
    """優先看 class 名稱；名稱比對只認路徑最後一段，否則 router.proj 也會被誤判。"""
    leaf = name.rsplit(".", 1)[-1]
    if name_filter:
        return leaf == name_filter
    if type(mod).__name__ == ROUTER_CLASS:
        return True
    return leaf == ROUTER_LEAF_NAME


def _layer_sort_key(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def dump_modules(model, name_filter=None, max_lines=40):
    mods = iter_modules(model)
    api = "named_modules()" if hasattr(model, "named_modules") else "apply_to_modules()"
    print(f"\n=== 模組樹共 {len(mods)} 個節點（用 {api}）===\n")
    hits = [(n, m) for n, m in mods if is_router(n, m, name_filter)]
    shown = 0
    for n, m in mods:
        r = is_router(n, m, name_filter)
        if shown >= max_lines and not r:
            continue
        print(f"  {n or '<root>':<62} {type(m).__name__}{'  ★ ← router' if r else ''}")
        shown += 1
        if shown > max_lines + len(hits):
            break
    print(f"\n判定為 router 的模組：{len(hits)} 個")
    for n, m in hits[:6]:
        print(f"  {n}   [{type(m).__name__}]")
    if len(hits) > 6:
        print(f"  … 其餘 {len(hits)-6} 個")
    if hits:
        ok = "✅ 數量正確" if len(hits) == 30 else "⚠️ 數量不符，檢查上面的結構"
        print(f"\n→ 26B-A4B 應該是 30 個（每層一個）。{ok}")
    else:
        classes = sorted({type(m).__name__ for _, m in mods})
        print("  ⚠️ 沒找到。模組樹裡出現過的 class：")
        print("    " + ", ".join(classes))
        print("  用 --router-name <路徑最後一段> 指定，例如 --router-name router")
    return hits


class RouterTap:
    """攔截每個 router 的輸出。patch 的是 class 的 __call__，不是 instance（坑 2）。"""

    def __init__(self, model, name_filter, top_k, n_experts):
        self.top_k, self.n_experts = top_k, n_experts
        mods = [(n, m) for n, m in iter_modules(model) if is_router(n, m, name_filter)]
        mods.sort(key=lambda x: _layer_sort_key(x[0]))
        self.names = [n for n, _ in mods]
        self.index = {id(m): i for i, (_, m) in enumerate(mods)}
        self.counts = [Counter() for _ in mods]
        self.mode = {}
        self.patched = {}
        for _, m in mods:
            cls = type(m)
            if cls not in self.patched:
                self.patched[cls] = (cls.__call__, "__call__" in cls.__dict__)

    def _record(self, i, out):
        import mlx.core as mx
        try:
            arr = out[0] if isinstance(out, (tuple, list)) else out
            last = int(arr.shape[-1])
            if last == self.n_experts and last != self.top_k:
                self.mode[i] = "logits"
                flat = arr.reshape(-1, last)
                k = min(self.top_k, last)
                sel = mx.argpartition(-flat, k - 1, axis=-1)[:, :k]
            else:
                self.mode[i] = "indices"
                sel = arr.reshape(-1, last)
            mx.eval(sel)
            c = self.counts[i]
            for row in sel.tolist():
                for v in (row if isinstance(row, list) else [row]):
                    c[int(v)] += 1
        except Exception as e:
            self.mode[i] = f"error:{type(e).__name__}"

    def __enter__(self):
        tap = self
        for cls, (orig, _) in self.patched.items():
            def make(orig_fn):
                def patched_call(self, *a, **kw):
                    out = orig_fn(self, *a, **kw)
                    i = tap.index.get(id(self))
                    if i is not None:
                        tap._record(i, out)
                    return out
                return patched_call
            cls.__call__ = make(orig)
        return self

    def __exit__(self, *exc):
        for cls, (orig, had_own) in self.patched.items():
            try:
                if had_own:
                    cls.__call__ = orig
                else:
                    del cls.__call__
            except Exception:
                cls.__call__ = orig
        return False


def collect(model, tokenizer, prompts, name_filter, top_k, n_experts, thinking=True):
    import mlx.core as mx

    tap = RouterTap(model, name_filter, top_k, n_experts)
    if not tap.names:
        raise RuntimeError("找不到 router 模組。先跑 --dump-modules，再用 --router-name 指定。")
    print(f"  攔截 {len(tap.names)} 個 router（patch 了 {len(tap.patched)} 個 class："
          f"{', '.join(c.__name__ for c in tap.patched)}）")

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

    total_sel = sum(sum(c.values()) for c in tap.counts)
    expected = n_tok * top_k * len(tap.names)
    print(f"  攔截模式：{set(tap.mode.values()) or '（沒有被呼叫到！）'}")
    print(f"  收集到 {total_sel:,} 次專家選擇（理論值 {expected:,}，"
          f"比值 {total_sel/expected if expected else 0:.2f}）")
    if total_sel == 0:
        raise RuntimeError(
            "一次都沒攔截到 —— 代表 patch 沒生效。\n"
            "  檢查：是不是有人把 class-level patch 改回 instance-level？\n"
            "  （mod.__call__ = f 對 mod(x) 無效，Python 的 dunder 查的是 type）")
    if expected and not 0.8 < total_sel / expected < 1.25:
        print("  ⚠️ 比值離 1.0 太遠，攔到的可能不是 top-k 索引 —— 先跑 --dump-modules 確認")

    return {
        "n_layers": len(tap.names), "layer_names": tap.names,
        "n_experts": n_experts, "top_k": top_k, "n_tokens": n_tok,
        "tap_modes": {str(k): v for k, v in tap.mode.items()},
        "counts": [dict(c) for c in tap.counts],
    }


def balance_noise_floor(n_experts, n_selections):
    """路由完全均勻時，變異係數仍會有的取樣噪音。

    每個 expert 的次數 ~ Binomial(N, 1/K)：
        mean = N/K,  sd = sqrt(N·(1/K)·(1-1/K))
        cv   = sd/mean = sqrt((1-1/K)·K/N)
    實測 cv 必須明顯大於這個數，「不均衡」才是真的結論而不是樣本太少。
    """
    K, N = n_experts, max(n_selections, 1)
    return math.sqrt((1 - 1 / K) * K / N)


def kl_noise_floor(n_experts, n_selections):
    """兩份「來自同一個分布」的樣本，plug-in KL 估計量的期望值 ≈ (K-1)/(2N)。

    KL 的樣本估計是**有偏的**：就算兩邊真的同分布，算出來也不會是 0。
    所以 E4 的「中英不同」必須和這個基準比，不能只看絕對值。
    """
    return (n_experts - 1) / (2 * max(n_selections, 1))


def summarize(res, label=""):
    n_e, k = res["n_experts"], res["top_k"]

    # ⚠️ 每個 expert 的期望佔比是 **1/n_experts**，不是 top_k/n_experts。
    #    因為 shares = count / (所有 expert 的 count 總和)，全部加起來是 1.0。
    #    早期版本誤寫成 k/n_e，導致 cv 被低估 k 倍、「閒置」門檻高到 80% 平均值，
    #    於是把明顯不均衡的路由判成「均衡 ✅」。
    ideal = 1.0 / n_e

    rows, n_sel = [], 0
    for layer, c in enumerate(res["counts"]):
        c = {int(a): b for a, b in c.items()}
        total = sum(c.values()) or 1
        n_sel = max(n_sel, total)
        shares = [c.get(e, 0) / total for e in range(n_e)]
        cv = statistics.pstdev(shares) / ideal
        dead = sum(1 for s in shares if s < ideal * 0.1)   # 用量不到平均 1/10
        rows.append({"layer": layer, "cv": cv, "dead": dead, "max_share": max(shares)})

    floor = balance_noise_floor(n_e, n_sel)
    avg_cv = statistics.mean(r["cv"] for r in rows) if rows else 0
    tot_dead = sum(r["dead"] for r in rows)
    ratio = avg_cv / floor if floor else 0

    print(f"\n=== E3 路由負載均衡 {label} ===")
    print(f"每層 {n_sel} 次選擇分到 {n_e} 個 expert → 平均每個 {n_sel/n_e:.1f} 次"
          f"（期望佔比 {ideal:.2%}）")
    print(f"{'層':>3}  {'變異係數':>8}  {'低用量 expert':>12}  {'最熱 expert':>12}")
    for r in rows:
        print(f"{r['layer']:>3}  {r['cv']:>8.2f}  {r['dead']:>12}  {r['max_share']:>11.2%}")

    print(f"\n平均變異係數 {avg_cv:.2f}")
    print(f"取樣噪音下限 {floor:.2f}（路由完全均勻時也會有這麼多變異）")
    print(f"訊號 / 噪音 = {ratio:.2f}")
    print(f"低用量 expert（<平均 1/10）{tot_dead} / {len(rows)*n_e}")
    print(f"最熱的 expert 佔 {max(r['max_share'] for r in rows):.2%}"
          f"（是平均的 {max(r['max_share'] for r in rows)/ideal:.1f} 倍）")
    if ratio < 2:
        print("→ E3 判定：均衡，或樣本數不足以分辨 ⚠️（加 prompt 再看）")
    elif avg_cv < 1.0:
        print("→ E3 判定：輕度不均衡")
    else:
        print("→ E3 判定：**明顯不均衡** ⚠️（有少數專家被大量偏好）")
    return {"avg_cv": avg_cv, "noise_floor": floor, "signal_to_noise": ratio,
            "total_dead": tot_dead, "n_selections_per_layer": n_sel,
            "ideal_share": ideal, "per_layer": rows}


def kl(a, b, n_experts):
    a = {int(k): v for k, v in a.items()}
    b = {int(k): v for k, v in b.items()}
    ta, tb = sum(a.values()) or 1, sum(b.values()) or 1
    eps = 1e-9
    return sum((a.get(e, 0) / ta + eps) *
               math.log((a.get(e, 0) / ta + eps) / (b.get(e, 0) / tb + eps))
               for e in range(n_experts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dump-modules", action="store_true")
    ap.add_argument("--router-name", default=None,
                    help="router 模組路徑的最後一段（例如 router）；不給就用 class 名稱判斷")
    ap.add_argument("--n-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--compare-lang", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--compare-with", default=None)
    ap.add_argument("--no-thinking", action="store_true")
    ap.add_argument("--restat", default=None,
                    help="用已存的 json 重算統計（不載入模型），修正舊版指標用")
    args = ap.parse_args()

    if args.restat:
        d = json.loads(Path(args.restat).read_text(encoding="utf-8"))
        print(f"重算 {args.restat} 的統計（不載入模型）")
        d["zh_stats"] = summarize(d["zh"], "（繁體中文）")
        if "en" in d:
            summarize(d["en"], "（英文）")
            kls = d.get("lang_kl") or []
            if kls:
                avg = sum(kls) / len(kls)
                n_sel = d["zh_stats"]["n_selections_per_layer"]
                floor = kl_noise_floor(d["zh"]["n_experts"], n_sel)
                snr = avg / floor if floor else 0
                print(f"\n=== E4 重算 ===")
                print(f"平均 KL = {avg:.4f}；噪音下限 {floor:.4f}；訊號/噪音 = {snr:.1f} 倍")
                print("→", "明顯不同 ✅" if snr > 5 else
                      ("邊際 ⚠️" if snr > 2 else "不顯著 ❌"))
                d.update(lang_kl_avg=avg, lang_kl_noise_floor=floor, lang_kl_snr=snr)
        Path(args.restat).write_text(
            json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✅ 已更新 {args.restat}")
        return

    from mlx_lm import load
    print(f"載入 {args.model}" + (f"（+ adapter {args.adapter}）" if args.adapter else "") + " …")
    model, tokenizer = load(args.model, **({"adapter_path": args.adapter} if args.adapter else {}))

    if args.dump_modules:
        dump_modules(model, args.router_name)
        return

    zh = [p[0] for p in PROMPT_PAIRS]
    en = [p[1] for p in PROMPT_PAIRS]

    print("\n跑繁中 prompt…")
    res_zh = collect(model, tokenizer, zh, args.router_name, args.top_k,
                     args.n_experts, not args.no_thinking)
    stats = summarize(res_zh, "（繁體中文）")
    payload = {"model": args.model, "adapter": args.adapter, "zh": res_zh, "zh_stats": stats}

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
        n_sel = stats["n_selections_per_layer"]
        floor = kl_noise_floor(args.n_experts, n_sel)
        snr = avg / floor if floor else 0
        print(f"\n平均 KL = {avg:.4f}"
              f"（繁中 {res_zh['n_tokens']} tokens / 英文 {res_en['n_tokens']} tokens）")
        print(f"取樣噪音下限 ≈ (K-1)/(2N) = {floor:.4f}"
              f"　← 兩份同分布的樣本也會有這麼大的 KL")
        print(f"訊號 / 噪音 = {snr:.1f} 倍")
        if snr > 5:
            print("→ E4 判定：中英路由明顯不同 ✅（支持「語言是可辨識的路由特徵」）")
        elif snr > 2:
            print("→ E4 判定：有差異但邊際 ⚠️（建議加 prompt 再確認）")
        else:
            print("→ E4 判定：和取樣噪音同量級，不能下結論 ❌")
        payload.update(en=res_en, lang_kl=kls, lang_kl_avg=avg,
                       lang_kl_noise_floor=floor, lang_kl_snr=snr)

    if args.compare_with:
        prev = json.loads(Path(args.compare_with).read_text(encoding="utf-8"))
        pc = prev["zh"]["counts"]
        kls = [kl(pc[i], res_zh["counts"][i], args.n_experts)
               for i in range(min(len(pc), res_zh["n_layers"]))]
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
