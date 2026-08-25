#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
router_bias_probe.py — MoE 方案 E：推論期路由偏置（零訓練成本的因果探針）。

在 gemma-4-26B-A4B 的每一層 router logits 上，對「中文專家」加常數偏置 b，
量測（a）中文專家的路由份額是否隨 b 上升（劑量反應）、
（b）繁中／英文文本的 NLL 是否往相反方向動（因果訊號）。

如果 b>0 讓繁中 NLL ↓（更好）而英文 NLL ↑（更差）、b<0 方向相反，
那 §3 的「語言分工」就從相關性升級成因果證據 —— 這是 moe_routing_分析.md §5.5。

用法：
    .venv/bin/python scripts/router_bias_probe.py                       # NLL 掃描（~30 分鐘）
    .venv/bin/python scripts/router_bias_probe.py --gen --gen-n 25      # 加跑生成式快速評測（久）

設計上刻意沿用 inspect_router_mlx.py 的三個教訓：
  【坑 1】用 named_modules() 找模組，不自己走 dict。
  【坑 2】patch class 的 __call__ 而不是 instance（instance 指派對 mod(x) 無效）。
  【坑 3】Router.__call__ 回傳 top-k 索引不是 logits —— 所以偏置不能加在 Router 輸出上，
          要加在 router 內部那顆 out_dims=128 的 (Quantized)Linear 的輸出（= logits）上，
          top-k 與權重歸一化維持原邏輯，實驗才乾淨。

內建對帳（不通過就中止，不會安靜給錯數字）：
  A. 每層恰好找到一顆 out_dims=128 的 router linear，共 30 顆。
  B. b=0 時的路由 counts 必須和「完全不 patch」逐 expert 相等。
  C. b=+8 煙霧測試：選中專家的份額必須大幅上升（劑量反應存在才往下跑）。
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from inspect_router_mlx import iter_modules, is_router, RouterTap, _layer_sort_key  # noqa: E402

DEFAULT_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"
N_EXPERTS, TOP_K, N_LAYERS = 128, 8, 30

# ---- 量 NLL 用的繁中/英文「翻譯配對」段落（10 對，逐句對譯）--------------------
# 這也是 §5.1 缺的「翻譯配對」對照的一個小規模版本：同義不同語言，主題完全對齊。
PARA_PAIRS = [
    ("颱風接近台灣時，氣象署會發布海上與陸上警報，各縣市再依風雨預測決定是否停班停課。",
     "When a typhoon approaches Taiwan, the weather administration issues sea and land warnings, "
     "and each county decides on work and school closures based on wind and rain forecasts."),
    ("珍珠奶茶起源於台灣，把粉圓加進奶茶裡，後來成為風靡全球的飲料。",
     "Bubble tea originated in Taiwan, adding tapioca pearls to milk tea, "
     "and later became a drink popular around the world."),
    ("高速鐵路把台北到高雄的行車時間縮短到大約九十分鐘，改變了西部走廊的通勤型態。",
     "The high-speed rail cut travel time from Taipei to Kaohsiung to about ninety minutes, "
     "changing commuting patterns along the western corridor."),
    ("半導體產業是台灣經濟的支柱，晶圓代工的產能佔全球相當高的比例。",
     "The semiconductor industry is a pillar of Taiwan's economy, "
     "with wafer foundries accounting for a very high share of global capacity."),
    ("夜市文化反映了庶民的飲食習慣，從蚵仔煎到鹽酥雞都能在攤位上找到。",
     "Night market culture reflects everyday eating habits; "
     "from oyster omelets to popcorn chicken, everything can be found at the stalls."),
    ("健保制度讓民眾以較低的費用就醫，但也面臨財務永續的挑戰。",
     "The national health insurance system lets people see doctors at low cost, "
     "but it also faces challenges of financial sustainability."),
    ("梅雨季節的鋒面帶來連日降雨，水庫的蓄水量因此得到補充。",
     "Fronts during the plum rain season bring days of continuous rainfall, "
     "which replenishes the water stored in reservoirs."),
    ("捷運系統的路網逐年擴張，讓大台北地區的大眾運輸使用率持續上升。",
     "The metro network expands year by year, steadily raising public transit usage "
     "in the greater Taipei area."),
    ("原住民族的傳統祭儀與語言保存，是文化政策裡重要的一環。",
     "Preserving the traditional ceremonies and languages of indigenous peoples "
     "is an important part of cultural policy."),
    ("面對地震頻繁的環境，新建築必須符合更嚴格的耐震規範。",
     "Facing an environment with frequent earthquakes, "
     "new buildings must meet stricter seismic design codes."),
]

ZH_PROMPTS = [p[0] for p in PARA_PAIRS]
EN_PROMPTS = [p[1] for p in PARA_PAIRS]


# ---------------------------------------------------------------- expert 選擇
def load_selected_experts(path):
    """挑選規則（Week 3 §3.5 的更新版）：每層取「繁 vs 英」與「簡 vs 英」top-3 的交集，
    交集為空就退回繁體版 top-3 —— 交集規則順便把字符集專家排除在外。"""
    d = json.loads(Path(path).read_text())
    per_layer = d["expert_overlap_trad_simp"]["per_layer"]
    assert len(per_layer) == N_LAYERS, f"預期 {N_LAYERS} 層，拿到 {len(per_layer)}"
    sel, n_inter = {}, 0
    for row in sorted(per_layer, key=lambda r: r["layer"]):
        trad, simp = set(row["top_trad_vs_en"]), set(row["top_simp_vs_en"])
        inter = sorted(trad & simp)
        if inter:
            n_inter += 1
            sel[row["layer"]] = inter
        else:
            sel[row["layer"]] = sorted(trad)
    n_sel = sum(len(v) for v in sel.values())
    print(f"  選中專家：共 {n_sel} 個（{n_inter}/{N_LAYERS} 層用交集，其餘退回繁體 top-3）")
    return sel


# ---------------------------------------------------------------- bias patch
class RouterBias:
    """把每層 router 內部的 logits linear 攔下來，對選中專家加偏置 b。
    off() / set_bias() 可以在不重載模型的情況下切換。"""

    def __init__(self, model):
        import mlx.core as mx
        self.mx = mx
        routers = [(n, m) for n, m in iter_modules(model) if is_router(n, m, None)]
        routers.sort(key=lambda x: _layer_sort_key(x[0]))
        assert len(routers) == N_LAYERS, \
            f"找到 {len(routers)} 個 router（預期 {N_LAYERS}）。先跑 inspect_router_mlx.py --dump-modules"
        self.bias_vec = {}          # id(linear module) -> mx.array [128] or None
        self.layer_of = {}
        self.patched = {}
        import mlx.nn as nn
        for li, (rname, rmod) in enumerate(routers):
            # router 內部找 out_dims=128 的 linear（對帳 A）
            cands = []
            for sn, sm in iter_modules(rmod):
                if sn == "":
                    continue
                w = getattr(sm, "weight", None)
                if w is not None and hasattr(w, "shape") and len(w.shape) >= 2 \
                        and w.shape[0] == N_EXPERTS:
                    cands.append((sn, sm))
            if not cands and getattr(getattr(rmod, "weight", None), "shape", [None])[0] == N_EXPERTS:
                cands = [("<self>", rmod)]   # router 本身就是一顆 linear 的情況
            assert len(cands) == 1, \
                (f"層 {li} 的 router 裡找到 {len(cands)} 顆 out=128 的 linear：{[c[0] for c in cands]}\n"
                 f"  預期恰好 1 顆。先 --dump-modules 看結構再改 candidates 判斷。")
            sm = cands[0][1]
            self.bias_vec[id(sm)] = None
            self.layer_of[id(sm)] = li
            cls = type(sm)
            if cls not in self.patched:
                self.patched[cls] = (cls.__call__, "__call__" in cls.__dict__)
        print(f"  掛上 bias hook：{N_LAYERS} 顆 router linear"
              f"（patch {len(self.patched)} 個 class：{', '.join(c.__name__ for c in self.patched)}）")

    def set_bias(self, selected, b):
        """selected: {layer: [expert_id,...]}；b=0 時仍會加一個全零向量（對帳 B 用）。"""
        mx = self.mx
        for mid, li in self.layer_of.items():
            vec = [0.0] * N_EXPERTS
            for e in selected.get(li, []):
                vec[e] = float(b)
            self.bias_vec[mid] = mx.array(vec)

    def off(self):
        for mid in self.bias_vec:
            self.bias_vec[mid] = None

    def __enter__(self):
        pb = self
        for cls, (orig, _) in self.patched.items():
            def make(orig_fn):
                def patched_call(self, *a, **kw):
                    out = orig_fn(self, *a, **kw)
                    v = pb.bias_vec.get(id(self), None)
                    if v is not None:
                        out = out + v.astype(out.dtype)
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


# ---------------------------------------------------------------- 量測
def route_counts(model, tokenizer, prompts, thinking=False):
    import mlx.core as mx
    tap = RouterTap(model, None, TOP_K, N_EXPERTS)
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
    total = sum(sum(c.values()) for c in tap.counts)
    expected = n_tok * TOP_K * len(tap.names)
    assert total == expected, f"對帳失敗：攔到 {total} ≠ 預期 {expected}"
    return [Counter(c) for c in tap.counts]


def selected_share(counts, selected):
    """選中專家吃到的路由份額（逐層平均）。"""
    shares = []
    for li, c in enumerate(counts):
        tot = sum(c.values())
        if not tot:
            continue
        shares.append(sum(c.get(e, 0) for e in selected.get(li, [])) / tot)
    return sum(shares) / len(shares)


def mean_nll(model, tokenizer, texts):
    """裸文本的每 token 平均 NLL（teacher forcing、不做生成）。
    絕對值不重要，重要的是同一份文本在不同 b 之下的差。"""
    import mlx.core as mx
    tot_nll, tot_tok = 0.0, 0
    for t in texts:
        ids = tokenizer.encode(t)
        x = mx.array([ids])
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        logp = logits[0, :-1].astype(mx.float32) - mx.logsumexp(
            logits[0, :-1].astype(mx.float32), axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        nll = -mx.take_along_axis(logp, tgt[:, None], axis=-1).sum()
        mx.eval(nll)
        tot_nll += float(nll)
        tot_tok += len(ids) - 1
    return tot_nll / tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--analysis", default="reports/router_prestudy_analysis.json")
    ap.add_argument("--bias-list", default="-2,-1,0,0.5,1,2")
    ap.add_argument("--gen", action="store_true", help="加跑生成式快速評測（TMMLU+ 每科 --gen-n 題）")
    ap.add_argument("--gen-n", type=int, default=25)
    ap.add_argument("--gen-bias", default="0,1,2")
    ap.add_argument("--out", default="reports/week4/router_bias_probe.json")
    a = ap.parse_args()

    biases = [float(x) for x in a.bias_list.split(",")]
    assert 0.0 in biases, "bias 列表必須含 0（對帳 B 的基準）"

    print("=== 方案 E：推論期路由偏置探針 ===")
    print("事前預測（跑之前寫下來，跑完對答案）：")
    print("  P-E1  選中專家份額隨 b 單調上升（劑量反應）")
    print("  P-E2  b>0：繁中 NLL 持平或↓、英文 NLL ↑；b<0：繁中 NLL ↑ 且升幅 > 英文")
    print("  P-E3  |b|=2 的效果 > |b|=1 > |b|=0.5\n")

    selected = load_selected_experts(ROOT / a.analysis)

    from mlx_lm import load
    print(f"載入 {a.model} …")
    model, tokenizer = load(a.model)

    # ---- 對帳 B：不 patch vs patch(b=0) 的路由 counts 必須逐 expert 相等 ----
    print("\n[對帳 B] 無 patch 基準 …")
    base_counts = route_counts(model, tokenizer, ZH_PROMPTS[:3])
    probe = RouterBias(model)
    with probe:
        probe.set_bias(selected, 0.0)
        zero_counts = route_counts(model, tokenizer, ZH_PROMPTS[:3])
        for li, (c0, c1) in enumerate(zip(base_counts, zero_counts)):
            assert c0 == c1, f"對帳 B 失敗：層 {li} 在 b=0 時路由變了 —— patch 本身有副作用"
        print("[對帳 B] 通過：b=0 與無 patch 逐 expert 相等 ✅")

        # ---- 對帳 C：b=+8 煙霧測試（劑量反應存在才往下跑）----
        probe.set_bias(selected, 8.0)
        smoke = route_counts(model, tokenizer, ZH_PROMPTS[:3])
        s0 = selected_share(base_counts, selected)
        s8 = selected_share(smoke, selected)
        print(f"[對帳 C] 選中專家份額：b=0 → {s0:.3f}，b=+8 → {s8:.3f}")
        assert s8 > s0 + 0.10, "對帳 C 失敗：b=+8 沒有明顯抬升份額 —— bias 沒有加在 logits 上？"

        # ---- 主量測 ----
        results = []
        for b in biases:
            probe.set_bias(selected, b)
            t0 = time.time()
            counts_zh = route_counts(model, tokenizer, ZH_PROMPTS)
            counts_en = route_counts(model, tokenizer, EN_PROMPTS)
            row = {
                "bias": b,
                "share_selected_zh": selected_share(counts_zh, selected),
                "share_selected_en": selected_share(counts_en, selected),
                "nll_zh": mean_nll(model, tokenizer, ZH_PROMPTS),
                "nll_en": mean_nll(model, tokenizer, EN_PROMPTS),
                "seconds": round(time.time() - t0, 1),
            }
            results.append(row)
            print(f"  b={b:+.1f}  份額 zh {row['share_selected_zh']:.3f} / en "
                  f"{row['share_selected_en']:.3f}   NLL zh {row['nll_zh']:.4f} / "
                  f"en {row['nll_en']:.4f}   ({row['seconds']}s)")

        gen_results = None
        if a.gen:
            gen_results = run_gen_eval(model, tokenizer, probe, selected,
                                       [float(x) for x in a.gen_bias.split(",")], a.gen_n)
    probe.off()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    b0 = next(r for r in results if r["bias"] == 0.0)
    payload = {
        "model": a.model, "selected_experts": {str(k): v for k, v in selected.items()},
        "materials": {"n_pairs": len(PARA_PAIRS), "type": "translation-paired paragraphs"},
        "sweep": results,
        "delta_vs_b0": [{"bias": r["bias"],
                         "d_nll_zh": r["nll_zh"] - b0["nll_zh"],
                         "d_nll_en": r["nll_en"] - b0["nll_en"]} for r in results],
        "gen_eval": gen_results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\n→ {out}")
    print("判讀提示：ΔNLL(zh) 與 ΔNLL(en) 對 b 的方向相反 → 因果訊號成立；"
          "兩者同向 → 偏置只是普遍變好/變壞，語言分工的因果解讀不成立。")


def run_gen_eval(model, tokenizer, probe, selected, biases, n_per_subject):
    """生成式快速評測（選配）：TMMLU+ 三科各 n 題，嚴格 \\box 計分。"""
    import pandas as pd
    import random as _random
    from mlx_lm import generate
    sys.path.insert(0, str(ROOT / "scripts"))
    from week4_eval_server import (SYS_BOX_ZH, TMMLU_SUBJECTS, shuffle_options,
                                   build_prompt, extract_strict, strip_thinking)
    out = []
    for b in biases:
        probe.set_bias(selected, b)
        n_ok = n_tot = n_unp = 0
        for s in TMMLU_SUBJECTS:
            df = pd.read_parquet(ROOT / "datasets" / "ikala__tmmluplus" / f"{s}.parquet")
            rng = _random.Random(42)
            qs = [q for q in (shuffle_options(dict(r), rng) for _, r in df.iterrows()) if q]
            for q in qs[:n_per_subject]:
                text = tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYS_BOX_ZH},
                     {"role": "user", "content": build_prompt(q)}],
                    add_generation_prompt=True, tokenize=False, enable_thinking=False)
                resp = generate(model, tokenizer, prompt=text, max_tokens=256)
                pred = extract_strict(strip_thinking(resp))
                n_tot += 1
                n_ok += pred == q["answer"]
                n_unp += pred is None
        row = {"bias": b, "n": n_tot, "acc_strict": n_ok / n_tot, "unparsed": n_unp / n_tot}
        out.append(row)
        print(f"  [gen] b={b:+.1f}  n={n_tot}  嚴格 {row['acc_strict']:.3f}  無法解析 {row['unparsed']:.3f}")
    return out


if __name__ == "__main__":
    main()
