#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roofline_read.py — E2：算出「解碼時每個 token 實際要讀多少權重」

背景：verify_load_mlx.py 的 roofline 用的分母是「整包權重」，對兩個模型都不對：

  E4B  ：7.46B 參數裡有 2.82B 是 Per-Layer Embeddings（佔 37.8%）。
         PLE 是查表 —— 每個 token 每層只讀其中一列（42 × 256 = 10,752 個值），
         不是把整張表讀一遍。用整包當分母會算出「達成率 106%」這種
         物理上不可能的結果。
  26B  ：MoE 每個 token 只走 8/128 個路由專家，其餘 120 個不讀。
         verify_load_mlx.py 的 preset 寫死 active_frac=0.20，
         但實際是 active/total = 3.822B / 25.233B = 0.1515。

這支程式把「每 token 要讀的參數」從架構重新推一次，並提供兩種資料來源：

  --source report （預設，不用載模型）
      用 reports/load_verification_gemma.json 裡已經量到的 weight_bytes，
      乘上「要讀的參數佔全部參數的比例」。
      前提：所有張量的 bit/param 一致（MLX 的通用量化確實是這樣）。

  --source model  （要載模型，最精確）
      實際載入模型，逐張量看 .nbytes，用名稱樣式把「不是每 token 都讀」的
      張量挑出來扣掉。不依賴「bit/param 一致」這個假設。
      第一次跑請先用 --dump-tensors 確認名稱樣式對得上。

用法：
    # 不載模型，最快
    python scripts/roofline_read.py

    # 精確版（要有模型快取）
    python scripts/roofline_read.py --source model --model gemma4-e4b

    # 先看張量名稱長什麼樣（--source model 第一次務必先跑這個）
    python scripts/roofline_read.py --source model --model gemma4-e4b --dump-tensors

    # 換機器要改頻寬：M4=120 / M4 Pro=273 / M4 Max=546 GB/s
    python scripts/roofline_read.py --bandwidth 546
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- 沿用 predict_memory_gemma.py 的架構常數，不要在這裡重寫一份 ----
_spec = importlib.util.spec_from_file_location("pm", ROOT / "scripts" / "predict_memory_gemma.py")
_pm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pm)
CONFIGS = _pm.CONFIGS
param_breakdown = _pm.param_breakdown

MODEL_IDS = {
    "gemma4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
    "gemma4-26b-a4b": "mlx-community/gemma-4-26B-A4B-it-4bit",
}

# --source model 時，用來認出「不是每 token 都完整讀取」的張量。
# 【坑】這兩個樣式沒配到任何東西時程式會直接停下來，不會安靜地算出
#       和 --source report 一樣的答案 —— 那樣你會以為驗證過了其實沒有。
DEFAULT_EXCLUDE = {
    # PLE 查表：每 token 每層只讀一列
    "ple": r"per_layer|ple_embed|per_layer_model_embed",
    # MoE 路由專家：每 token 只走 top_k/n_experts
    "routed_experts": r"switch_glu|\.experts\.|mlp\.moe|gate_up_proj|down_proj.*expert",
}


# ---------------------------------------------------------------- 分母推導

def read_fraction(key):
    """
    從架構算出「每 token 要讀的參數 / 全部參數」。

    每 token 都要完整讀的：
      - 所有 attention 投影、所有 norm
      - embedding 矩陣（Gemma tie_word_embeddings=True，lm_head 就是它的轉置，
        算 logits 時整張都要讀）
      - dense 的 FFN；MoE 的 shared expert + router
      - MoE 被選中的 top_k 個路由專家
    不用完整讀的：
      - PLE 表（每層只讀一列，相對於 2.82B 可忽略）
      - MoE 沒被選中的 (1 − top_k/n_experts) 個路由專家
    """
    c = CONFIGS[key]
    p = param_breakdown(c)
    total = p["total"]

    skipped = {}
    if p["embed_ple"] > 0:
        # 每 token 實際讀的 PLE 是 L × ple_h 個值，相對整張表可忽略，但還是算進去
        per_token_ple = c["L"] * c["ple_h"]
        skipped["PLE 表（每 token 只讀 %s 個值）" % f"{per_token_ple:,}"] = p["embed_ple"] - per_token_ple
    if c.get("moe"):
        inactive = 1.0 - c["top_k"] / c["n_experts"]
        skipped["未選中的路由專家（%d/%d）" % (c["n_experts"] - c["top_k"], c["n_experts"])] = \
            p["expert_only"] * inactive

    read = total - sum(skipped.values())

    # ---- 對帳 1：MoE 算出來的 read 必須等於 predict_memory 的 active ----
    if c.get("moe"):
        drift = abs(read - p["active"]) / p["active"]
        assert drift < 0.01, (
            f"{key}：推出的每 token 讀取參數 {read/1e9:.3f}B 與 "
            f"predict_memory_gemma.py 的 active {p['active']/1e9:.3f}B 差 {drift:.1%}，"
            "兩邊的假設不一致，先修好再往下走")

    return {
        "total_params": total,
        "read_params": read,
        "fraction": read / total,
        "skipped": skipped,
        "breakdown": p,
    }


# ---------------------------------------------------------------- 兩種來源

def bytes_from_report(key, report_path):
    """用 reports/load_verification_gemma.json 已量到的 weight_bytes 換算。"""
    data = json.loads((ROOT / report_path).read_text())
    mid = MODEL_IDS[key]
    hit = next((r for r in data if r.get("model") == mid), None)
    if hit is None:
        raise SystemExit(
            f"{report_path} 裡找不到 {mid}。\n"
            f"   先跑：python scripts/verify_load_mlx.py --both")
    rf = read_fraction(key)
    return {
        "source": "report",
        "total_bytes": hit["weight_bytes"],
        "read_bytes": hit["weight_bytes"] * rf["fraction"],
        "measured_tok_per_s": hit.get("tok_per_s"),
        **rf,
    }


def bytes_from_model(key, exclude, dump_only=False):
    """實際載入模型，逐張量分類。不依賴『bit/param 一致』這個假設。"""
    try:
        import mlx.core as mx
        from mlx_lm import load
    except ImportError:
        raise SystemExit("--source model 需要 mlx / mlx-lm，請在 Mac 上跑，或改用 --source report")

    mid = MODEL_IDS[key]
    print(f"載入 {mid} …（第一次會下載）")
    model, _ = load(mid)
    mx.eval(model.parameters())

    def walk(tree, prefix=""):
        if isinstance(tree, dict):
            for k, v in tree.items():
                yield from walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(tree, (list, tuple)):
            for i, v in enumerate(tree):
                yield from walk(v, f"{prefix}.{i}")
        elif isinstance(tree, mx.array):
            yield prefix, tree

    tensors = [(n, a.nbytes) for n, a in walk(model.parameters())]
    total_bytes = sum(b for _, b in tensors)

    if dump_only:
        print(f"\n共 {len(tensors):,} 個張量，合計 {total_bytes/2**30:.2f} GiB")
        print("\n最大的 25 個：")
        for n, b in sorted(tensors, key=lambda x: -x[1])[:25]:
            print(f"  {b/2**20:>10.1f} MiB  {n}")
        print("\n目前的排除樣式：")
        for label, pat in exclude.items():
            hit = [(n, b) for n, b in tensors if re.search(pat, n)]
            print(f"  [{label}] /{pat}/ → 配到 {len(hit)} 個張量，"
                  f"{sum(b for _, b in hit)/2**30:.2f} GiB "
                  f"（{100*sum(b for _, b in hit)/total_bytes:.1f}%）")
            for n, b in sorted(hit, key=lambda x: -x[1])[:3]:
                print(f"        {b/2**20:>10.1f} MiB  {n}")
        return None

    c = CONFIGS[key]
    skipped_bytes, detail = 0.0, {}

    if param_breakdown(c)["embed_ple"] > 0:
        pat = exclude["ple"]
        hit = [(n, b) for n, b in tensors if re.search(pat, n)]
        assert hit, (
            f"PLE 樣式 /{pat}/ 一個張量都沒配到。\n"
            f"   先跑 --dump-tensors 看實際名稱，再用 --ple-pattern 指定。\n"
            f"   （沒有這道檢查，程式會算出和沒扣一樣的答案，而你不會發現。）")
        b = sum(x[1] for x in hit)
        share = b / total_bytes
        assert 0.25 < share < 0.50, (
            f"PLE 佔 {share:.1%}，預期 30–45%（架構推算 37.8%）。樣式可能配錯東西：\n   "
            + "\n   ".join(f"{n} ({v/2**20:.0f} MiB)" for n, v in sorted(hit, key=lambda x: -x[1])[:5]))
        # 每 token 實際讀的 PLE 是 L × ple_h 個值（E4B 是 10,752 個），
        # 相對於 2.82B 的整張表是 4 個數量級以下，直接當 0。
        skipped_bytes += b
        detail[f"PLE 表（{len(hit)} 個張量，每 token 只讀 {c['L']*c['ple_h']:,} 個值）"] = b

    if c.get("moe"):
        pat = exclude["routed_experts"]
        hit = [(n, b) for n, b in tensors if re.search(pat, n)]
        assert hit, (
            f"路由專家樣式 /{pat}/ 一個張量都沒配到。\n"
            f"   先跑 --dump-tensors 看實際名稱，再用 --expert-pattern 指定。")
        b = sum(x[1] for x in hit)
        share = b / total_bytes
        assert 0.80 < share < 0.97, (
            f"路由專家佔 {share:.1%}，預期 85–95%（架構推算 90.5%）。樣式可能配錯東西。")
        inactive = 1.0 - c["top_k"] / c["n_experts"]
        skipped_bytes += b * inactive
        detail[f"未選中的路由專家（{c['n_experts']-c['top_k']}/{c['n_experts']}）"] = b * inactive

    read_bytes = total_bytes - skipped_bytes
    assert 0 < read_bytes < total_bytes, "扣完之後的讀取量不合理"

    return {
        "source": "model",
        "total_bytes": total_bytes,
        "read_bytes": read_bytes,
        "fraction": read_bytes / total_bytes,
        "skipped": detail,
        "n_tensors": len(tensors),
        "measured_tok_per_s": None,
    }


# ---------------------------------------------------------------- 輸出

def render(key, r, bw_gbps, measured):
    c = CONFIGS[key]
    lines = []
    A = lines.append
    label = "E4B（dense）" if not c.get("moe") else "26B-A4B（MoE）"
    A(f"\n{'='*72}")
    A(f"{label}   來源：{'實測 bytes × 架構比例' if r['source']=='report' else '逐張量分類（精確）'}")
    A('='*72)

    tot_gb, read_gb = r["total_bytes"] / 1e9, r["read_bytes"] / 1e9
    bpp = r["total_bytes"] / r["total_params"] if r["source"] == "report" else None

    A(f"  整包權重                                {tot_gb:>8.2f} GB")
    for k, v in r["skipped"].items():
        gb = (v / 1e9) if bpp is None else (v * bpp / 1e9)
        A(f"  − {k}")
        A(f"  {'':<38}{gb:>8.2f} GB")
    A(f"  {'':<40}{'-'*8}")
    A(f"  每 token 實際讀取                       {read_gb:>8.2f} GB"
      f"   （佔整包 {100*r['fraction']:.1f}%）")
    A("")

    ceiling = bw_gbps * 1e9 / r["read_bytes"]
    A(f"  記憶體頻寬                              {bw_gbps:>8.0f} GB/s")
    A(f"  理論上限                                {ceiling:>8.1f} tok/s")
    if measured:
        pct = measured / ceiling * 100
        A(f"  實測                                    {measured:>8.1f} tok/s")
        A(f"  達成率                                  {pct:>8.1f}%")
        if pct > 100:
            A("  ❌ 超過 100%，物理上不可能 —— 分母還是算多了，回頭查排除項")
        elif pct < 30:
            A("  ⚠️ 低於 30%，可能不是頻寬受限，或分母算少了")
        else:
            A("  ✅ 落在合理區間")

    if c.get("moe"):
        dense_ceiling = bw_gbps * 1e9 / r["total_bytes"]
        A("")
        A(f"  對照：假如它是 dense（整包都要讀）")
        A(f"    每 token 讀取                         {tot_gb:>8.2f} GB")
        A(f"    理論上限                              {dense_ceiling:>8.1f} tok/s")
        if measured:
            A(f"    **實測是 dense 上限的 {measured/dense_ceiling:.2f} 倍**"
              f" → MoE 確實只讀 active 參數")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gemma4-e4b", "gemma4-26b-a4b", "both"], default="both")
    ap.add_argument("--source", choices=["report", "model"], default="report")
    ap.add_argument("--bandwidth", type=float, default=273.0,
                    help="統一記憶體頻寬 GB/s（M4=120, M4 Pro=273, M4 Max=546）")
    ap.add_argument("--report", default="reports/load_verification_gemma.json")
    ap.add_argument("--dump-tensors", action="store_true",
                    help="只印張量名稱與排除樣式的命中情況，不計算")
    ap.add_argument("--ple-pattern", default=DEFAULT_EXCLUDE["ple"])
    ap.add_argument("--expert-pattern", default=DEFAULT_EXCLUDE["routed_experts"])
    ap.add_argument("--save", default="reports/roofline_read.md")
    args = ap.parse_args()

    exclude = {"ple": args.ple_pattern, "routed_experts": args.expert_pattern}
    keys = list(CONFIGS) if args.model == "both" else [args.model]

    measured = {}
    rp = ROOT / args.report
    if rp.exists():
        for row in json.loads(rp.read_text()):
            for k, mid in MODEL_IDS.items():
                if row.get("model") == mid:
                    measured[k] = row.get("tok_per_s")

    out = ["# E2 — 每 token 讀取量與 roofline（修正版）", "",
           f"頻寬假設：{args.bandwidth:.0f} GB/s", ""]
    for key in keys:
        if args.source == "model":
            r = bytes_from_model(key, exclude, dump_only=args.dump_tensors)
            if r is None:
                continue
        else:
            r = bytes_from_report(key, args.report)
        text = render(key, r, args.bandwidth, measured.get(key))
        print(text)
        out.append("```")
        out.append(text.strip())
        out.append("```")
        out.append("")

    if not args.dump_tensors:
        p = ROOT / args.save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out))
        print(f"\n寫出：{p}")
        print("\n把上面的『每 token 實際讀取／理論上限／達成率』填回 "
              "week2_執行總結.md 第 3.2 節的第二張表。")


if __name__ == "__main__":
    main()
