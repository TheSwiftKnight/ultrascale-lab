#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_adapters.py — 解析 adapters.safetensors 的 header，對帳「實際訓練到什麼」。

Week 2 的教訓：k/v 一層都沒訓到、scale 語意差 10 倍，都是解析這個檔才發現的。
所以 Week 4 規定：**每次訓練完，先跑這支，再談結果。**

用法：
    python scripts/verify_adapters.py out/week4-26b-planA --expect configs/week4/week4_meta.json --plan planA
    python scripts/verify_adapters.py out/week4-26b-planB --expect configs/week4/week4_meta.json --plan planB
    python scripts/verify_adapters.py out/lora-e4b                       # 純盤點，不對帳
"""

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--file", default="adapters.safetensors")
    ap.add_argument("--expect", default=None, help="make_week4_configs.py 產的 week4_meta.json")
    ap.add_argument("--plan", choices=["planA", "planB"], default=None)
    a = ap.parse_args()

    p = Path(a.adapter_dir) / a.file
    if not p.exists():
        sys.exit(f"❌ 找不到 {p}")
    hdr = read_header(p)
    hdr.pop("__metadata__", None)

    by_module, by_layer, total = defaultdict(int), defaultdict(set), 0
    for name, info in hdr.items():
        shape = info.get("shape", [])
        n = 1
        for s in shape:
            n *= s
        total += n
        m = re.search(r"layers\.(\d+)\.(.+?)\.(lora_a|lora_b)$", name)
        if m:
            by_module[m.group(2)] += n
            by_layer[m.group(2)].add(int(m.group(1)))
        else:
            by_module[name] += n

    print(f"=== {p} ===")
    print(f"tensor 數：{len(hdr)}；實際可訓練參數：{total:,}（{total/1e6:.2f}M）\n")
    for mod in sorted(by_module):
        layers = sorted(by_layer.get(mod, []))
        span = (f"{len(layers)} 層（{layers[0]}–{layers[-1]}"
                f"{'，缺 ' + str(sorted(set(range(layers[0], layers[-1]+1)) - set(layers))) if layers and len(layers) != layers[-1]-layers[0]+1 else ''}）"
                if layers else "")
        print(f"  {mod:<28} {by_module[mod]/1e6:8.3f}M   {span}")

    if a.expect and a.plan:
        meta = json.loads(Path(a.expect).read_text())
        pred = meta["predicted_trainable"][a.plan]
        diff = (total - pred) / pred * 100 if pred else float("inf")
        print(f"\n對帳：預測 {pred/1e6:.2f}M vs 實際 {total/1e6:.2f}M（差 {diff:+.1f}%）")
        if abs(diff) > 1.0:
            print("❌ 差超過 1% —— 有 key 被靜默略過或多掛了。逐列比對上表和 week4_meta.json 的 inventory！")
            sys.exit(1)
        expect_keys = (["self_attn.q_proj", "self_attn.k_proj",
                        "self_attn.v_proj", "self_attn.o_proj"]
                       if a.plan == "planA" else [meta["router_key"]])
        missing = [k for k in expect_keys if k not in by_module]
        if missing:
            print(f"❌ 這些 key 完全沒有 adapter：{missing}")
            sys.exit(1)
        for k in expect_keys:
            got, exp = by_layer.get(k, set()), set(meta["inventory"][k])
            if got != exp:
                print(f"⚠️ {k}：實際層 {sorted(got)} ≠ 盤點層 {sorted(exp)}")
        print("✅ 對帳通過：訓到的就是設定要訓的")


if __name__ == "__main__":
    main()
