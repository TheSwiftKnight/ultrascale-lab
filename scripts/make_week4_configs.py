#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_week4_configs.py — 產生 Week 4 MoE 方案 A / 方案 B 的 mlx_lm.lora 設定檔。

為什麼要用程式產生而不是手寫 yaml：
  Week 2 的兩大事故（k/v 靜默略過、scale 語意不同）都是「設定檔寫的和實際跑的不一樣」。
  這支腳本先把模型載起來，**從真實的模組樹**找出：
    1. router 內部 logits linear 的相對 key（方案 B 要掛的位置）——
       不同版本的 mlx-lm 可能叫 router.proj / mlp.router / router.gate，用猜的必錯；
    2. 每個 key 在 30 層裡實際存在幾顆（預先告訴你 full_attention 層會少幾顆 v_proj）；
    3. 據此算出**精確的**預測可訓練參數量，訓練完拿 verify_adapters.py 對帳。

用法：
    .venv/bin/python scripts/make_week4_configs.py            # 需要能載入 26B（本機）
產出：
    configs/week4/lora_26b_planA.yaml   方案 A：attention-only 全 30 層（基準線）
    configs/week4/lora_26b_planB.yaml   方案 B：router-only（機制二分：路由 vs 能力）
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from inspect_router_mlx import iter_modules, is_router  # noqa: E402

MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"
N_LAYERS, N_EXPERTS = 30, 128
RANK_A, RANK_B = 16, 32          # 方案 B 的 router 是 2816×128 小矩陣，r 拉高一點補容量
SCALE = 2.0                      # ⚠️ mlx 的 scale 是直接乘數（≠ alpha/rank）。2.0 = 業界常規。
ITERS = 200                      # Week 3 checkpoint 掃描：400 步後格式崩加速 → 停在 200
SEQ = 1024                       # 壓 262K-vocab logits 的記憶體（沿用 Week 2 的 26B 設定）

COMMON = f"""model: "{MODEL}"
train: true
data: "data/mlx"
seed: 42
num_layers: {N_LAYERS}
batch_size: 1
iters: {ITERS}
max_seq_length: {SEQ}
learning_rate: 1e-4
lr_schedule:
  name: cosine_decay
  warmup: 20
  warmup_init: 1e-6
  arguments: [1e-4, {ITERS}, 1e-6]
steps_per_report: 10
steps_per_eval: 50
val_batches: 10
save_every: 100
resume_adapter_file: null
test: false
grad_checkpoint: true
"""


def layer_relative(path):
    """language_model.model.layers.7.router.proj → (7, 'router.proj')"""
    m = re.search(r"\.layers\.(\d+)\.(.+)$", path)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def main():
    from mlx_lm import load
    print(f"載入 {MODEL} …（只為了讀模組樹，~1 分鐘）")
    model, _ = load(MODEL)

    mods = iter_modules(model)

    # ---- 1. 找 router logits linear 的相對 key ----
    router_paths = sorted(n for n, m in mods if is_router(n, m, None))
    assert len(router_paths) == N_LAYERS, f"router 數量 {len(router_paths)} ≠ {N_LAYERS}"
    router_key, router_shapes = None, []
    for n, m in mods:
        w = getattr(m, "weight", None)
        if w is None or not hasattr(w, "shape") or len(w.shape) < 2:
            continue
        li, rel = layer_relative(n)
        if li is None:
            continue
        if any(n == rp or n.startswith(rp + ".") for rp in router_paths) \
                and w.shape[0] == N_EXPERTS:
            router_shapes.append((li, rel, tuple(int(x) for x in w.shape)))
    rels = sorted({r for _, r, _ in router_shapes})
    assert len(rels) == 1, f"router linear 的相對 key 不唯一：{rels} —— 手動確認後改這支腳本"
    router_key = rels[0]
    assert len(router_shapes) == N_LAYERS
    print(f"  ✅ 方案 B 要掛的 key：'{router_key}'（30 層都在，out={N_EXPERTS}）")

    # ---- 2. 盤點 attention keys 每層實際存在幾顆 ----
    attn_keys = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
    inventory = {k: [] for k in attn_keys + [router_key]}
    dims = {k: {} for k in attn_keys + [router_key]}   # ⚠️ 逐層存！
    # Gemma 4 的 sliding / full_attention 層維度不同（k/v 尤其），
    # 只存一組維度乘層數會差 ~5%（Week 4 實跑抓到的 bug）。
    for n, m in mods:
        li, rel = layer_relative(n)
        if li is None or rel not in inventory:
            continue
        w = getattr(m, "weight", None)
        if w is None:
            continue
        inventory[rel].append(li)
        # QuantizedLinear 的 weight 第二維是壓縮過的；輸入維度用 scales 推
        out_d = int(w.shape[0])
        sc = getattr(m, "scales", None)
        in_d = int(sc.shape[1] * getattr(m, "group_size", 64)) if sc is not None \
            else int(w.shape[1])
        dims[rel][li] = (in_d, out_d)
    for k in attn_keys + [router_key]:
        n_found = len(set(inventory[k]))
        missing = sorted(set(range(N_LAYERS)) - set(inventory[k]))
        distinct = sorted(set(dims[k].values()))
        note = f"  ⚠️ 缺席層：{missing}（full_attention 共用 KV → adapter 掛不上，屬預期）" \
            if missing else ""
        print(f"  {k:<22} {n_found}/{N_LAYERS} 層有此模組  dims(in,out)={distinct}{note}")

    # ---- 3. 精確預測可訓練參數（逐層維度加總）----
    def lora_params(keys, rank):
        tot = 0
        for k in keys:
            for li in set(inventory[k]):
                in_d, out_d = dims[k][li]
                tot += rank * (in_d + out_d)
        return tot

    pred_a = lora_params(attn_keys, RANK_A)
    pred_b = lora_params([router_key], RANK_B)
    print(f"\n  預測可訓練參數：方案 A = {pred_a/1e6:.2f}M（r={RANK_A}）、"
          f"方案 B = {pred_b/1e6:.2f}M（r={RANK_B}）")
    print("  （訓練完務必跑 verify_adapters.py 對這兩個數字 —— 設定檔不是事實，產出的檔案才是）")

    # ---- 4. 寫 yaml ----
    cfg_dir = ROOT / "configs" / "week4"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    plan_a = (f"# Week 4 方案 A — attention-only LoRA 全 {N_LAYERS} 層（MoE 主線的基準線）\n"
              f"# 由 make_week4_configs.py 產生；預測可訓練參數 {pred_a/1e6:.2f}M\n"
              f"# ⚠️ v_proj 在 full_attention 層會被靜默略過（上面盤點過的缺席層），這是預期行為。\n"
              + COMMON +
              f"adapter_path: \"out/week4-26b-planA\"\n"
              "lora_parameters:\n"
              + "  keys:\n"
              + "".join(f"    - \"{k}\"\n" for k in attn_keys)
              + f"  rank: {RANK_A}\n"
              f"  scale: {SCALE}   # 直接乘數；2.0 = 業界常規（Week 2 誤用 20 的教訓）\n"
              "  dropout: 0.0\n")
    plan_b = (f"# Week 4 方案 B — router-only LoRA（機制二分：路由問題 vs 能力問題）\n"
              f"# 由 make_week4_configs.py 產生；預測可訓練參數 {pred_b/1e6:.2f}M\n"
              f"# 注意：這是 LoRA(r={RANK_B}) 而非全參數 —— router 矩陣 out=128，\n"
              f"#       r={RANK_B} 已能表達 128 維輸出空間的 1/4，夠做機制二分；報告要註明這個限制。\n"
              f"# expert / attention 全凍結：mlx_lm.lora 只會訓練 keys 指到的 adapter。\n"
              + COMMON +
              f"adapter_path: \"out/week4-26b-planB\"\n"
              "lora_parameters:\n"
              "  keys:\n"
              f"    - \"{router_key}\"\n"
              f"  rank: {RANK_B}\n"
              f"  scale: {SCALE}\n"
              "  dropout: 0.0\n")

    (cfg_dir / "lora_26b_planA.yaml").write_text(plan_a)
    (cfg_dir / "lora_26b_planB.yaml").write_text(plan_b)
    meta = {"router_key": router_key, "dims": dims,
            "inventory": {k: sorted(set(v)) for k, v in inventory.items()},
            "predicted_trainable": {"planA": pred_a, "planB": pred_b},
            "rank": {"planA": RANK_A, "planB": RANK_B}, "scale": SCALE, "iters": ITERS}
    (cfg_dir / "week4_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n→ {cfg_dir}/lora_26b_planA.yaml")
    print(f"→ {cfg_dir}/lora_26b_planB.yaml")
    print(f"→ {cfg_dir}/week4_meta.json（verify_adapters.py 會拿這份對帳）")


if __name__ == "__main__":
    main()
