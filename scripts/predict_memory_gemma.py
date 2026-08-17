#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_memory_gemma.py — 用 Playbook ch01 / ch10 的公式，對 Gemma 4 做記憶體「事前預測」

本支同時支援兩個模型，因為 Week 2 的主軸就是「dense vs MoE 對照」：

    gemma4-e4b     Gemma 4 E4B      dense，42 層   → 本機 24GB 微調主線
    gemma4-26b-a4b Gemma 4 26B-A4B  MoE，30 層     → MoE 對照組

⚠️ **為什麼不是 12B？**
   Gemma 4 12B Unified 的 `model_type` 是 `gemma4_unified`，
   mlx-lm 0.31.3（截至目前最新）**不支援**，載入會噴
   `ValueError: Model type gemma4_unified not supported.`
   （ml-explore/mlx-lm issue #1481，自 2026-07 開著、無 PR。）
   E4B / 26B-A4B / 31B 用的都是 `gemma4` / `gemma4_text`，都能跑。

   E4B 取代 12B 其實讓對照更乾淨：
   **E4B 非嵌入參數 3.97B ≈ 26B-A4B 的 active 3.82B** ——
   兩者每 token 的計算量幾乎相同，總參數卻差 3.4 倍。
   這正是「MoE 的價值是什麼」最直接的問法。

用法：
    python scripts/predict_memory_gemma.py                        # 兩個模型都算
    python scripts/predict_memory_gemma.py --seq 1024
    python scripts/predict_memory_gemma.py --model gemma4-26b-a4b --lora-target all
    python scripts/predict_memory_gemma.py --verify-config        # 上網抓 config.json 核對常數

輸出：reports/memory_prediction_gemma.md（可直接貼進報告）

⚠️ 這是**預測**不是量測。
   本機實測值用 scripts/verify_load_mlx.py（Metal）回填；
   CUDA 實測值在租卡時用 torch profiler 回填。
"""

import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GiB = 1024 ** 3


def _layers(n_slide, n_full, pattern=None):
    """產生 layer_types 清單。pattern 給定時直接用，否則平均散布 full 層。"""
    if pattern:
        return list(pattern)
    L = n_slide + n_full
    every = L / n_full if n_full else L + 1
    return ["full_attention" if (i + 1) % round(every) == 0 else "sliding_attention"
            for i in range(L)]


# ---------------------------------------------------------------- 模型規格
# 來源：google/gemma-4-E4B-it 與 google/gemma-4-26B-A4B-it 的 config.json（text_config）
# 用 --verify-config 可以上網重抓核對。
CONFIGS = {
    "gemma4-e4b": dict(
        hf_id="google/gemma-4-E4B-it",
        mlx_id="mlx-community/gemma-4-e4b-it-4bit",
        mlx_bf16_id="mlx-community/gemma-4-e4b-it-bf16",
        label="Gemma 4 E4B（dense）",
        H=2560, L=42, V=262144,
        n_heads=8, head_dim=256, global_head_dim=512,
        n_kv=2, n_kv_global=2,
        inter=10240,
        moe=False, n_experts=0, top_k=0, moe_inter=0,
        n_full=7, n_slide=35, sliding=512,
        tie_embed=True, k_eq_v=False,          # attention_k_eq_v = False
        kv_shared_layers=18,                   # 後 18 層共用前面的 K/V（沒有自己的 k/v_proj）
        ple_h=256, ple_vocab=262144,           # Per-Layer Embeddings
        n_norms=6,
        official_total=8.0e9, official_active=4.5e9,
        official_note="官方標示 4.5B effective / 8B with embeddings",
        disk_4bit_gb=5.2, disk_bf16_gb=15.9,   # HF repo 實際檔案大小（含視覺／音訊塔）
    ),
    "gemma4-26b-a4b": dict(
        hf_id="google/gemma-4-26B-A4B-it",
        mlx_id="mlx-community/gemma-4-26B-A4B-it-4bit",
        mlx_bf16_id=None,
        label="Gemma 4 26B-A4B（MoE）",
        H=2816, L=30, V=262144,
        n_heads=16, head_dim=256, global_head_dim=512,
        n_kv=8, n_kv_global=2,
        inter=2112,                            # 共享專家（shared expert）的 intermediate
        moe=True, n_experts=128, top_k=8, moe_inter=704,
        n_full=5, n_slide=25, sliding=1024,
        tie_embed=True, k_eq_v=True,           # 只作用在 full_attention 層
        kv_shared_layers=0,
        ple_h=0, ple_vocab=0,
        n_norms=6,
        official_total=25.2e9, official_active=3.8e9,
        official_note="官方標示 25.2B total / 3.8B active",
        disk_4bit_gb=None, disk_bf16_gb=None,
    ),
}


# ---------------------------------------------------------------- 參數量
def layer_specs(c):
    """每一層的 (是否 full attention, 是否有自己的 K/V)。"""
    types = _layers(c["n_slide"], c["n_full"])
    first_shared = c["L"] - c["kv_shared_layers"]
    return [(t == "full_attention", i < first_shared) for i, t in enumerate(types)]


def attn_params(c, is_full, has_kv):
    """
    單層 attention 參數。Gemma 4 有三個和一般 Transformer 不同的地方：

    1. **full_attention 層用 global_head_dim**（512，sliding 層是 256），
       但 KV 頭數可能不同。
    2. **`attention_k_eq_v`** —— K/V 共用一組投影。看 mlx-lm 的 gemma4_text.py：
           use_k_eq_v = config.attention_k_eq_v and not is_sliding
       **只有 full_attention 層共用**，sliding 層仍是分開兩組。
       （E4B 的 attention_k_eq_v=False，所以永遠分開兩組。）
    3. **`num_kv_shared_layers`** —— 後 N 層直接沿用前面算好的 K/V，
       這些層根本沒有 k_proj / v_proj。E4B 有 18 層是這樣，26B 沒有。
    """
    H = c["H"]
    hd = c["global_head_dim"] if is_full else c["head_dim"]
    kvh = c["n_kv_global"] if (is_full and c["k_eq_v"]) else c["n_kv"]
    q = H * (c["n_heads"] * hd)
    o = (c["n_heads"] * hd) * H
    if not has_kv:
        return q + o
    n_kv_proj = 1 if (is_full and c["k_eq_v"]) else 2
    return q + n_kv_proj * H * (kvh * hd) + o


def param_breakdown(c):
    H, V, L = c["H"], c["V"], c["L"]
    specs = layer_specs(c)

    attn_tot = sum(attn_params(c, f, k) for f, k in specs)

    if c["moe"]:
        one_expert = 3 * H * c["moe_inter"]           # gate / up / down
        shared = 3 * H * c["inter"]                   # 1 個常駐共享專家
        router = H * c["n_experts"]
        ffn = c["n_experts"] * one_expert + shared + router
        ffn_active = c["top_k"] * one_expert + shared + router
    else:
        one_expert = shared = router = 0
        ffn = ffn_active = 3 * H * c["inter"]

    ffn_tot = L * ffn
    ffn_tot_active = L * ffn_active
    norms = L * c["n_norms"] * H
    embed = V * H                                     # tie_word_embeddings=True

    # Per-Layer Embeddings（E4B 這類 on-device 模型才有）
    if c["ple_h"]:
        embed_ple = c["ple_vocab"] * (L * c["ple_h"])
        ple_proj = H * (L * c["ple_h"]) + L * 2 * (H * c["ple_h"])
    else:
        embed_ple = ple_proj = 0

    total = attn_tot + ffn_tot + norms + embed + embed_ple + ple_proj
    active = attn_tot + ffn_tot_active + norms + embed + embed_ple + ple_proj
    nonembed = total - embed - embed_ple

    return dict(
        attn_tot=attn_tot, ffn=ffn, ffn_active=ffn_active,
        ffn_tot=ffn_tot, one_expert=one_expert, shared=shared, router=router,
        expert_only=L * c["n_experts"] * one_expert if c["moe"] else 0,
        embed=embed, embed_ple=embed_ple, ple_proj=ple_proj,
        total=total, active=active, nonembed=nonembed,
        per_layer=attn_tot / L + ffn + c["n_norms"] * H,
    )


def weight_memory(c, p):
    """
    四種載入方式的權重記憶體。

    MLX 的 4-bit（group_size=64）：4 bit 權重 + 每 64 個元素一組 scale 和 bias，
    兩者都是 bf16（各 2 bytes）→ 4 + (2+2)*8/64 = 4.50 bit/參數 = 0.5625 byte。
    **所有線性層一起壓**，沒有排除清單。
    （對帳：out/gemma4-e4b-tw/model.safetensors 實測 4.501 bit/參數。）
    """
    total, embed = p["total"], p["embed"] + p["embed_ple"]
    body = total - embed
    return dict(
        bf16=total * 2,
        q8=total * 8.5 / 8,
        q4_e8=body * 4.50 / 8 + embed * 8.5 / 8,
        q4=total * 4.50 / 8,
    )


def lora_params(c, p, rank, target):
    """LoRA 可訓練參數量。target=all 時把 adapter 也掛到 FFN／每個 expert 上。"""
    H = c["H"]
    n = 0
    for is_full, has_kv in layer_specs(c):
        hd = c["global_head_dim"] if is_full else c["head_dim"]
        kvh = c["n_kv_global"] if (is_full and c["k_eq_v"]) else c["n_kv"]
        q_out, kv_out = c["n_heads"] * hd, kvh * hd
        n += ((H + q_out) + (q_out + H)) * rank        # q_proj + o_proj
        if has_kv:
            n_kv_proj = 1 if (is_full and c["k_eq_v"]) else 2
            n += n_kv_proj * (H + kv_out) * rank

    if target == "all":
        if c["moe"]:
            per_exp = (2 * (H + c["moe_inter"]) + (c["moe_inter"] + H)) * rank
            n += c["L"] * c["n_experts"] * per_exp
            n += c["L"] * (2 * (H + c["inter"]) + (c["inter"] + H)) * rank   # 共享專家
        else:
            n += c["L"] * (2 * (H + c["inter"]) + (c["inter"] + H)) * rank
    return n


# ---------------------------------------------------------------- 活化
def activation_book(c, seq, bs):
    """ch01 原始公式：m_act = L·seq·bs·h·(34 + 5·n_heads·seq/h)，bf16。"""
    return c["L"] * seq * bs * c["H"] * (34 + 5 * c["n_heads"] * seq / c["H"])


def _one_layer_elems(c, seq, flash, is_full, has_kv):
    """單層、單 token 要保存的活化元素數（依實際架構修正）。"""
    H = c["H"]
    hd = c["global_head_dim"] if is_full else c["head_dim"]
    kvh = c["n_kv_global"] if (is_full and c["k_eq_v"]) else c["n_kv"]
    q_out, kv_out = c["n_heads"] * hd, kvh * hd

    e = H * 2               # attn pre/post norm
    e += q_out              # Q
    if has_kv:
        e += kv_out * (1 if (is_full and c["k_eq_v"]) else 2)
    e += q_out              # attention 輸出
    e += H                  # o_proj 輸出
    e += H * 2              # ffn pre/post norm
    if c["moe"]:
        e += c["top_k"] * (2 * c["moe_inter"] + c["moe_inter"])   # 只有 active 專家
        e += 2 * c["inter"] + c["inter"]                          # 共享專家
        e += c["n_experts"]                                       # router logits
    else:
        e += 2 * c["inter"] + c["inter"]
    e += H                  # FFN 輸出
    if c["ple_h"]:
        e += c["ple_h"] * 2                                       # PLE gate / projection
    if not flash:
        # 沒有 Flash Attention 時 S/P 矩陣要落地 —— 這就是 ch01 那個 seq² 項
        ctx = seq if is_full else min(seq, c["sliding"])
        e += 2 * c["n_heads"] * ctx
    return e


def activation_corrected(c, seq, bs, flash=True, checkpointing=False):
    specs = layer_specs(c)
    per = [_one_layer_elems(c, seq, flash, f, k) for f, k in specs]
    if checkpointing:
        # full recompute：每層只留輸入（h 個元素），加上重算單層時的瞬間峰值
        return (c["L"] * c["H"] + max(per)) * seq * bs * 2
    return sum(per) * seq * bs * 2


def logits_memory(c, seq, bs):
    """輸出 logits：seq·bs·V，bf16 一份 + fp32 一份做 loss。

    Gemma 4 的 vocab 是 262,144 —— 開了梯度檢查點之後，這一項常常比活化還大，
    是 24GB 機器上最容易 OOM 的地方。
    """
    return seq * bs * c["V"] * (2 + 4)


# ---------------------------------------------------------------- 核對
def verify_config(c):
    url = f"https://huggingface.co/{c['hf_id']}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            cfg = json.loads(r.read().decode())
    except Exception as e:
        print(f"  ⚠️ 抓不到 {url}：{e}")
        return
    t = cfg.get("text_config", cfg)
    checks = [
        ("hidden_size", c["H"]), ("num_hidden_layers", c["L"]),
        ("vocab_size", c["V"]), ("num_attention_heads", c["n_heads"]),
        ("head_dim", c["head_dim"]), ("global_head_dim", c["global_head_dim"]),
        ("num_key_value_heads", c["n_kv"]),
        ("intermediate_size", c["inter"]), ("sliding_window", c["sliding"]),
        ("attention_k_eq_v", c["k_eq_v"]),
        ("num_kv_shared_layers", c["kv_shared_layers"]),
        ("hidden_size_per_layer_input", c["ple_h"]),
    ]
    if c["moe"]:
        checks += [("num_experts", c["n_experts"]), ("top_k_experts", c["top_k"]),
                   ("moe_intermediate_size", c["moe_inter"])]
    print(f"\n  核對 {c['hf_id']}：")
    for key, mine in checks:
        theirs = t.get(key)
        ok = "✅" if theirs == mine else "❌"
        print(f"    {ok} {key}: 本檔={mine} / config={theirs}")
    lt = t.get("layer_types") or []
    if lt:
        nf, ns = lt.count("full_attention"), lt.count("sliding_attention")
        ok = "✅" if (nf, ns) == (c["n_full"], c["n_slide"]) else "❌"
        print(f"    {ok} layer_types: 本檔=({c['n_full']} full, {c['n_slide']} sliding)"
              f" / config=({nf} full, {ns} sliding)")
    mt = cfg.get("model_type")
    ok = "✅" if mt in ("gemma4", "gemma4_text") else "❌"
    print(f"    {ok} model_type={mt}"
          + ("" if ok == "✅" else "  ← mlx-lm 不支援這個 model_type！"))


# ---------------------------------------------------------------- 報告
def render_model(c, args, A):
    p = param_breakdown(c)
    w = weight_memory(c, p)
    n_lora = lora_params(c, p, args.rank, args.lora_target)
    n_lora_all = lora_params(c, p, args.rank, "all")
    lora_mem = n_lora * 16          # bf16 權重2 + 梯度2 + fp32 master4 + Adam m4 v4
    full_ft = p["total"] * 16

    act_book = activation_book(c, args.seq, args.bs)
    act_flash = activation_corrected(c, args.seq, args.bs, flash=True)
    act_noflash = activation_corrected(c, args.seq, args.bs, flash=False)
    act_ckpt = activation_corrected(c, args.seq, args.bs, flash=True, checkpointing=True)
    logits = logits_memory(c, args.seq, args.bs)
    specs = layer_specs(c)
    n_has_kv = sum(1 for _, k in specs if k)

    A(f"\n---\n\n# {c['label']}\n")
    A(f"MLX 權重：`{c['mlx_id']}`"
      + (f"　bf16：`{c['mlx_bf16_id']}`" if c.get("mlx_bf16_id") else ""))
    A(f"\n設定：seq={args.seq}, batch={args.bs}, LoRA rank={args.rank}, "
      f"target={args.lora_target}\n")

    A("\n## 一、參數量驗算\n")
    A("| 項目 | 參數量 | 備註 |")
    A("|---|---:|---|")
    A(f"| 所有層的 attention | {p['attn_tot']:,} | {c['n_slide']} sliding(hd={c['head_dim']}) "
      f"+ {c['n_full']} full(hd={c['global_head_dim']})；"
      f"{n_has_kv}/{c['L']} 層有自己的 K/V |")
    if c["moe"]:
        A(f"| 每層 MoE（{c['n_experts']} 路由專家 + 1 共享 + router） | {p['ffn']:,} | "
          f"active 只有 {p['ffn_active']:,}（{p['ffn_active']/p['ffn']*100:.1f}%） |")
    else:
        A(f"| 每層 FFN（dense, inter={c['inter']:,} = {c['inter']/c['H']:.0f}h） | {p['ffn']:,} | "
          f"全部層合計 {p['ffn_tot']:,} |")
    A(f"| embedding（tied） | {p['embed']:,} | v={c['V']:,} × h={c['H']:,} |")
    if p["embed_ple"]:
        A(f"| **Per-Layer Embeddings** | {p['embed_ple']:,} | "
          f"{c['ple_vocab']:,} × ({c['L']} 層 × {c['ple_h']})，佔全模型 "
          f"**{p['embed_ple']/p['total']*100:.0f}%** |")
        A(f"| PLE 的投影層 | {p['ple_proj']:,} | 每層一組 gate + projection |")
    A(f"| **總參數** | **{p['total']/1e9:.2f}B** | {c['official_note']}；"
      f"誤差 {abs(p['total']-c['official_total'])/c['official_total']*100:.1f}% |")
    A(f"| **active / 非嵌入參數** | **{(p['active'] if c['moe'] else p['nonembed'])/1e9:.2f}B** | "
      f"官方 {c['official_active']/1e9:.1f}B，誤差 "
      f"{abs((p['active'] if c['moe'] else p['nonembed'])-c['official_active'])/c['official_active']*100:.1f}% |")
    A("\n> 殘差來自視覺／音訊塔（E4B 原生支援影像與音訊）與少數 norm 項 —— 本表只算**語言主幹**。\n> 官方標的是含多模態的完整模型，所以本表會系統性偏低約 5–12%，方向一致即視為公式正確。\n")
    if c["moe"]:
        A(f"每 token 只走 **{c['top_k']}/{c['n_experts']} = "
          f"{c['top_k']/c['n_experts']*100:.1f}%** 的路由專家。"
          f"全部 expert 權重共 {p['expert_only']/1e9:.2f}B，佔全模型 "
          f"**{p['expert_only']/p['total']*100:.1f}%**。\n")

    A("\n## 二、Q1：載入權重要多少？\n")
    A("| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |")
    A("|---|---:|---:|---|")
    for key, name in [("bf16", "bf16 全精度"), ("q8", "8-bit（MLX q8）"),
                      ("q4_e8", "4-bit body + 8-bit embed"),
                      ("q4", "4-bit 全量（MLX 預設，group_size=64）")]:
        gb = w[key] / GiB
        verdict = "✅ 有餘裕" if gb < 10 else ("⚠️ 貼邊" if gb < 16 else "❌ 放不下")
        A(f"| {name} | {w[key]/p['total']:.3f} | **{gb:.1f} GiB** | {verdict} |")
    if c.get("disk_4bit_gb"):
        A(f"\n> 交叉驗證：HF repo 上 4-bit 版實際 {c['disk_4bit_gb']} GB "
          f"（= {c['disk_4bit_gb']/1.073741824:.1f} GiB）、bf16 版 {c['disk_bf16_gb']} GB。"
          f"比預測略高，差額是視覺／音訊塔（本表未計）。\n")

    A("\n## 三、Q2：LoRA 的優化器狀態\n")
    A("| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |")
    A("|---|---:|---:|---:|")
    A(f"| 全參數微調 | {p['total']/1e9:.2f}B | 100% | **{full_ft/GiB:.0f} GiB** |")
    A(f"| LoRA r={args.rank}（attention only） | {n_lora/1e6:.1f}M | "
      f"{n_lora/p['total']*100:.4f}% | **{n_lora*16/GiB:.3f} GiB** |")
    A(f"| LoRA r={args.rank}（含 FFN/expert） | {n_lora_all/1e6:.1f}M | "
      f"{n_lora_all/p['total']*100:.4f}% | **{n_lora_all*16/GiB:.3f} GiB** |")
    A(f"\n掛到 FFN 上會膨脹 **{n_lora_all/n_lora:.1f} 倍**。")
    if c["moe"]:
        A(f"對 MoE 而言等於要處理 {c['L']}×{c['n_experts']} = "
          f"{c['L']*c['n_experts']:,} 組矩陣，這是 **H6 / E6 的量測點**。\n")
    else:
        A(f"dense 只有 {c['L']} 組 FFN，膨脹幅度遠小於 MoE —— "
          "這正是 dense vs MoE 對照要呈現的差別之一。\n")

    A("\n## 四、Q3：活化記憶體\n")
    A("| 情境 | 活化記憶體 | 說明 |")
    A("|---|---:|---|")
    A(f"| ch01 原始公式直接代入 | {act_book/GiB:.1f} GiB | 高估，架構假設不符 |")
    A(f"| 修正後，**無** Flash Attention | {act_noflash/GiB:.2f} GiB | S/P 矩陣要落地 |")
    A(f"| 修正後，**有** Flash Attention | {act_flash/GiB:.2f} GiB | ch10：不具現化 S/P |")
    A(f"| 修正後，Flash + full checkpointing | {act_ckpt/GiB:.2f} GiB | ch01：以算換記憶體 |")
    A(f"\n修正的地方：{c['n_slide']}/{c['L']} 層的注意力視窗只有 {c['sliding']} 個 token；"
      f"KV 只有 {c['n_kv']} 頭（GQA）")
    if c["kv_shared_layers"]:
        A(f"；且後 {c['kv_shared_layers']} 層共用前面的 K/V，連投影都沒有")
    if c["moe"]:
        A(f"；MoE 每 token 只過 {c['top_k']}/{c['n_experts']} 個專家，"
          f"**活化跟著 active {p['active']/1e9:.1f}B 走**")
    A(".\n")

    A(f"\n## 五、24GB 統一記憶體的預算表（seq={args.seq}, bs={args.bs}）\n")
    A(f"> ⚠️ **logits 是最容易被忽略的一項**：vocab={c['V']:,}，"
      f"seq={args.seq} 時光 logits 就要 {logits/GiB:.2f} GiB，隨 seq 與 bs 線性成長。"
      f"這是 H4 的量測點。\n")
    rows = [
        ("4-bit + Flash，不開 checkpointing", w["q4"], act_flash),
        ("4-bit + Flash + full checkpointing", w["q4"], act_ckpt),
        ("bf16 + Flash + full checkpointing", w["bf16"], act_ckpt),
    ]
    A("| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |")
    A("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, wm, act in rows:
        overhead = 1.0 * GiB
        tot = wm + lora_mem + act + logits + overhead
        gb = tot / GiB
        if gb < 15:
            v = "✅ 安全"
        elif gb < 20:
            v = "⚠️ 需調高 wired limit"
        else:
            v = "❌ OOM"
        A(f"| {name} | {wm/GiB:.1f} | {lora_mem/GiB:.2f} | {act/GiB:.2f} | "
          f"{logits/GiB:.2f} | {overhead/GiB:.1f} | **{gb:.1f} GiB** | {v} |")

    return dict(p=p, w=w, n_lora=n_lora, n_lora_all=n_lora_all,
                act_flash=act_flash, act_ckpt=act_ckpt, logits=logits, c=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIGS) + ["both"], default="both")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-target", choices=["attn", "all"], default="attn")
    ap.add_argument("--verify-config", action="store_true",
                    help="上網抓 config.json 核對本檔寫死的常數")
    args = ap.parse_args()

    names = list(CONFIGS) if args.model == "both" else [args.model]

    if args.verify_config:
        print("=== 核對 config.json ===")
        for n in names:
            verify_config(CONFIGS[n])
        print()

    lines = []
    A = lines.append
    A("# Gemma 4 記憶體預測（Playbook ch01 + ch10 公式推算）\n")
    A("> 本檔是**事前預測**；本機實測由 `verify_load_mlx.py` 回填，")
    A("> CUDA 實測由租卡時的 torch profiler 回填。\n")
    A("> ⚠️ 主線模型是 **E4B** 不是 12B —— Gemma 4 12B Unified 的 `model_type` 是")
    A("> `gemma4_unified`，mlx-lm 0.31.3 不支援（issue #1481）。")
    A("> E4B / 26B-A4B / 31B 用的是 `gemma4`，都能跑。\n")

    res = {}
    for n in names:
        res[n] = render_model(CONFIGS[n], args, A)

    if len(names) == 2:
        a, b = res["gemma4-e4b"], res["gemma4-26b-a4b"]
        ap_, bp = a["p"], b["p"]
        A("\n---\n\n# 六、dense vs MoE 對照（Week 2 的核心論點）\n")
        A("**這組配對的關鍵**：E4B 的非嵌入參數與 26B-A4B 的 active 參數幾乎一樣，"
          "也就是**每 token 的計算量相當**。差別只在總參數與記憶體佔用。"
          "所以「MoE 到底買到了什麼」可以被乾淨地量出來。\n")
        A("| 指標 | E4B dense | 26B-A4B MoE | 差異 |")
        A("|---|---:|---:|---|")
        A(f"| 總參數 | {ap_['total']/1e9:.2f}B | {bp['total']/1e9:.2f}B | "
          f"MoE 大 {bp['total']/ap_['total']:.1f}× |")
        A(f"| 每 token 實際用到 | {ap_['nonembed']/1e9:.2f}B（非嵌入） | "
          f"{bp['active']/1e9:.2f}B（active） | **幾乎相同** |")
        A(f"| 4-bit 權重 | {a['w']['q4']/GiB:.1f} GiB | {b['w']['q4']/GiB:.1f} GiB | "
          f"dense 小 {b['w']['q4']/a['w']['q4']:.1f}× |")
        A(f"| bf16 權重 | {a['w']['bf16']/GiB:.1f} GiB | {b['w']['bf16']/GiB:.1f} GiB | "
          f"E4B **本機塞得下**，26B 不行 |")
        A(f"| 活化（Flash, 無 ckpt） | {a['act_flash']/GiB:.2f} GiB | "
          f"{b['act_flash']/GiB:.2f} GiB | "
          f"{'MoE 小' if b['act_flash'] < a['act_flash'] else 'dense 小'} |")
        A(f"| LoRA(attn) 可訓練參數 | {a['n_lora']/1e6:.1f}M | {b['n_lora']/1e6:.1f}M | — |")
        A(f"| LoRA 掛到 FFN 的膨脹 | {a['n_lora_all']/a['n_lora']:.1f}× | "
          f"{b['n_lora_all']/b['n_lora']:.1f}× | "
          f"MoE 兇 {(b['n_lora_all']/b['n_lora'])/(a['n_lora_all']/a['n_lora']):.0f} 倍 |")
        A("\n**要回答的問題**：在每 token 計算量相同的前提下，"
          f"MoE 多花 {b['w']['q4']/GiB - a['w']['q4']/GiB:.1f} GiB 的記憶體養 "
          f"{bp['total']/1e9:.0f}B 參數，換到多少準確率？"
          "用微調前後的 TMMLU+ 與實測吞吐回答。\n")

    out = ROOT / "reports" / "memory_prediction_gemma.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n\n✅ 已寫入 {out}")


if __name__ == "__main__":
    main()
