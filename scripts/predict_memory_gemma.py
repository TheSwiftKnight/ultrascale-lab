#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_memory_gemma.py — 用 Playbook ch01 / ch10 的公式，對 Gemma 4 做記憶體「事前預測」

取代 Week 1 的 predict_memory.py（那支寫死 GPT-OSS 20B）。
本支同時支援兩個模型，因為 Week 2 的主軸就是「dense vs MoE 對照」：

    gemma4-12b     Gemma 4 12B Unified   dense，48 層   → 本機 24GB 微調主線
    gemma4-26b-a4b Gemma 4 26B-A4B       MoE，30 層     → MoE 對照組（接續 Week 1 的 ch06 分析）

用法：
    python scripts/predict_memory_gemma.py                        # 兩個模型都算，seq2048 bs1 r16
    python scripts/predict_memory_gemma.py --seq 1024
    python scripts/predict_memory_gemma.py --model gemma4-26b-a4b --lora-target all
    python scripts/predict_memory_gemma.py --verify-config        # 上網抓 config.json 核對常數

輸出：reports/memory_prediction_gemma.md（可直接貼進報告）

⚠️ 這是**預測**不是量測。
   本機實測值用 scripts/verify_load_mlx.py（Metal）回填；
   CUDA 實測值在租卡那 2–4 小時用 torch profiler 回填。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GiB = 1024 ** 3

# ---------------------------------------------------------------- 模型規格
# 來源：google/gemma-4-12B-it 與 google/gemma-4-26B-A4B-it 的 config.json（text_config 區塊）
# 用 --verify-config 可以上網重抓核對。
CONFIGS = {
    "gemma4-12b": dict(
        hf_id="google/gemma-4-12B-it",
        mlx_id="mlx-community/gemma-4-12B-it-4bit",
        label="Gemma 4 12B Unified（dense）",
        H=3840, L=48, V=262144,
        n_heads=16, head_dim=256, global_head_dim=512,
        n_kv=8, n_kv_global=1,
        inter=15360,
        moe=False, n_experts=0, top_k=0, moe_inter=0,
        n_full=8, n_slide=40, sliding=1024,
        tie_embed=True, k_eq_v=True,
        official_total=11.95e9, official_active=11.95e9,
    ),
    "gemma4-26b-a4b": dict(
        hf_id="google/gemma-4-26B-A4B-it",
        mlx_id="mlx-community/gemma-4-26B-A4B-it-4bit",
        label="Gemma 4 26B-A4B（MoE）",
        H=2816, L=30, V=262144,
        n_heads=16, head_dim=256, global_head_dim=512,
        n_kv=8, n_kv_global=2,
        inter=2112,                      # 共享專家（shared expert）的 intermediate
        moe=True, n_experts=128, top_k=8, moe_inter=704,
        n_full=5, n_slide=25, sliding=1024,
        tie_embed=True, k_eq_v=True,
        official_total=25.2e9, official_active=3.8e9,
    ),
}


# ---------------------------------------------------------------- 參數量
def param_breakdown(c):
    """逐項算參數量。Gemma 4 有三個和一般 Transformer 不同的地方，都在這裡處理。"""
    H, V = c["H"], c["V"]

    def attn(kv_heads, hd, k_eq_v):
        """
        注意力層參數。兩個 Gemma 4 特有的點：

        1. `attention_k_eq_v=True` —— K 和 V 共用同一組投影，只算一次。
           ⚠️ 但看 mlx-lm 的實作（`models/gemma4_text.py`）：
               use_k_eq_v = config.attention_k_eq_v and not is_sliding
           **只有 full_attention 層共用，sliding 層還是 K/V 分開兩組。**
           一開始漏掉這個條件，12B 的總參數會少算 3%。
        2. full_attention 層用 global_head_dim=512（sliding 層是 256），
           但 KV 頭更少（12B 只有 1 頭，26B 只有 2 頭）。
        """
        q = H * (c["n_heads"] * hd)
        n_kv_proj = 1 if k_eq_v else 2
        kv = n_kv_proj * H * (kv_heads * hd)
        o = (c["n_heads"] * hd) * H
        return q + kv + o

    a_slide = attn(c["n_kv"], c["head_dim"], k_eq_v=False)             # sliding：K/V 分開
    a_full = attn(c["n_kv_global"], c["global_head_dim"], c["k_eq_v"])  # full：可能共用

    if c["moe"]:
        one_expert = 3 * H * c["moe_inter"]           # gate / up / down
        shared = 3 * H * c["inter"]                   # 1 個常駐共享專家
        router = H * c["n_experts"]
        ffn = c["n_experts"] * one_expert + shared + router
        ffn_active = c["top_k"] * one_expert + shared + router
    else:
        one_expert = 0
        ffn = ffn_active = 3 * H * c["inter"]
        shared = router = 0

    norms = 6 * H if c["moe"] else 4 * H              # Gemma 用 pre+post norm
    embed = V * H                                     # tie_word_embeddings=True → 不另計 lm_head

    total = c["n_slide"] * (a_slide + ffn + norms) + c["n_full"] * (a_full + ffn + norms) + embed
    active = (c["n_slide"] * (a_slide + ffn_active + norms)
              + c["n_full"] * (a_full + ffn_active + norms) + embed)

    return dict(
        a_slide=a_slide, a_full=a_full, ffn=ffn, ffn_active=ffn_active,
        one_expert=one_expert, shared=shared, router=router,
        expert_only=c["L"] * c["n_experts"] * one_expert if c["moe"] else 0,
        embed=embed, total=total, active=active,
        per_layer_slide=a_slide + ffn + norms,
    )


def weight_memory(c, p):
    """
    四種載入方式的權重記憶體。

    和 GPT-OSS 最大的差別：GPT-OSS 原生 MXFP4 只量化 expert（attn/embed/lm_head 留 bf16），
    Gemma 4 沒有官方量化權重，走的是 MLX / bitsandbytes 的**通用量化**——
    所有線性層一起量化。這反而讓 26B-A4B 壓得比 GPT-OSS 更兇。
    """
    total, embed = p["total"], p["embed"]
    body = total - embed

    bf16 = total * 2
    # MLX 4-bit，group_size=64：4 bit 權重 + 每 64 個元素一組 scale/bias(fp16) → 4.25 bpw
    q4 = total * 4.25 / 8
    # MLX 常見配方：body 4-bit、embedding 留 8-bit（embedding 量化掉點最明顯）
    q4_e8 = body * 4.25 / 8 + embed * 8.5 / 8
    q8 = total * 8.5 / 8
    return dict(bf16=bf16, q4=q4, q4_e8=q4_e8, q8=q8)


def lora_params(c, p, rank, target):
    """LoRA 可訓練參數量。target=all 時把 adapter 也掛到 FFN／每個 expert 上。"""
    H = c["H"]
    q_out_s, q_out_f = c["n_heads"] * c["head_dim"], c["n_heads"] * c["global_head_dim"]
    kv_s, kv_f = c["n_kv"] * c["head_dim"], c["n_kv_global"] * c["global_head_dim"]

    def per_attn(q_out, kv_out, n_kv_proj):
        # 每個被掛上的線性層貢獻 (in+out)*rank
        return ((H + q_out) + n_kv_proj * (H + kv_out) + (q_out + H)) * rank

    # sliding 層有 q/k/v/o 四個投影；full 層若 k_eq_v 就只有 q/k/o 三個
    n = (c["n_slide"] * per_attn(q_out_s, kv_s, 2)
         + c["n_full"] * per_attn(q_out_f, kv_f, 1 if c["k_eq_v"] else 2))

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


def _one_layer_elems(c, seq, flash, is_full):
    """單層、單 token 要保存的活化元素數（依實際架構修正）。"""
    H = c["H"]
    hd = c["global_head_dim"] if is_full else c["head_dim"]
    kvh = c["n_kv_global"] if is_full else c["n_kv"]
    q_out, kv_out = c["n_heads"] * hd, kvh * hd

    e = 0
    e += H * 2          # attn pre/post norm
    e += q_out          # Q
    # sliding 層 K/V 各一份；full 層若 k_eq_v 則共用一份
    e += kv_out * (1 if (is_full and c["k_eq_v"]) else 2)
    e += q_out          # attention 輸出
    e += H              # o_proj 輸出
    e += H * 2          # ffn pre/post norm
    if c["moe"]:
        # 只有 top_k 個路由專家 + 1 個共享專家的中間結果要保存 —— 活化跟 active 走
        e += c["top_k"] * (2 * c["moe_inter"] + c["moe_inter"])
        e += 2 * c["inter"] + c["inter"]
        e += c["n_experts"]     # router logits
    else:
        e += 2 * c["inter"] + c["inter"]
    e += H              # FFN 輸出
    if not flash:
        # 沒有 Flash Attention 時 S/P 矩陣要落地 —— 這就是 ch01 那個 seq² 項
        ctx = seq if is_full else min(seq, c["sliding"])
        e += 2 * c["n_heads"] * ctx
    return e


def activation_corrected(c, seq, bs, flash=True, checkpointing=False):
    per_slide = _one_layer_elems(c, seq, flash, False)
    per_full = _one_layer_elems(c, seq, flash, True)
    if checkpointing:
        # full recompute：每層只留輸入（h 個元素），加上重算單層時的瞬間峰值
        boundary = c["L"] * c["H"]
        return (boundary + max(per_slide, per_full)) * seq * bs * 2
    return (c["n_slide"] * per_slide + c["n_full"] * per_full) * seq * bs * 2


def logits_memory(c, seq, bs):
    """輸出 logits：seq·bs·V，bf16 一份 + fp32 一份做 loss。

    Gemma 4 的 vocab 是 262,144 —— 比 GPT-OSS 的 201,088 還大 30%。
    在 24GB 機器上這一項會變成僅次於權重的第二大戶，是本週最容易 OOM 的地方。
    """
    return seq * bs * c["V"] * (2 + 4)


# ---------------------------------------------------------------- 核對
def verify_config(c):
    """上網抓 config.json，逐項核對本檔寫死的常數。"""
    import urllib.request
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
        ("num_key_value_heads", c["n_kv"]), ("num_global_key_value_heads", c["n_kv_global"]),
        ("intermediate_size", c["inter"]), ("sliding_window", c["sliding"]),
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

    A(f"\n---\n\n# {c['label']}\n")
    A(f"設定：seq={args.seq}, batch={args.bs}, LoRA rank={args.rank}, "
      f"target={args.lora_target}\n")

    A("\n## 一、參數量驗算\n")
    A("| 項目 | 參數量 | 備註 |")
    A("|---|---:|---|")
    A(f"| sliding 層 attention × {c['n_slide']} | {p['a_slide']:,} | "
      f"head_dim={c['head_dim']}，KV {c['n_kv']} 頭，K/V **分開兩組** |")
    A(f"| full 層 attention × {c['n_full']} | {p['a_full']:,} | "
      f"head_dim={c['global_head_dim']}，KV {c['n_kv_global']} 頭，"
      f"K/V **{'共用一組' if c['k_eq_v'] else '分開兩組'}** |")
    if c["moe"]:
        A(f"| 每層 MoE（{c['n_experts']} 路由專家 + 1 共享） | {p['ffn']:,} | "
          f"佔單層 {p['ffn']/p['per_layer_slide']*100:.1f}% |")
        A(f"| 每層 MoE（只算 active，top-{c['top_k']}） | {p['ffn_active']:,} | "
          f"= 全部的 {p['ffn_active']/p['ffn']*100:.1f}% |")
    else:
        A(f"| 每層 FFN（dense, inter={c['inter']:,}） | {p['ffn']:,} | "
          f"佔單層 {p['ffn']/p['per_layer_slide']*100:.1f}% |")
    A(f"| embedding（tied，不另計 lm_head） | {p['embed']:,} | "
      f"v={c['V']:,} × h={c['H']:,}，佔全模型 {p['embed']/p['total']*100:.1f}% |")
    A(f"| **總參數** | **{p['total']/1e9:.2f}B** | 官方標示 "
      f"{c['official_total']/1e9:.2f}B，誤差 "
      f"{abs(p['total']-c['official_total'])/c['official_total']*100:.1f}% |")
    A(f"| **active 參數** | **{p['active']/1e9:.2f}B** | 官方標示 "
      f"{c['official_active']/1e9:.2f}B，誤差 "
      f"{abs(p['active']-c['official_active'])/c['official_active']*100:.1f}% |")
    A("\n> 殘差來自視覺／音訊塔（26B 的視覺編碼器約 550M）與 per-layer embedding，"
      "本表只算語言主幹。誤差 <5% 即視為公式正確。\n")
    if c["moe"]:
        A(f"\n每 token 只走 **{c['top_k']}/{c['n_experts']} = "
          f"{c['top_k']/c['n_experts']*100:.1f}%** 的路由專家。"
          f"全部 expert 權重共 {p['expert_only']/1e9:.2f}B，佔全模型 "
          f"**{p['expert_only']/p['total']*100:.1f}%**。\n")

    A("\n## 二、Q1：載入權重要多少？\n")
    A("| 載入方式 | 每參數位元組 | 權重記憶體 | 24GB 機器可行？ |")
    A("|---|---:|---:|---|")
    for key, name in [("bf16", "bf16 全精度"), ("q8", "8-bit（MLX q8）"),
                      ("q4_e8", "4-bit body + 8-bit embed"), ("q4", "4-bit 全量（MLX 預設）")]:
        gb = w[key] / GiB
        verdict = "✅ 有餘裕" if gb < 10 else ("⚠️ 貼邊" if gb < 16 else "❌ 放不下")
        A(f"| {name} | {w[key]/p['total']:.3f} | **{gb:.1f} GiB** | {verdict} |")
    A("")
    if c["moe"]:
        A(f"> **和 GPT-OSS 的關鍵差異**：GPT-OSS 原生 MXFP4 只量化 expert，"
          f"attention／router／embed／lm_head 都留 bf16，所以 20.9B 壓到 12.8 GiB 就到底了。"
          f"Gemma 4 沒有官方量化權重，走 MLX 的通用量化——**所有線性層一起壓**，"
          f"加上 {p['expert_only']/p['total']*100:.0f}% 的參數本來就在 expert 裡，"
          f"結果 25.2B 反而只要 {w['q4']/GiB:.1f} GiB，比 GPT-OSS 還小。\n")

    A("\n## 三、Q2：LoRA 的優化器狀態\n")
    A("| | 可訓練參數 | 佔總參數 | 16 bytes/參數 |")
    A("|---|---:|---:|---:|")
    A(f"| 全參數微調 | {p['total']/1e9:.2f}B | 100% | **{full_ft/GiB:.0f} GiB** |")
    A(f"| LoRA r={args.rank}（attention only） | {n_lora/1e6:.1f}M | "
      f"{n_lora/p['total']*100:.4f}% | **{n_lora*16/GiB:.3f} GiB** |")
    A(f"| LoRA r={args.rank}（含 FFN/expert） | {n_lora_all/1e6:.1f}M | "
      f"{n_lora_all/p['total']*100:.4f}% | **{n_lora_all*16/GiB:.3f} GiB** |")
    A(f"\n差距 {n_lora_all/n_lora:.0f} 倍。")
    if c["moe"]:
        A(f"對 MoE 而言掛到 expert 上等於要處理 {c['L']}×{c['n_experts']} = "
          f"{c['L']*c['n_experts']:,} 組矩陣，這是 **H6 / E6 的量測點**。\n")
    else:
        A("dense 模型的 FFN 只有 L 組矩陣，膨脹幅度遠小於 MoE —— "
          "這正是 dense vs MoE 對照要呈現的差別之一。\n")

    A("\n## 四、Q3：活化記憶體\n")
    A("| 情境 | 活化記憶體 | 說明 |")
    A("|---|---:|---|")
    A(f"| ch01 原始公式直接代入 | {act_book/GiB:.1f} GiB | 高估，架構假設不符 |")
    A(f"| 修正後，**無** Flash Attention | {act_noflash/GiB:.2f} GiB | S/P 矩陣要落地 |")
    A(f"| 修正後，**有** Flash Attention | {act_flash/GiB:.2f} GiB | ch10：不具現化 S/P |")
    A(f"| 修正後，Flash + full checkpointing | {act_ckpt/GiB:.2f} GiB | ch01：以算換記憶體 |")
    A(f"\n修正的三處：K/V 共用投影且只有 {c['n_kv']} 頭（GQA）；"
      f"{c['n_slide']}/{c['L']} 層的注意力視窗只有 {c['sliding']} 個 token；")
    if c["moe"]:
        A(f"MoE 每 token 只過 {c['top_k']}/{c['n_experts']} 個專家，"
          f"**活化跟著 active {p['active']/1e9:.1f}B 走，不跟著 {p['total']/1e9:.1f}B 走**。\n")
    else:
        A(f"dense FFN 的 intermediate={c['inter']:,}（= {c['inter']/c['H']:.0f}h）。\n")

    A(f"\n## 五、24GB 統一記憶體的預算表（seq={args.seq}, bs={args.bs}）\n")
    A(f"> ⚠️ **logits 是本週最大的陷阱**：Gemma 4 的 vocab={c['V']:,}，"
      f"比 GPT-OSS 的 201,088 大 30%。seq={args.seq} 時光 logits 就要 "
      f"{logits/GiB:.2f} GiB，且隨 seq 線性成長。這是 H4 的量測點。\n")
    rows = [
        ("4-bit + Flash，不開 checkpointing", w["q4"], act_flash),
        ("4-bit + Flash + full checkpointing", w["q4"], act_ckpt),
        ("bf16 + Flash + full checkpointing", w["bf16"], act_ckpt),
    ]
    A("| 配置 | 權重 | LoRA | 活化 | logits | 框架開銷 | 合計 | 24GB 判定 |")
    A("|---|---:|---:|---:|---:|---:|---:|---|")
    for name, wm, act in rows:
        overhead = 1.0 * GiB          # Metal buffer pool + KV cache + 碎片，Week 1 實測外推
        tot = wm + n_lora * 16 + act + logits + overhead
        gb = tot / GiB
        # 24GB Mac 實際可用的 GPU wired memory 預設約 16 GiB，調高後上限約 21 GiB
        if gb < 15:
            v = "✅ 安全"
        elif gb < 20:
            v = "⚠️ 需調高 wired limit"
        else:
            v = "❌ OOM"
        A(f"| {name} | {wm/GiB:.1f} | {n_lora*16/GiB:.2f} | {act/GiB:.2f} | "
          f"{logits/GiB:.2f} | {overhead/GiB:.1f} | **{gb:.1f} GiB** | {v} |")

    return dict(p=p, w=w, n_lora=n_lora, n_lora_all=n_lora_all,
                act_flash=act_flash, act_ckpt=act_ckpt, logits=logits)


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
    A("> 取代 Week 1 的 `memory_prediction.md`（GPT-OSS 20B 版）。")
    A("> 本檔是**事前預測**；本機實測由 `verify_load_mlx.py` 回填，")
    A("> CUDA 實測由租卡時的 torch profiler 回填。\n")

    res = {}
    for n in names:
        res[n] = render_model(CONFIGS[n], args, A)

    if len(names) == 2:
        a = res["gemma4-12b"]
        b = res["gemma4-26b-a4b"]
        A("\n---\n\n# 六、dense vs MoE 對照（Week 2 的核心論點）\n")
        A("| 指標 | 12B dense | 26B-A4B MoE | 誰贏 |")
        A("|---|---:|---:|---|")
        A(f"| 總參數 | {a['p']['total']/1e9:.2f}B | {b['p']['total']/1e9:.2f}B | MoE 大 "
          f"{b['p']['total']/a['p']['total']:.1f}× |")
        A(f"| active 參數 | {a['p']['active']/1e9:.2f}B | {b['p']['active']/1e9:.2f}B | "
          f"MoE 只有 dense 的 {b['p']['active']/a['p']['active']*100:.0f}% |")
        A(f"| 4-bit 權重 | {a['w']['q4']/GiB:.1f} GiB | {b['w']['q4']/GiB:.1f} GiB | "
          f"dense 小 {b['w']['q4']/a['w']['q4']:.1f}× |")
        A(f"| 活化（Flash, 無 ckpt） | {a['act_flash']/GiB:.2f} GiB | "
          f"{b['act_flash']/GiB:.2f} GiB | "
          f"{'MoE 小' if b['act_flash'] < a['act_flash'] else 'dense 小'} |")
        A(f"| LoRA(attn) 可訓練參數 | {a['n_lora']/1e6:.1f}M | {b['n_lora']/1e6:.1f}M | — |")
        A(f"| LoRA(全掛) 可訓練參數 | {a['n_lora_all']/1e6:.1f}M | "
          f"{b['n_lora_all']/1e6:.1f}M | MoE 膨脹 "
          f"{(b['n_lora_all']/b['n_lora'])/(a['n_lora_all']/a['n_lora']):.1f}× 更兇 |")
        A("\n**這張表就是 Week 2 要交付的核心對照**："
          "MoE 用 2.2 倍的總參數與 1.4 倍的權重記憶體，換到只有 dense 三分之一的 "
          "active 參數（＝三分之一的每 token 計算量與記憶體頻寬）。"
          "值不值得，用微調後的 TMMLU+ 準確率與實測吞吐來回答。\n")

    out = ROOT / "reports" / "memory_prediction_gemma.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n\n✅ 已寫入 {out}")


if __name__ == "__main__":
    main()
