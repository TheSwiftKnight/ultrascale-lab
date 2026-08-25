#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dlm_ar_bench.py — AR vs DLM 的載入/記憶體/吞吐對照（同一台 M4 Pro、同為 4-bit）。

實驗載體是 proposal 想要的「同架構乾淨對照」：
    AR :  mlx-community/gemma-4-26B-A4B-it-4bit          （Week 2 的 MoE 對照組）
    DLM:  mlx-community/diffusiongemma-26B-A4B-it-4bit   （同一個 Gemma 4 26B-A4B 骨幹）
兩者總參數 / active / 層數 / 專家數完全相同，唯一差別是生成範式（自迴歸 vs 區塊擴散）。

事前預測（跑之前先想清楚，寫進報告）——這直接接回 Week 2 的 roofline：
    AR 解碼是記憶體頻寬瓶頸（實測 53.6 tok/s，= 頻寬上限的 48%）。
    DLM 每次 forward 處理 256-token canvas，比較像 prefill（算力瓶頸）。
    而 H5 已證明 M4 Pro 在 batch=1 訓練時算力就飽和 → 官方宣稱的 4×加速
    是「GPU 算力閒置」場景的數字，**在 M4 Pro 上不一定重現**。量出來多少就是答案。

用法：
    .venv/bin/python scripts/dlm_ar_bench.py --model ar    # 先跑 AR（已知可跑，當 sanity）
    .venv/bin/python scripts/dlm_ar_bench.py --model dlm   # 再跑 DLM
    .venv/bin/python scripts/dlm_ar_bench.py --report      # 兩份都跑完後，出對照表

⚠️ DLM 需要 diffusion-aware runtime。這支先試 mlx_lm（新版已列入官方支援清單）；
   如果 load/generate 丟 model type 錯誤，腳本會印出 llama.cpp 的 fallback 指令
   （llama-diffusion-cli，Metal 可用），吞吐就從它的輸出抄。
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "week4"

MODELS = {
    "ar": "mlx-community/gemma-4-26B-A4B-it-4bit",
    "dlm": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
}

PROMPTS = [
    "請用大約兩百字介紹台灣的夜市文化。",
    "解釋什麼是動態規劃，並舉一個例子。",
    "寫一段 Python 程式碼，讀取 CSV 並計算每欄平均值。",
    "說明梅雨季節對台灣水資源的影響。",
]

FALLBACK = """
── mlx_lm 還不支援這個模型時的 fallback（llama.cpp, Metal）──────────────
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
gh pr checkout 24423          # DiffusionGemma 的 diffusion runtime PR
cmake -B build -DGGML_CUDA=OFF && cmake --build build -j --target llama-diffusion-cli
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF --include "*Q4_K_M*"
./build/bin/llama-diffusion-cli -m <gguf 路徑> -ngl 99 -cnv -n 512
#   吞吐直接抄 CLI 的輸出；⚠️ Q4_K_M 和 MLX 4-bit 位元組成不同（Week 2 的 4.50 bit 教訓），
#   跨 engine 的記憶體數字要標註量化格式，不可直接並列。
──────────────────────────────────────────────────────────────────────
"""


def bench(key, max_tokens):
    import mlx.core as mx
    from mlx_lm import load, generate
    model_id = MODELS[key]
    peak = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    active = getattr(mx, "get_active_memory", None) or mx.metal.get_active_memory
    reset = getattr(mx, "reset_peak_memory", None) or mx.metal.reset_peak_memory
    GiB = 1024 ** 3

    reset()
    print(f"載入 {model_id} …")
    t0 = time.time()
    try:
        model, tokenizer = load(model_id)
        mx.eval(model.parameters())
    except Exception as e:
        print(f"❌ 載入失敗：{type(e).__name__}: {e}")
        if key == "dlm":
            print("   多半是 mlx-lm 版本還沒支援 model_type=diffusion_gemma。先升級：")
            print("   uv pip install -U mlx mlx-lm")
            print(FALLBACK)
        raise SystemExit(1)
    load_s = time.time() - t0
    after_load = active() / GiB
    print(f"  載入 {load_s:.1f}s；權重佔用 {after_load:.2f} GiB")

    reset()
    runs = []
    for p in PROMPTS:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=False)
        t0 = time.time()
        try:
            out = generate(model, tokenizer, prompt=text, max_tokens=max_tokens)
        except Exception as e:
            print(f"❌ 生成失敗：{type(e).__name__}: {e}")
            if key == "dlm":
                print("   mlx_lm 的 generate 還沒接 diffusion sampler。")
                print(FALLBACK)
            raise SystemExit(1)
        dt = time.time() - t0
        n_out = len(tokenizer.encode(out))
        runs.append({"prompt": p[:20], "n_tokens": n_out, "seconds": round(dt, 2),
                     "tok_per_s": round(n_out / dt, 2)})
        print(f"    {n_out:>4} tokens / {dt:>6.1f}s = {n_out/dt:>6.1f} tok/s")
    peak_gib = peak() / GiB
    thr = sum(r["n_tokens"] for r in runs) / sum(r["seconds"] for r in runs)
    print(f"\n  {key.upper()}  平均吞吐 {thr:.1f} tok/s；生成峰值 {peak_gib:.2f} GiB")

    OUT.mkdir(parents=True, exist_ok=True)
    res = {"key": key, "model": model_id, "load_seconds": round(load_s, 1),
           "after_load_gib": round(after_load, 2), "peak_gib": round(peak_gib, 2),
           "tok_per_s": round(thr, 1), "max_tokens": max_tokens, "runs": runs,
           "note": "單一 session、單一請求；跨硬體不可比"}
    (OUT / f"bench_{key}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"  → {OUT}/bench_{key}.json")
    print("  ⚠️ 記憶體守則：這是本 session 首次量測才可引用（session 殘留會灌水 —— Week 3 教訓）。"
          "\n     AR 和 DLM 各自用**全新的 python process** 跑，不要在同一個行程裡連跑兩個模型。")


def bench_llama(key, args):
    """llama.cpp fallback：逐 prompt 跑 llama-diffusion-cli / llama-cli，
    解析它印的 timing 與 model buffer，峰值記憶體用 getrusage(RUSAGE_CHILDREN)。
    設計要點：stdin 接 /dev/null —— cli 的對話模式生成完第一輪回覆後讀到 EOF
    就會自己退出，不會掛在互動 prompt 上等輸入。"""
    import re
    import resource
    import subprocess
    bin_, gguf = args.llama_bin, args.gguf
    assert bin_ and gguf, "--engine llama 需要 --llama-bin 與 --gguf"
    assert Path(bin_).exists(), f"找不到 {bin_}（先照手冊附錄 F 編譯）"
    assert Path(gguf).exists(), f"找不到 {gguf}"
    quant = re.search(r"(Q\d[\w_]*|IQ\d[\w_]*|F16|BF16)", Path(gguf).name)
    engine = f"llama.cpp {quant.group(1) if quant else '?'}"
    print(f"engine: {engine}\nbinary: {bin_}\ngguf:   {gguf}\n")

    import shlex
    import time as _time
    extra = shlex.split(args.llama_extra or "")
    log_dir = OUT / "llama_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # 吞吐行的候選格式（不同 CLI / PR 的講法不一樣，全部候選；取最後一個非 prompt-eval 的）
    TPS_PAT = re.compile(r"([\d.]+)\s*(?:tokens per second|tokens?/s(?:ec)?|tok/s|t/s)",
                         re.IGNORECASE)
    runs, load_ms_list, buf_mib = [], [], None
    for i, p in enumerate(PROMPTS):
        cmd = [bin_, "-m", gguf, "-ngl", "99", "-c", "4096",
               "-n", str(args.max_tokens)] + extra + ["-p", p]
        print(f"  ▶ {p[:24]} …", flush=True)
        t0 = _time.time()
        try:
            r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            print("    ❌ 超時（30 分鐘）"); continue
        wall = _time.time() - t0
        log = (r.stdout or "") + "\n===STDERR===\n" + (r.stderr or "")
        log_path = log_dir / f"{key}_{i}.log"
        log_path.write_text(log)
        # model buffer（權重佔用）：把各 backend 的 model buffer size 加總
        bufs = [float(x) for x in re.findall(r"model buffer size\s*=\s*([\d.]+)\s*MiB", log)]
        if bufs:
            buf_mib = sum(bufs)
        m_load = re.search(r"load time\s*=\s*([\d.]+)\s*ms", log)
        if m_load:
            load_ms_list.append(float(m_load.group(1)))
        tps = None
        for line in log.splitlines():
            m = TPS_PAT.search(line)
            if m and "prompt eval" not in line:
                tps = float(m.group(1))
        entry = {"prompt": p[:20], "tok_per_s": tps, "wall_seconds": round(wall, 1),
                 "returncode": r.returncode, "log": str(log_path)}
        if tps is None:
            print(f"    ⚠️ 解析不到 tok/s（wall {wall:.1f}s，rc={r.returncode}）。"
                  f"完整 log → {log_path}")
            print("    找計時行：grep -inE 'second|/s|token|perf|timing' " + str(log_path))
        else:
            print(f"    {tps:.1f} tok/s（wall {wall:.1f}s）")
        runs.append(entry)
    parsed = [r for r in runs if r["tok_per_s"]]
    assert parsed, (
        "每一輪都解析不到 tok/s。完整 log 已存在 reports/week4/llama_logs/ ——\n"
        "  用上面的 grep 找到它的計時行長相，貼給 Claude 補 regex；\n"
        "  或自己換算後用 --manual dlm 登記（tok/s ≈ 生成 token 數 ÷ wall 秒數）。")

    peak_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 3  # macOS: bytes
    thr = sum(r["tok_per_s"] for r in parsed) / len(parsed)
    res = {"key": key, "model": Path(gguf).name, "engine": engine,
           "load_seconds": round(sum(load_ms_list) / len(load_ms_list) / 1000, 1)
           if load_ms_list else None,
           "after_load_gib": round(buf_mib / 1024, 2) if buf_mib else None,
           "peak_gib": round(peak_gib, 2),
           "tok_per_s": round(thr, 1), "max_tokens": args.max_tokens, "runs": runs,
           "note": ("峰值=子行程 RSS（含 mmap 頁面，量法與 MLX 的 metal API 不同）；"
                    "跨 engine/量化格式數字要標註，不可直接並列")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"bench_{key}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n  {key.upper()}  平均吞吐 {thr:.1f} tok/s；權重 buffer "
          f"{res['after_load_gib']} GiB；子行程峰值 {res['peak_gib']} GiB")
    print(f"  → {OUT}/bench_{key}.json")
    print("  ⚠️ -p 模式沒有套 chat template（量吞吐夠用）；品質評測走 llama-server 或 Colab。")


def manual(key, args):
    """mlx 不支援時的登記口：把 llama.cpp（或其他 engine）量到的數字寫成 bench json，
    讓 --report 照常出表。engine 一定要照實填 —— 量化格式不同的數字不能混著比。"""
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"key": key, "model": args.model_id or MODELS[key],
           "engine": args.engine_label,
           "load_seconds": args.load_seconds,
           "after_load_gib": args.load_gib, "peak_gib": args.peak_gib,
           "tok_per_s": args.tok_per_s, "max_tokens": None, "runs": [],
           "note": f"手動登記（{args.engine}）；跨 engine/量化格式的數字要標註，不可直接並列"}
    (OUT / f"bench_{key}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"→ {OUT}/bench_{key}.json  已登記：{res}")


def _row(name, av, dv, unit="", strong=False):
    def f(v):
        return f"{v}{unit}" if v is not None else "—"
    ratio = f"{dv/av:.2f}×" if (av and dv) else "—"
    if strong and ratio != "—":
        ratio = f"**{ratio}**"
    return f"| {name} | {f(av)} | {f(dv)} | {ratio} |"


def report():
    rows = {}
    for key in MODELS:
        p = OUT / f"bench_{key}.json"
        if not p.exists():
            raise SystemExit(f"❌ 缺 {p}，先跑 --model {key}（或 --manual {key}）")
        rows[key] = json.loads(p.read_text())
    a, d = rows["ar"], rows["dlm"]
    ea, ed = a.get("engine", "MLX 4-bit"), d.get("engine", "MLX 4-bit")
    md = [
        "# AR vs DLM 成本對照（M4 Pro / 單請求）",
        "",
        f"AR engine：{ea}｜DLM engine：{ed}",
        "" if ea == ed else
        "⚠️ **兩邊 engine／量化格式不同**：比值僅供方向參考。要嚴謹比較，AR 也用同一個"
        " engine 重量一次（例如兩邊都跑 llama.cpp 同一個 Q4_K_M）。"
        "（W2 教訓：MLX 4-bit = 4.50 bit/param，GGUF Q4_K_M 的位元組成不同。）",
        "",
        "| 指標 | AR gemma-4-26B-A4B | DLM diffusiongemma-26B-A4B | 比值 DLM/AR |",
        "|---|---:|---:|---:|",
        _row("載入耗時", a.get("load_seconds"), d.get("load_seconds"), " s"),
        _row("權重佔用", a.get("after_load_gib"), d.get("after_load_gib"), " GiB"),
        _row("生成峰值", a.get("peak_gib"), d.get("peak_gib"), " GiB"),
        _row("生成吞吐", a.get("tok_per_s"), d.get("tok_per_s"), " tok/s", strong=True),
        "",
        "判讀（對回 Week 2 roofline）：AR 實測 53.6 tok/s ≈ 頻寬上限的 48%；",
        "若 DLM 比值 <4×，落差來自 M4 Pro 沒有「閒置算力」可以吃（H5 已證 batch=1 就算力飽和），",
        "官方 4× 是 GPU 算力閒置場景的數字 —— 這本身就是 roofline 的一次成功預測。",
        "若比值 >1×，代表擴散把「每 token 讀一次全部權重」攤到 canvas 上，頻寬瓶頸被繞開了。",
    ]
    out = OUT / "bench_ar_vs_dlm.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n→ {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["ar", "dlm"])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--report", action="store_true")
    # llama.cpp fallback（mlx 不支援 diffusiongemma 時）：
    ap.add_argument("--engine", default="mlx", choices=["mlx", "llama"],
                    help="llama = 走 llama.cpp（需 --llama-bin 與 --gguf）")
    ap.add_argument("--llama-bin", default=None,
                    help="llama-diffusion-cli（DLM）或 llama-cli（AR）的路徑")
    ap.add_argument("--gguf", default=None)
    ap.add_argument("--llama-extra", default="",
                    help='附加參數字串，放在 -p 之前（後面的同名參數會蓋掉前面的），'
                         '例如 "-fa -c 2048 -b 256 -ub 256" 或 "-ngl 26"')
    # 手動登記（自動解析失敗時的最後手段）：
    ap.add_argument("--manual", choices=["ar", "dlm"])
    ap.add_argument("--engine-label", default="llama.cpp Q4_K_M",
                    help="--manual 登記時寫進 json 的 engine 標籤")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--tok-per-s", type=float, default=None)
    ap.add_argument("--load-gib", type=float, default=None)
    ap.add_argument("--peak-gib", type=float, default=None)
    ap.add_argument("--load-seconds", type=float, default=None)
    a = ap.parse_args()
    if a.manual:
        assert a.tok_per_s, "--manual 至少要給 --tok-per-s"
        manual(a.manual, a)
    elif a.report:
        report()
    elif a.model and a.engine == "llama":
        bench_llama(a.model, a)
    elif a.model:
        bench(a.model, a.max_tokens)
    else:
        raise SystemExit("用法：--model ar|dlm [--engine llama --llama-bin … --gguf …]、"
                         "--manual ar|dlm --tok-per-s …、或 --report")


if __name__ == "__main__":
    main()
