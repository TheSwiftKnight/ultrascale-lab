#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_env_week2.py — Week 2 環境驗收：缺什麼直接告訴你怎麼補

用法：
    python scripts/verify_env_week2.py
    python scripts/verify_env_week2.py --check-models   # 順便確認模型是否已下載
"""

import argparse
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GiB = 1024 ** 3

PKGS = [
    ("mlx", "uv pip install -U mlx", "MLX 核心（Apple Silicon）"),
    ("mlx_lm", "uv pip install -U mlx-lm", "MLX 語言模型（載入/生成/LoRA/server）—— 需 ≥0.31.3 才有 gemma4"),
    ("transformers", "uv pip install -U 'transformers>=5.0'",
     "tokenizer 與 chat template（Gemma 4 需要 v5+）"),
    ("datasets", "uv pip install -U 'datasets>=2.19'", "資料集"),
    ("datasketch", "uv pip install datasketch", "MinHash 近似去重"),
    ("matplotlib", "uv pip install matplotlib", "token 長度分布圖"),
    ("yaml", "uv pip install pyyaml", "讀設定檔"),
]

OPTIONAL = [
    ("twinkle_eval", "uv pip install twinkle-eval", "TMMLU+ 評測"),
]

FILES = [
    ("scripts/predict_memory_gemma.py", "記憶體預測"),
    ("scripts/prepare_data_gemma.py", "資料管線"),
    ("scripts/verify_load_mlx.py", "載入驗證 + roofline"),
    ("scripts/inspect_router_mlx.py", "MoE 路由分析"),
    ("scripts/run_ablation.py", "消融跑批"),
    ("configs/lora_gemma4_e4b.yaml", "E4B LoRA 訓練設定"),
    ("configs/eval_gemma4_e4b_base.yaml", "微調前評測設定"),
    ("configs/eval_gemma4_e4b_tuned.yaml", "微調後評測設定"),
    ("datasets/ikala__tmmluplus", "TMMLU+ 資料（Week 1 已下載）"),
]

MODELS = [
    "mlx-community/gemma-4-e4b-it-4bit",
    "mlx-community/gemma-4-26B-A4B-it-4bit",
]
OPTIONAL_MODELS = [
    ("mlx-community/gemma-4-e4b-it-bf16", "P1/P4 的 bf16 對照，本機跑得起來"),
]
# ⚠️ 不要用 gemma-4-12B —— model_type=gemma4_unified，mlx-lm 0.31.3 不支援
UNSUPPORTED = ["gemma-4-12B", "gemma-4-12b"]


def ok(msg):
    print(f"  ✅ {msg}")


def bad(msg, fix=""):
    print(f"  ❌ {msg}")
    if fix:
        print(f"     → {fix}")


def warn(msg, fix=""):
    print(f"  ⚠️  {msg}")
    if fix:
        print(f"     → {fix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-models", action="store_true")
    args = ap.parse_args()
    fails = 0

    print("\n[1] Python 與平台")
    print(f"  Python {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        bad("Python 太舊", "uv venv --python 3.11 && source .venv/bin/activate"); fails += 1
    else:
        ok("Python 版本 OK")
    try:
        import platform
        m = platform.machine()
        if m == "arm64":
            ok(f"Apple Silicon（{m}）")
        else:
            warn(f"不是 arm64（{m}）—— MLX 只在 Apple Silicon 上跑")
    except Exception:
        pass

    print("\n[2] 必要套件")
    for mod, fix, desc in PKGS:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            ok(f"{mod:<14} {v:<12} {desc}")
        except ImportError:
            bad(f"{mod:<14} 未安裝     {desc}", fix); fails += 1

    print("\n[3] 選用套件")
    for mod, fix, desc in OPTIONAL:
        try:
            m = importlib.import_module(mod)
            ok(f"{mod:<14} {getattr(m,'__version__','?'):<12} {desc}")
        except ImportError:
            warn(f"{mod:<14} 未安裝     {desc}", fix)

    print("\n[4] MLX 是否真的能用 GPU")
    try:
        import mlx.core as mx
        a = mx.ones((512, 512))
        mx.eval(a @ a)
        ok("Metal 矩陣運算通過")
        for name in ("get_active_memory", "get_peak_memory"):
            holder = mx if hasattr(mx, name) else getattr(mx, "metal", None)
            if holder and hasattr(holder, name):
                ok(f"記憶體 API {name}() 可用")
            else:
                warn(f"找不到 {name}()", "uv pip install -U mlx")
    except Exception as e:
        bad(f"MLX 執行失敗：{type(e).__name__}: {e}", "uv pip install -U mlx"); fails += 1

    print("\n[5] GPU wired memory 上限（26B MoE 需要）")
    try:
        out = subprocess.run(["sysctl", "-n", "iogpu.wired_limit_mb"],
                             capture_output=True, text=True, timeout=5)
        val = (out.stdout or "").strip()
        if val and val != "0":
            ok(f"iogpu.wired_limit_mb = {val} MB（已手動設定）")
        else:
            warn("iogpu.wired_limit_mb = 0（系統預設，24GB 機器約 16 GiB 可用）",
                 "跑 26B-A4B 前執行：sudo sysctl iogpu.wired_limit_mb=20480")
    except Exception:
        warn("讀不到 iogpu.wired_limit_mb（非 macOS？）")

    print("\n[6] 專案檔案")
    for rel, desc in FILES:
        p = ROOT / rel
        (ok if p.exists() else lambda m: (bad(m), None))(f"{rel:<42} {desc}")
        if not p.exists():
            fails += 1

    print("\n[7] 磁碟空間")
    try:
        total, used, free = shutil.disk_usage(ROOT)
        msg = f"剩餘 {free/GiB:.0f} GiB"
        (ok if free / GiB > 40 else warn)(
            msg + "（E4B 4-bit 約 5GB + bf16 約 16GB + 26B 約 16GB + adapter 約 3GB）")
    except Exception:
        pass

    if args.check_models:
        print("\n[8] 模型快取")
        try:
            from huggingface_hub import scan_cache_dir
            cache = scan_cache_dir()
            have = {r.repo_id: r.size_on_disk for r in cache.repos}
            for m in MODELS:
                if m in have:
                    ok(f"{m}  ({have[m]/GiB:.1f} GiB)")
                else:
                    warn(f"{m} 尚未下載", f"hf download {m}")
            for m, why in OPTIONAL_MODELS:
                (ok if m in have else warn)(
                    f"{m}  ({have[m]/GiB:.1f} GiB)" if m in have
                    else f"{m} 尚未下載 —— {why}")
            bad_dl = [r for r in cache.repos
                      if any(u.lower() in r.repo_id.lower() for u in UNSUPPORTED)]
            if bad_dl:
                print()
                warn("快取裡有 gemma-4-12B —— mlx-lm 0.31.3 載不動"
                     "（model_type=gemma4_unified, issue #1481）",
                     "可以刪掉：" + "、".join(f"hf cache delete {r.repo_id}" for r in bad_dl))
            stale = [r for r in cache.repos if "gpt-oss" in r.repo_id.lower()]
            if stale:
                print()
                warn(f"快取裡還有 {len(stale)} 個 gpt-oss repo，共 "
                     f"{sum(r.size_on_disk for r in stale)/GiB:.1f} GiB")
                for r in stale:
                    print(f"       hf cache delete {r.repo_id}")
        except Exception as e:
            warn(f"讀不到 HF 快取：{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if fails:
        print(f"❌ {fails} 項未通過，照上面的 → 指令補齊後再跑一次。")
        sys.exit(1)
    print("✅ 全部通過，可以開始 Week 2。")


if __name__ == "__main__":
    main()
