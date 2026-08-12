#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_env.py — Week 1 環境驗收（Day 5 之前該裝的東西是否都到位）

用法：python scripts/verify_env.py
不會安裝任何東西，只檢查並告訴你缺什麼、該打哪一行指令。
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "✅", "⚠️ ", "❌"
results = []


def check(name, ok, detail="", fix=""):
    results.append((OK if ok else FAIL, name, detail, fix))
    return ok


def soft(name, ok, detail="", fix=""):
    results.append((OK if ok else WARN, name, detail, fix))
    return ok


def has(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


print("=" * 68)
print(" Week 1 環境驗收")
print("=" * 68)

# ---------------------------------------------------------------- 硬體
mac = platform.system() == "Darwin"
if mac:
    try:
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 1024**3
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        ram, chip = 0, "?"
    check(
        f"統一記憶體 {ram:.0f} GB（{chip}）",
        ram >= 24,
        "gpt-oss-20b MXFP4 權重約 12.8 GB，載入 + 推論需要 24 GB 以上才舒適",
        "若 <24GB：本機只能跑資料管線，模型載入驗證改用雲端",
    )
free = shutil.disk_usage(Path.home()).free / 1024**3
check(f"可用硬碟空間 {free:.0f} GB", free >= 40,
      "權重 13 GB + 資料 1 GB + HF cache", "清空間或改 HF_HOME 到外接碟")

# ---------------------------------------------------------------- Python
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro}", v >= (3, 11),
      "Twinkle Eval 需要 ≥3.11", "uv venv --python 3.11")
in_venv = sys.prefix != sys.base_prefix
check("在虛擬環境內執行", in_venv, sys.prefix,
      "source .venv/bin/activate 之後再跑一次")

# ---------------------------------------------------------------- 套件
CORE = {
    "datasets": "資料載入",
    "transformers": "tokenizer / chat template",
    "tokenizers": "",
    "huggingface_hub": "下載",
    "pandas": "統計",
    "datasketch": "MinHash 近似去重",
}
for mod, why in CORE.items():
    ver = has(mod)
    check(f"{mod} {ver or ''}", ver is not None, why,
          f"uv pip install {mod}")

OPT = {
    "mlx_lm": ("本機推論（Apple Silicon）", "uv pip install mlx-lm"),
    "hf_transfer": ("加速下載", "uv pip install hf_transfer"),
    "twinkle_eval": ("baseline 評測", "uv pip install twinkle-eval"),
    "ipykernel": ("Jupyter notebook 逐段執行", "uv pip install jupyterlab ipykernel"),
    "matplotlib": ("畫圖（報告用）", "uv pip install matplotlib"),
    "opencc": ("簡繁轉換檢查（選配）", "uv pip install opencc-python-reimplemented"),
}
for mod, (why, fix) in OPT.items():
    ver = has(mod)
    soft(f"{mod} {ver or ''}", ver is not None, why, fix)

# ---------------------------------------------------------------- HF 認證
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    token = (Path.home() / ".cache/huggingface/token").exists() or None
soft("Hugging Face 已登入", bool(token),
     "未登入也能下載公開模型，但會限速",
     "hf auth login")

# ---------------------------------------------------------------- 快取內容
hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
hub = hf_home / "hub"
def cached(repo_kind, repo):
    d = hub / f"{repo_kind}--{repo.replace('/', '--')}"
    if not d.exists():
        return None
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1024**3
    return size

sz = cached("models", "openai/gpt-oss-20b")
soft(f"gpt-oss-20b 權重已下載（{sz:.1f} GB）" if sz else "gpt-oss-20b 權重已下載",
     bool(sz and sz > 10), "約 13.8 GB",
     "HF_HUB_ENABLE_HF_TRANSFER=1 hf download openai/gpt-oss-20b")
sz = cached("datasets", "twinkle-ai/tw-reasoning-instruct-50k")
soft("tw-reasoning-instruct-50k 已快取", bool(sz),
     "", "python scripts/prepare_data.py 會自動下載")

# ---------------------------------------------------------------- 專案產物
for p, why in [
    ("data/train", "prepare_data.py Step 5 產物"),
    ("data/val", "prepare_data.py Step 5 產物"),
    ("reports/data_stats.json", "資料管線統計"),
    ("reports/memory_prediction.md", "ch01/ch10 記憶體預測"),
    ("Eval", "Twinkle Eval 原始碼"),
]:
    soft(f"{p}", (ROOT / p).exists(), why,
         "python scripts/prepare_data.py" if "data" in p else "")

# ---------------------------------------------------------------- LM Studio
try:
    import urllib.request, json as _json
    with urllib.request.urlopen("http://127.0.0.1:1234/v1/models", timeout=2) as r:
        models = [m["id"] for m in _json.load(r).get("data", [])]
    soft(f"LM Studio API 在線（載入：{', '.join(models) or '無'}）", bool(models),
         "baseline 評測會打這個端點",
         "LM Studio → Developer 分頁 → Start Server，並載入 gpt-oss-20b")
except Exception:
    soft("LM Studio API（127.0.0.1:1234）", False,
         "Day 5 的 baseline 評測需要它",
         "LM Studio → Developer 分頁 → Start Server")

# ---------------------------------------------------------------- 輸出
print()
w = max(len(r[1]) for r in results) + 2
fails = 0
for mark, name, detail, fix in results:
    print(f"{mark} {name:<{w}} {detail}")
    if mark != OK and fix:
        print(f"     ↳ 修法：{fix}")
    if mark == FAIL:
        fails += 1
print()
print("=" * 68)
if fails:
    print(f" {fails} 項必要條件未通過 —— 依上面的「修法」補齊後再跑一次。")
else:
    print(" 必要條件全數通過。⚠️ 的項目是選配，依 Day 5 進度補即可。")
print("=" * 68)
sys.exit(1 if fails else 0)
