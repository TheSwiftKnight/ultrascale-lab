#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_eval_subset.py — 產生 Twinkle Eval 要吃的「科目子集資料夾」

⚠️ 為什麼需要這支程式：
   Twinkle Eval 的 `dataset_paths` **只吃目錄，不吃單一檔案**。
   看 twinkle_eval/core/validators.py：

       if not os.path.isdir(dataset_path):
           raise ValidationError(f"Dataset path is not a directory: {dataset_path}")

   然後 `find_all_evaluation_files()` 會用 os.walk 掃整個目錄，
   把裡面所有 .parquet / .jsonl / .csv 都當成評測檔。

   所以 `datasets/ikala__tmmluplus/geography_of_taiwan.parquet` 這種寫法
   會直接被擋下；而指向整個 `datasets/ikala__tmmluplus/` 又會跑滿 66 科。
   要只跑幾科，就得**另外做一個只放那幾科的目錄**——這支程式就是做這件事。

   實作上用**相對路徑的 symlink**，不複製檔案：
   os.walk 會把 symlink 當成一般檔案列出來（已實測），
   所以 Twinkle Eval 讀得到，而磁碟不會多出一份。

用法：
    python scripts/make_eval_subset.py                 # 建立 smoke 與 tw10 兩個子集
    python scripts/make_eval_subset.py --list          # 列出所有可用科目
    python scripts/make_eval_subset.py --subset smoke  # 只建其中一個
    python scripts/make_eval_subset.py --copy          # 用複製取代 symlink（1 MB，無妨）

輸出：
    datasets/subsets/tmmluplus_smoke/   3 科，確認流程與時間成本用
    datasets/subsets/tmmluplus_tw10/    10 科，正式報告用
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets" / "ikala__tmmluplus"
OUT = ROOT / "datasets" / "subsets"

# 選取原則（沿用 Week 1 定案）：台灣在地情境 + 繁中語言能力，
# 避開純理工科目 —— 那些考的是學科知識，不是繁中／台灣情境，
# 放進來會稀釋掉這個實驗要 measure 的東西。
SUBSETS = {
    "tmmluplus_smoke": {
        "desc": "3 科試水溫：先確認流程正確與單科耗時，再決定要不要擴大",
        "subjects": [
            "geography_of_taiwan",
            "taiwanese_hokkien",
            "three_principles_of_people",
        ],
    },
    "tmmluplus_tw10": {
        "desc": "正式版 10 科：台灣在地情境 + 繁中語言能力",
        "subjects": [
            "geography_of_taiwan",
            "taiwanese_hokkien",
            "three_principles_of_people",
            "national_protection",
            "administrative_law",
            "introduction_to_law",
            "taxation",
            "junior_chinese_exam",
            "tve_chinese_language",
            "chinese_language_and_literature",
        ],
    },
}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=list(SUBSETS) + ["all"], default="all")
    ap.add_argument("--list", action="store_true", help="列出所有可用科目後結束")
    ap.add_argument("--copy", action="store_true",
                    help="用複製取代 symlink（若檔案系統不支援 symlink）")
    args = ap.parse_args()

    if not SRC.is_dir():
        sys.exit(f"❌ 找不到 {SRC}\n"
                 f"   先跑：twinkle-eval --download-dataset ikala/tmmluplus")

    available = sorted(p.stem for p in SRC.glob("*.parquet"))
    if args.list:
        print(f"{SRC} 底下共 {len(available)} 科：\n")
        for i, s in enumerate(available, 1):
            size = (SRC / f"{s}.parquet").stat().st_size
            print(f"  {i:>2}. {s:<52} {human(size):>9}")
        return

    names = list(SUBSETS) if args.subset == "all" else [args.subset]
    OUT.mkdir(parents=True, exist_ok=True)

    for name in names:
        spec = SUBSETS[name]
        dst = OUT / name
        dst.mkdir(parents=True, exist_ok=True)

        missing = [s for s in spec["subjects"] if s not in available]
        if missing:
            sys.exit(f"❌ {name}：找不到這些科目 {missing}\n"
                     f"   用 --list 看有哪些可用")

        total = 0
        print(f"\n▶ {name}  —— {spec['desc']}")
        for s in spec["subjects"]:
            src = SRC / f"{s}.parquet"
            link = dst / f"{s}.parquet"
            # 舊的先清掉，避免殘留上一次的科目
            if link.is_symlink() or link.exists():
                try:
                    link.unlink()
                except OSError as e:
                    print(f"    ⚠️ 清不掉舊的 {link.name}：{e}")
            if args.copy:
                shutil.copy2(src, link)
            else:
                # 相對路徑：整個 repo 搬家也不會斷
                link.symlink_to(os.path.relpath(src, dst))
            total += src.stat().st_size
            print(f"    ✓ {s:<50} {human(src.stat().st_size):>9}")
        print(f"  → {dst}  （{len(spec['subjects'])} 科，{human(total)}）")

        # 驗收：模擬 Twinkle Eval 的 os.walk，確認它真的讀得到
        found = []
        for root, dirs, files in os.walk(dst):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            found += [f for f in files if f.endswith(".parquet")]
        ok = len(found) == len(spec["subjects"])
        print(f"  {'✅' if ok else '❌'} os.walk 掃到 {len(found)} 個檔案"
              f"（Twinkle Eval 會這樣找）")

    print("\n把這一行填進評測 config 的 dataset_paths：")
    for name in names:
        print(f'    - "datasets/subsets/{name}"')
    print("\n⚠️ 注意是**目錄**不是檔案 —— Twinkle Eval 的 validate_dataset_path()"
          " 明確要求 isdir()。")


if __name__ == "__main__":
    main()
