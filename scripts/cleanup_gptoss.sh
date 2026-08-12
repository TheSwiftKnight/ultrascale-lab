#!/usr/bin/env bash
# cleanup_gptoss.sh — Week 2 Step 0：清掉 GPT-OSS 相關、之後不會再用到的資源
#
# repo 內的檔案我已經幫你移到 _to_delete/gptoss_week1/ 了（78 MB）。
# 這支腳本處理**我碰不到的**兩個地方：HF 模型快取（十幾 GB）與 LM Studio。
#
# 用法：
#   bash scripts/cleanup_gptoss.sh          # 只列出要刪什麼，不動手（預設）
#   bash scripts/cleanup_gptoss.sh --yes    # 真的刪
set -u
DRY=1
[ "${1:-}" = "--yes" ] && DRY=0

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
run() {
  if [ $DRY -eq 1 ]; then printf '  [dry-run] %s\n' "$*"
  else printf '  $ %s\n' "$*"; eval "$@"; fi
}

say "1. Hugging Face 快取裡的 gpt-oss 權重"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
if [ -d "$HF_CACHE" ]; then
  found=0
  for d in "$HF_CACHE"/models--*gpt?oss* "$HF_CACHE"/models--*gpt-oss*; do
    [ -e "$d" ] || continue
    found=1
    printf '  %-72s %s\n' "$(basename "$d")" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    run "rm -rf '$d'"
  done
  [ $found -eq 0 ] && echo "  （沒找到 gpt-oss 的快取，可能已經清掉了）"
else
  echo "  找不到 $HF_CACHE"
fi

say "2. Xet 區塊快取（下載大模型時的中繼檔，可安全清空）"
XET="${HF_HOME:-$HOME/.cache/huggingface}/xet"
if [ -d "$XET" ]; then
  printf '  %s  %s\n' "$XET" "$(du -sh "$XET" 2>/dev/null | cut -f1)"
  run "rm -rf '$XET'"
else
  echo "  （沒有 xet 快取）"
fi

say "3. LM Studio 裡的 gpt-oss 模型"
LMS="$HOME/.lmstudio/models"
[ -d "$LMS" ] || LMS="$HOME/.cache/lm-studio/models"
if [ -d "$LMS" ]; then
  found=0
  while IFS= read -r d; do
    found=1
    printf '  %-72s %s\n' "${d#$LMS/}" "$(du -sh "$d" 2>/dev/null | cut -f1)"
    run "rm -rf '$d'"
  done < <(find "$LMS" -maxdepth 3 -type d -iname '*gpt*oss*' 2>/dev/null)
  [ $found -eq 0 ] && echo "  （LM Studio 裡沒有 gpt-oss 模型）"
else
  echo "  （找不到 LM Studio 模型目錄，若你是用 GUI 下載的，"
  echo "    請開 LM Studio → My Models → 找 gpt-oss → 右鍵 Delete）"
fi

say "4. repo 內已移到 _to_delete/ 的 GPT-OSS 產物"
D="$(cd "$(dirname "$0")/.." && pwd)/_to_delete"
if [ -d "$D" ]; then
  du -sh "$D"
  echo "  內容：Week 1 的 data/train+val、predict_memory.py、verify_load.py、"
  echo "        inspect_router.py、prepare_data.py、baseline_lmstudio.yaml、"
  echo "        reports/ 舊產物、舊評測 log、Harmony 版 notebook、week1_執行手冊.md"
  echo
  echo "  ⚠️ 先確認 Week 2 的資料管線跑通、拿到新的 data/train 之後再刪這個資料夾。"
  run "rm -rf '$D'"
else
  echo "  （已經清掉了）"
fi

say "5. 保留下來的東西（不要刪）"
cat <<'EOF'
  week1_執行總結.md            Week 1 的成果紀錄，最終報告的方法論依據
  notes/ch01,02,06,10.md      Playbook 章節筆記（公式部分與模型無關，
                              「🧪 對照我的實驗」小節在 Step 8 改寫成 Gemma 版）
  Eval/                       Twinkle Eval 原始碼
  datasets/ikala__tmmluplus/  TMMLU+ 66 科評測資料
  ultrascale-proposal_1.pptx  提案本體
EOF

if [ $DRY -eq 1 ]; then
  printf '\n\033[33m以上是 dry-run。確認沒問題後執行：bash scripts/cleanup_gptoss.sh --yes\033[0m\n'
else
  printf '\n\033[32m✅ 清理完成。用 df -h 確認空間釋出。\033[0m\n'
fi
