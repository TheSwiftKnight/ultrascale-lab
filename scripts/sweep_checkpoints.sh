#!/usr/bin/env bash
# sweep_checkpoints.sh — 掃描 LoRA checkpoint，畫「訓練步數 vs 格式保留率」曲線
#
# 用法：
#   bash scripts/sweep_checkpoints.sh --dry-run     # 先看它會做什麼，不動任何東西
#   bash scripts/sweep_checkpoints.sh               # 正式跑（約 7–8 小時）
#   bash scripts/sweep_checkpoints.sh --steps "0000200 0001000"   # 只跑其中兩個
#
# 為什麼需要這支腳本，而不是直接寫個 for 迴圈：
#
# 【坑 1】mlx_lm.fuse 沒有 --adapter-file。
#   它只吃 --model / --save-path / --adapter-path / --upload-repo /
#   --dequantize / --export-gguf / --gguf-path，而 tuner/utils.py 的
#   load_adapters() 把檔名寫死成 adapter_path/"adapters.safetensors"。
#   → 必須先把 000XXXX_adapters.safetensors 複製成 adapters.safetensors。
#
# 【坑 2 —— 會讓整輪白跑】run_eval.sh 的 tuned 模式把模型寫死成
#   out/gemma4-e4b-tw。只改 config 的 model.name 是沒有用的：
#   twinkle-eval 只是把那個字串當 API 參數送出去，而 mlx_lm.server
#   對 model id 照單全收。所以五個 checkpoint 會全部被 1000 步的模型
#   評測一遍，五份結果一模一樣，而且不會有任何錯誤訊息。
#   → 這裡改用 run_eval.sh custom <模型> <config>，且 run_eval.sh
#     已加上「config.model.name 必須等於 --model」的硬性斷言。
#
# 【坑 3】不要 sed 原本的 configs/eval_gemma4_e4b_tuned.yaml。
#   那是 git 追蹤的檔案，迴圈中途掛掉會留下指向暫存路徑的髒設定。
#   → 每個 checkpoint 產一份 configs/_sweep/ 底下的暫存 config。
#
# 【坑 4】磁碟。每個融合模型 4.0 GB，五個就 20 GB。
#   → 評測完立刻刪掉，同一時間磁碟上只會有一個。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BASE_MODEL="mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_DIR="out/lora-e4b"
TEMPLATE="configs/eval_gemma4_e4b_tuned.yaml"
SWEEP_CFG_DIR="configs/_sweep"
SWEEP_OUT="results/sweep"
STEPS="0000200 0000400 0000600 0000800 0001000"
DRY_RUN=0
KEEP_FUSED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1; shift ;;
    --keep-fused) KEEP_FUSED=1; shift ;;
    --steps)      STEPS="$2"; shift 2 ;;
    --adapter-dir) ADAPTER_DIR="$2"; shift 2 ;;
    -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "不認得的參數：$1"; exit 1 ;;
  esac
done

mkdir -p "$SWEEP_CFG_DIR" "$SWEEP_OUT" logs

echo "════════════════════════════════════════════════════════════"
echo " checkpoint 掃描"
echo "════════════════════════════════════════════════════════════"
echo "  base 模型   : $BASE_MODEL"
echo "  adapter 來源: $ADAPTER_DIR"
echo "  要跑的步數  : $STEPS"
echo "  乾跑模式    : $([[ $DRY_RUN == 1 ]] && echo 是 || echo 否)"
echo

# ── 前置檢查 ──────────────────────────────────────────────────
[[ -f "$TEMPLATE" ]] || { echo "❌ 找不到 $TEMPLATE"; exit 1; }
for step in $STEPS; do
  f="$ADAPTER_DIR/${step}_adapters.safetensors"
  [[ -f "$f" ]] || { echo "❌ 找不到 $f"; echo "   有的是："; ls "$ADAPTER_DIR"/*_adapters.safetensors 2>/dev/null || true; exit 1; }
done
[[ -f "$ADAPTER_DIR/adapter_config.json" ]] || { echo "❌ 找不到 $ADAPTER_DIR/adapter_config.json"; exit 1; }

AVAIL_GB=$(df -k . | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
echo "▶ 可用磁碟 ${AVAIL_GB} GB（一次只會放一個 4 GB 的融合模型）"
if [[ "$AVAIL_GB" -lt 12 ]]; then
  echo "❌ 磁碟不足 12 GB，先清一些再跑。"; exit 1
fi

# ── adapter 內容必須真的不一樣 ─────────────────────────────────
# 如果複製錯檔、或 save_every 其實沒生效，五個 checkpoint 會是同一份。
# 那樣曲線會是一條平線，而且看起來很「合理」。
echo
echo "▶ 確認五個 checkpoint 的內容互不相同"
PREV_HASH=""
for step in $STEPS; do
  H=$(shasum -a 256 "$ADAPTER_DIR/${step}_adapters.safetensors" | cut -c1-12)
  echo "   $step  sha256:$H"
  if [[ "$H" == "$PREV_HASH" ]]; then
    echo "   ❌ 這一步和上一步的 adapter 完全相同 —— 訓練可能沒有真的在存 checkpoint。"; exit 1
  fi
  PREV_HASH="$H"
done

# ── 主迴圈 ────────────────────────────────────────────────────
for step in $STEPS; do
  echo
  echo "════════════════════════════════════════════════════════════"
  echo " step $step"
  echo "════════════════════════════════════════════════════════════"

  DEST="$SWEEP_OUT/ckpt-${step}.json"
  if [[ -f "$DEST" ]]; then
    echo "▶ $DEST 已存在，跳過。（要重跑就先刪掉它）"
    continue
  fi

  CKPT_DIR="out/ckpt-${step}"
  FUSED="out/fused-${step}"
  CFG="$SWEEP_CFG_DIR/eval_ckpt-${step}.yaml"

  echo "▶ 準備 $CKPT_DIR"
  if [[ $DRY_RUN == 0 ]]; then
    rm -rf "$CKPT_DIR"; mkdir -p "$CKPT_DIR"
    cp "$ADAPTER_DIR/adapter_config.json"            "$CKPT_DIR/"
    cp "$ADAPTER_DIR/${step}_adapters.safetensors"   "$CKPT_DIR/adapters.safetensors"
    # 對帳：複製過去的必須和來源一致
    a=$(shasum -a 256 "$ADAPTER_DIR/${step}_adapters.safetensors" | cut -d' ' -f1)
    b=$(shasum -a 256 "$CKPT_DIR/adapters.safetensors"            | cut -d' ' -f1)
    [[ "$a" == "$b" ]] || { echo "❌ 複製後 checksum 不符"; exit 1; }
    echo "   ok（adapter_config.json + adapters.safetensors）"
  else
    echo "   [乾跑] cp ${step}_adapters.safetensors → $CKPT_DIR/adapters.safetensors"
  fi

  echo "▶ 融合 → $FUSED"
  if [[ $DRY_RUN == 0 ]] && [[ -f "$FUSED/config.json" ]] \
     && compgen -G "$FUSED/*.safetensors" > /dev/null \
     && [[ $(du -sm "$FUSED" | cut -f1) -gt 3500 ]]; then
    echo "   已經有完整的 ${FUSED}（$(du -sh "$FUSED" | cut -f1)），跳過融合。"
  elif [[ $DRY_RUN == 0 ]]; then
    rm -rf "$FUSED"
    python -m mlx_lm.fuse \
      --model       "$BASE_MODEL" \
      --adapter-path "$CKPT_DIR" \
      --save-path    "$FUSED"
    [[ -f "$FUSED/config.json" ]] || { echo "❌ 融合後沒有 config.json"; exit 1; }
    echo "   ok（$(du -sh "$FUSED" | cut -f1)）"
  else
    echo "   [乾跑] python -m mlx_lm.fuse --model $BASE_MODEL --adapter-path $CKPT_DIR --save-path $FUSED"
  fi

  echo "▶ 產生暫存 config → $CFG"
  python3 - "$TEMPLATE" "$CFG" "$FUSED" <<'PY'
import sys, yaml
tpl, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = yaml.safe_load(open(tpl))
before = cfg["evaluation"]["system_prompt"]["zh"]
cfg["model"]["name"] = model
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
# 對帳：system_prompt 有反斜線（\box{}），round-trip 一定要逐字相同，
# 不然評測會 100% 無法解析（8/13 那次就是 system_prompt 出問題）。
after = yaml.safe_load(open(out))["evaluation"]["system_prompt"]["zh"]
assert after == before, "system_prompt 在 YAML round-trip 之後變了，中止"
assert yaml.safe_load(open(out))["model"]["name"] == model
print(f"   ok（model.name = {model}，system_prompt 逐字相同）")
PY

  echo "▶ 評測"
  if [[ $DRY_RUN == 0 ]]; then
    # 上一輪的 server 要先完全退掉，否則 run_eval.sh 會沿用它 ——
    # 那就等於用上一個 checkpoint 的模型評測這一個。
    for i in $(seq 1 30); do
      lsof -i :1234 -sTCP:LISTEN >/dev/null 2>&1 || break
      [[ $i == 1 ]] && echo "   等 :1234 釋放（上一輪的 server 還在收尾）"
      sleep 2
    done
    if lsof -i :1234 -sTCP:LISTEN >/dev/null 2>&1; then
      echo "❌ :1234 等了 60 秒還有人佔著："
      lsof -i :1234 -sTCP:LISTEN
      echo "   → 手動處理掉再重跑（已完成的步數會自動跳過）。"
      exit 1
    fi

    BEFORE_LIST=$(mktemp); ls results/results_*.json 2>/dev/null | sort > "$BEFORE_LIST" || true
    bash scripts/run_eval.sh custom "$FUSED" "$CFG"
    AFTER_LIST=$(mktemp); ls results/results_*.json 2>/dev/null | sort > "$AFTER_LIST"
    NEW=$(comm -13 "$BEFORE_LIST" "$AFTER_LIST" | tail -1)
    rm -f "$BEFORE_LIST" "$AFTER_LIST"
    if [[ -z "$NEW" ]]; then
      echo "❌ 評測沒有產生新的 results/*.json"; exit 1
    fi
    cp "$NEW" "$DEST"
    echo "   結果：$NEW → $DEST"
  else
    echo "   [乾跑] bash scripts/run_eval.sh custom $FUSED $CFG"
    echo "   [乾跑] 結果會複製到 $DEST"
  fi

  if [[ $DRY_RUN == 0 && $KEEP_FUSED == 0 ]]; then
    echo "▶ 清掉 $FUSED 與 ${CKPT_DIR}（省磁碟）"
    rm -rf "$FUSED" "$CKPT_DIR"
  fi
done

echo
echo "════════════════════════════════════════════════════════════"
if [[ $DRY_RUN == 1 ]]; then
  echo " 乾跑結束，什麼都沒動。確認上面的路徑沒問題就拿掉 --dry-run。"
else
  echo " 全部完成。彙整："
  echo "   python3 scripts/summarize_sweep.py"
fi
echo "════════════════════════════════════════════════════════════"
