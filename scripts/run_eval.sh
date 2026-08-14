#!/usr/bin/env bash
# run_eval.sh — 安全地跑一次 TMMLU+ 評測
#
# 用法：
#   bash scripts/run_eval.sh base     # 微調前
#   bash scripts/run_eval.sh tuned    # 微調後
#
# 這支腳本存在的理由：8/13 那次評測送出 4,144 個請求、全部 502、零個成功，
# 跑了 15 分鐘才知道失敗。這裡在正式開跑之前強制做三件事：
#   1. 把 localhost 排除在 HTTP proxy 之外（502 的元凶）
#   2. 確認 /v1/models 回得來，而且 model id 和 config 對得上
#   3. 先送**一題**，確認拿得到 200 和有內容的回覆，失敗就直接中止
# 任何一關沒過就不會進到 twinkle-eval。

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  base)  MODEL="mlx-community/gemma-4-e4b-it-4bit"; CFG="configs/eval_gemma4_e4b_base.yaml" ;;
  tuned) MODEL="out/gemma4-e4b-tw";                 CFG="configs/eval_gemma4_e4b_tuned.yaml" ;;
  *) echo "用法：bash scripts/run_eval.sh [base|tuned]"; exit 1 ;;
esac

PORT=1234
HOST=127.0.0.1
URL="http://${HOST}:${PORT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── 1. 代理 ───────────────────────────────────────────────────────
# 這兩行就是 8/13 那 4,144 個 502 的解法。httpx / requests / openai 都吃這個變數。
export no_proxy="127.0.0.1,localhost,::1"
export NO_PROXY="127.0.0.1,localhost,::1"
echo "▶ no_proxy = $no_proxy"

# ── 2. 端口上是誰 ─────────────────────────────────────────────────
echo
echo "▶ 檢查 :${PORT} ..."
if lsof -i ":${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -i ":${PORT}" -sTCP:LISTEN
  OWNER="$(lsof -t -i ":${PORT}" -sTCP:LISTEN | head -1)"
  CMD="$(ps -o comm= -p "$OWNER" 2>/dev/null || true)"
  echo "   佔用者 PID=$OWNER  ($CMD)"
  if [[ "$CMD" == *"LM Studio"* || "$CMD" == *"lms"* ]]; then
    echo "❌ :${PORT} 上跑的是 LM Studio，不是 mlx_lm.server。"
    echo "   → 先把 LM Studio 完全退出（不是只關視窗），再跑一次這支腳本。"
    exit 1
  fi
  KILL_AFTER=0
else
  echo "   沒人在聽，我來起 server。"
  mkdir -p logs
  nohup mlx_lm.server --model "$MODEL" --host "$HOST" --port "$PORT" \
        > "logs/server_${MODE}.log" 2>&1 &
  echo $! > "/tmp/mlxserver_${MODE}.pid"
  KILL_AFTER=1
  echo "   PID $(cat /tmp/mlxserver_${MODE}.pid)，log 在 logs/server_${MODE}.log"
fi

cleanup() {
  if [[ "${KILL_AFTER:-0}" == "1" && -f "/tmp/mlxserver_${MODE}.pid" ]]; then
    echo; echo "▶ 收掉 server PID $(cat /tmp/mlxserver_${MODE}.pid)"
    kill "$(cat /tmp/mlxserver_${MODE}.pid)" 2>/dev/null || true
    rm -f "/tmp/mlxserver_${MODE}.pid"
  fi
}
trap cleanup EXIT

# ── 3. 等 server 起來 ─────────────────────────────────────────────
echo
echo -n "▶ 等 ${URL}/v1/models "
for i in $(seq 1 90); do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' "${URL}/v1/models" || true)"
  [[ "$CODE" == "200" ]] && break
  echo -n "."
  sleep 2
done
echo
if [[ "$CODE" != "200" ]]; then
  echo "❌ /v1/models 回 $CODE（等了 180 秒）。看 logs/server_${MODE}.log。"
  exit 1
fi

echo "▶ server 上的 model id："
curl -s "${URL}/v1/models" | python3 -c "
import json,sys
d=json.load(sys.stdin)
ids=[m.get('id') for m in d.get('data',[])]
for i in ids: print('   -',i)
want='''$MODEL'''
if want not in ids:
    print(f'''\n⚠️  config 裡寫的是 {want}，但 server 報的是 {ids}''')
    print('   → mlx_lm.server 通常照單全收，不一定會擋；但如果評測回 404，就是這裡不合。')
"

# ── 4. 送一題探路（最重要的一關）────────────────────────────────
# 不只看 HTTP 200 —— 還要用 config 裡真正的 system_prompt 送一題，
# 再用 twinkle-eval 自己的 BoxExtractor 確認抽得出答案。
# 只檢查 200 的話，會漏掉 8/13 23:55 那種「全部成功但 100% 無法解析」。
echo
python3 scripts/probe_eval.py "$CFG"

# ── 5. 正式跑 ─────────────────────────────────────────────────────
echo
echo "▶ twinkle-eval --validate"
twinkle-eval --validate --config "$CFG"

echo
echo "▶ 正式評測：$CFG"
echo "   smoke 子集 = 1,036 題（geography 768 / hokkien 129 / three_principles 139）"
echo "   8/13 實測 4.85 秒/題 → 一輪約 84 分鐘。可以先去做別的事。"
echo "   中途想看進度：tail -f logs/\$(ls -t logs | head -1)"
echo
time twinkle-eval --config "$CFG" 2>&1 | tee "logs/eval_${MODE}_run.log"

echo
echo "▶ 產出："
ls -lt results/ | head -10
