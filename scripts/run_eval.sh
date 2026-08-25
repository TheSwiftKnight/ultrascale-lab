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
  base)  MODEL="mlx-community/gemma-4-e4b-it-4bit"; CFG="configs/eval_gemma4_e4b_base.yaml"; TAG="base" ;;
  tuned) MODEL="out/gemma4-e4b-tw";                 CFG="configs/eval_gemma4_e4b_tuned.yaml"; TAG="tuned" ;;
  custom)
    # 給 sweep_checkpoints.sh 用：模型與 config 都由呼叫端指定，不寫死。
    MODEL="${2:-}"; CFG="${3:-}"
    if [[ -z "$MODEL" || -z "$CFG" ]]; then
      echo "用法：bash scripts/run_eval.sh custom <模型路徑> <config.yaml>"; exit 1
    fi
    TAG="$(basename "$MODEL")"
    ;;
  *) echo "用法：bash scripts/run_eval.sh [base|tuned|custom <模型> <config>]"; exit 1 ;;
esac

PORT=1234
HOST=127.0.0.1
URL="http://${HOST}:${PORT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── 0. config 與 server 的模型必須是同一個 ───────────────────────
# 這一關是後來補的。原因：mlx_lm.server 起哪個模型是由 --model 決定，
# 而 twinkle-eval 只是把 config 的 model.name 當字串送進 API —— server
# 對 model id 照單全收，不會比對。所以「config 指到 A、server 起的是 B」
# 完全不會報錯，只會安靜地評測到錯的模型。
# checkpoint 掃描那種一次跑五個模型的情境，沒有這一關會五份結果全一樣。
CFG_MODEL="$(python3 -c "
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))['model']['name'])
" "$CFG")"
if [[ "$CFG_MODEL" != "$MODEL" ]]; then
  echo "❌ 模型不一致，中止："
  echo "   本腳本會用 --model '$MODEL' 起 server"
  echo "   但 $CFG 的 model.name 是 '$CFG_MODEL'"
  echo "   → 兩者必須相同，否則會評測到錯的模型而且不會有任何錯誤訊息。"
  exit 1
fi
echo "▶ 模型對帳通過：$MODEL"

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
        > "logs/server_${TAG}.log" 2>&1 &
  echo $! > "/tmp/mlxserver_${TAG}.pid"
  KILL_AFTER=1
  echo "   PID $(cat /tmp/mlxserver_${TAG}.pid)，log 在 logs/server_${TAG}.log"
fi

cleanup() {
  if [[ "${KILL_AFTER:-0}" == "1" && -f "/tmp/mlxserver_${TAG}.pid" ]]; then
    echo; echo "▶ 收掉 server PID $(cat /tmp/mlxserver_${TAG}.pid)"
    kill "$(cat /tmp/mlxserver_${TAG}.pid)" 2>/dev/null || true
    rm -f "/tmp/mlxserver_${TAG}.pid"
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
  echo "❌ /v1/models 回 ${CODE}（等了 180 秒）。看 logs/server_${TAG}.log。"
  exit 1
fi

echo "▶ server 上的 model id："
if [[ "${ALLOW_MODEL_ID_MISMATCH:-0}" == "1" ]]; then echo "   （已略過 model id 對帳）"; fi
curl -s "${URL}/v1/models" | WANT="$MODEL" ALLOW="${ALLOW_MODEL_ID_MISMATCH:-0}" python3 -c "
import json, sys, os
from pathlib import Path

d = json.load(sys.stdin)
ids = [m.get('id') for m in d.get('data', [])]
want = os.environ['WANT']

# mlx_lm.server 會把本機模型的 id 報成**絕對路徑**，而我們傳的是相對路徑，
# 而且它還會把 HF 快取裡的其他模型一起列出來。所以不能直接做字串比對，
# 要先把看起來像路徑的 id 正規化成絕對路徑再比。
def norm(x):
    try:
        p = Path(x).expanduser()
        if p.exists():
            return str(p.resolve())
    except OSError:
        pass
    return x

want_n = norm(want)
ids_n = [norm(i) for i in ids]
for i, n in zip(ids, ids_n):
    mark = '  ← 這個' if n == want_n else ''
    print(f'   - {i}{mark}')

if want_n not in ids_n and os.environ.get('ALLOW') != '1':
    print(f'\n❌ 要評測的是 {want}')
    print(f'   正規化後 = {want_n}')
    print( '   但 server 上沒有這個模型。可能是前一輪的 server 還沒收掉，或 :1234 被別人佔著。')
    print( '   → 確定要照跑就設 ALLOW_MODEL_ID_MISMATCH=1。')
    sys.exit(1)
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
time twinkle-eval --config "$CFG" 2>&1 | tee "logs/eval_${TAG}_run.log"

echo
echo "▶ 產出："
ls -lt results/ | head -10
