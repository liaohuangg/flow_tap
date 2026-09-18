#!/usr/bin/env bash
# wait_then_run_serial5.sh — 等机器空了, 再跑 run_serial5_newcases.sh。
#
# 为什么要等: 2026-09-18 12:19 那次跑在 load 63/28 的机器上, 单 iter 15.8s
# (acend910 干净时是 2.5s), 测出来的 Total time 会被争抢污染 —— 正是要避开的
# 那个 ~2 倍膨胀。必须等 Case7 wlsweep 和 RL 训练都退出。
#
# 判空: 没有 reproduce.py / train.py 在跑, 且 load1 < THRESH。
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVAL="${INTERVAL:-300}"   # 轮询间隔(秒)
THRESH="${THRESH:-8}"         # load1 阈值, 28 核
MAX_WAIT="${MAX_WAIT:-86400}" # 最多等 24 小时

log() { echo "[$(date '+%F %T')] $*"; }

deadline=$(( $(date +%s) + MAX_WAIT ))
log "开始等机器空 (阈值 load1<$THRESH, 无 reproduce.py/train.py), 最多 ${MAX_WAIT}s"

while :; do
  busy=$(pgrep -fc 'reproduce\.py|/train\.py' 2>/dev/null || true)
  busy="${busy:-0}"
  load1=$(cut -d' ' -f1 /proc/loadavg)

  if [ "$busy" = "0" ] && awk -v l="$load1" -v t="$THRESH" 'BEGIN{exit !(l<t)}'; then
    log "机器空了 (load1=$load1, busy=0) -> 开始串行跑"
    bash "$REPO/run_serial5_newcases.sh"
    log "=== 串行 AT 跑完, 结束 ==="
    exit 0
  fi

  if [ "$(date +%s)" -gt "$deadline" ]; then
    log "等超 24 小时仍不空 (load1=$load1, busy=$busy), 放弃"
    exit 1
  fi

  log "还在忙 (load1=$load1, busy=$busy), $((INTERVAL/60)) 分钟后再看"
  sleep "$INTERVAL"
done
