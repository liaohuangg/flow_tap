#!/usr/bin/env bash
# queue_wlsweep_c7_after_50set.sh — 等正在跑的 cases_50set 扫描结束后, 接着扫 Case7_bump。
#
# 为什么串而不并: 两边都是同一个 reproduce.py, 每进程 ~91 线程 (见记忆 atplace-run-cost),
# 同时跑只会互相拖慢并抬高超时风险 —— Case7 单槽本来就 ~20min。所以排队。
#
# 本脚本只做两件事:
#   1. 等 WAIT_PID (cases_50set 那批的 driver) 退出;
#   2. 用它自己的参数调 run_wlsweep_50set.sh 扫 Case7_bump。
#
# Case7_bump 的既有结果在 result_wlsweep_c7/Case7_bump (wl00~wl05 已落 layout.json),
# 扫描口径沿用上一轮: WL_MAX=0.01 共 50 点均匀网格, seed 固定 1。区别只有
# JOBS=4 (用户 2026-09-18 指定: Case7 开 4 线程)。已完成的槽位靠 run_one 里
# seed1/layout.json 的存在性自动跳过, 不用手工排范围。
#
# 注意: 上一轮 c7 被中断时在 wl06/wl07 留下过空的 .lock 目录 (run_one 见到 mkdir 失败
# 会记 "[skip] 另一进程正在跑" 而永远不跑那个槽) —— 已手工清掉。以后再手工接续 c7,
# 记得先 `ls -d result_wlsweep_c7/Case7_bump/*/.lock` 看一眼。
#
# 用法:
#   WAIT_PID=12345 ./queue_wlsweep_c7_after_50set.sh     # 前台等+跑
#   nohup WAIT_PID=12345 ./queue_wlsweep_c7_after_50set.sh >> result_wlsweep_c7/driver.log 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WAIT_PID="${WAIT_PID:?需要 WAIT_PID=<前一批 driver 的 pid>}"
WAIT_TAG="${WAIT_TAG:-run_wlsweep_50set}"

log() { echo "[$(date '+%F %T')] $*"; }

log "等 pid=$WAIT_PID ($WAIT_TAG) 结束 ..."
while [ -n "$(ps -p "$WAIT_PID" -o args= 2>/dev/null | grep -F "$WAIT_TAG")" ]; do
  sleep 60
done
log "前一批已结束 -> 开始扫 Case7_bump (JOBS=${JOBS:-4})"

CASES_OVERRIDE=Case7_bump \
WL_MAX=0.01 \
JOBS="${JOBS:-4}" \
OUTER_TIMEOUT="${OUTER_TIMEOUT:-10800}" \
RESULT_DIR="$PWD/result_wlsweep_c7" \
  ./run_wlsweep_50set.sh
rc=$?
log "Case7_bump 扫描结束 (rc=$rc)"
