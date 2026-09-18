#!/usr/bin/env bash
# run_par_newcases.sh — hp6_m / xerox6_m / xerox7_m 三个 case 的 AT 计时, JOBS 路并行。
#
# 与 run_serial5_newcases.sh 同一套求解参数 (thermal / Thermal-aware.json /
# ATP_TIMEOUT=30 / 每 slot 一次 100-iter), 差别只是并发 —— 并发会拉长单次墙钟,
# 见文件尾的说明。
#
# 两个并发必须处理的点 (沿用 run_wlsweep_50set.sh 的做法):
#   1. param 文件按 seed 分开 (param_seed<k>.json) —— 否则同 case 的多个 worker
#      会互相覆盖同一个 param.json。
#   2. 每次求解给私有 ATPLACE_THERMAL_DIR —— HotSpot 中间文件是固定文件名,
#      共享目录会互相覆盖、读回别人算的温度, 布局被污染。
#
# 输出: result/<case>_bump/seed<seed>/{layout.json,run.log}
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO/cases"
RESULT_DIR="$REPO/result"

MODE=thermal
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT=30
OUTER_TIMEOUT=3600
JOBS="${JOBS:-5}"
SEEDS=(1 2 3 4 5)
CASES=(hp6_m_bump xerox6_m_bump xerox7_m_bump)

LOG="$REPO/par_newcases.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_one() {
  local case="$1" seed="$2"
  local out_dir="$RESULT_DIR/$case"
  local seed_dir="$out_dir/seed$seed"
  mkdir -p "$seed_dir"

  if [ -f "$seed_dir/layout.json" ]; then
    log "SKIP  case=$case seed=$seed"
    return 0
  fi

  local param="$out_dir/param_seed${seed}.json"
  "$PYTHON" -c '
import json, sys
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src)); d["random_seed"] = seed
json.dump(d, open(dst, "w"), indent=2, ensure_ascii=False)
' "$CASES_DIR/$case/$PARAM_NAME" "$seed" "$param"

  # 私有 HotSpot 目录: 复制 config + 软链二进制
  local th_dir="$seed_dir/thermal"
  rm -rf "$th_dir"; mkdir -p "$th_dir"
  cp "$REPO/thermal/hotspot.config" "$th_dir/hotspot.config"
  ln -sf "$REPO/thermal/hotspot" "$th_dir/hotspot"

  local t0; t0=$(date +%s)
  log "START case=$case seed=$seed"
  ATPLACE_TIMEOUT="$ATP_TIMEOUT" ATPLACE_THERMAL_DIR="$th_dir" PYTHONHASHSEED=0 \
    timeout -k 30 "$OUTER_TIMEOUT" \
    "$PYTHON" "$REPO/reproduce.py" --case "$case" --mode "$MODE" \
    --case-dir "$CASES_DIR/$case" --param-file "$param" --out-dir "$out_dir" \
    > "$out_dir/run_seed${seed}.log" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - t0 ))

  if [ -f "$seed_dir/layout.json" ]; then
    log "DONE  case=$case seed=$seed ${elapsed}s  $(grep -o 'Total time [0-9.]*' "$seed_dir/run.log" | tail -1)"
  else
    log "FAIL  case=$case seed=$seed ${elapsed}s rc=$rc"
  fi
}

: > "$LOG"
log "开始: JOBS=$JOBS, ${#CASES[@]} case x ${#SEEDS[@]} seed"

for case in "${CASES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 3; done
    run_one "$case" "$seed" &
  done
done
wait
log "=== 全部结束 ==="
