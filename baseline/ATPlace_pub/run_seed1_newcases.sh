#!/usr/bin/env bash
# run_seed1_newcases.sh — 三个新 case 各跑 seed1 一次。
#
# 与 run_seed1-5.sh / result/ 里现有 12 行完全同口径 (thermal / Thermal-aware.json /
# ATP_TIMEOUT=30 / 共享 $REPO_ROOT/thermal / 串行), 所以数可以直接与 ATtime.log 并列。
# hp6_m seed1 已有 layout.json, 自动跳过。
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO/cases"
RESULT_DIR="$REPO/result"

MODE=thermal
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT=30
OUTER_TIMEOUT=3600
CASES=(hp6_m_bump xerox6_m_bump xerox7_m_bump)
SEED=1

log() { echo "[$(date '+%F %T')] $*"; }

for case in "${CASES[@]}"; do
  case_dir="$CASES_DIR/$case"
  out_dir="$RESULT_DIR/$case"
  seed_dir="$out_dir/seed${SEED}"
  mkdir -p "$seed_dir"

  if [ -f "$seed_dir/layout.json" ]; then
    log "SKIP  case=$case seed=$SEED (已有 layout.json)"
    continue
  fi

  "$PYTHON" -c '
import json, sys
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src)); d["random_seed"] = seed
json.dump(d, open(dst, "w"), indent=2, ensure_ascii=False)
' "$case_dir/$PARAM_NAME" "$SEED" "$out_dir/param.json"

  t0=$(date +%s)
  log "START case=$case seed=$SEED"
  ATPLACE_TIMEOUT="$ATP_TIMEOUT" timeout -k 30 "$OUTER_TIMEOUT" \
    "$PYTHON" "$REPO/reproduce.py" --case "$case" --mode "$MODE" \
    --case-dir "$case_dir" --param-file "$out_dir/param.json" --out-dir "$out_dir" \
    > "$out_dir/run_seed${SEED}.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))

  if [ -f "$seed_dir/layout.json" ]; then
    log "DONE  case=$case seed=$SEED ${elapsed}s  $(grep -o 'Total time [0-9.]*' "$seed_dir/run.log" | tail -1)"
  else
    log "FAIL  case=$case seed=$SEED ${elapsed}s rc=$rc"
  fi
done
log "=== 结束 ==="
