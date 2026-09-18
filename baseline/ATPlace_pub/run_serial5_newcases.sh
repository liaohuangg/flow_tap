#!/usr/bin/env bash
# run_serial5_newcases.sh — hp6_m / xerox6_m / xerox7_m 三个 case 的 AT 计时。
#
# 与 run_seed1-5.sh 完全同口径(thermal / Thermal-aware.json / ATP_TIMEOUT=30 /
# 每个 seed 一次 100-iter 求解), 只是**串行**跑 —— result_wlsweep 那批是 JOBS=12
# 并发, 单次求解墙钟被拉长约 2 倍, 不能与 ATtime.log 里现有的 12 行并列。
#
# 输出: result/<case>_bump/seed<seed>/{layout.json,run.log}, 与现有行同位置。
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO_ROOT/cases"
RESULT_DIR="$REPO_ROOT/result"

MODE=thermal
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT=30
OUTER_TIMEOUT=3600
SEEDS=(1 2 3 4 5)
CASES=(hp6_m_bump xerox6_m_bump xerox7_m_bump)

log() { echo "[$(date '+%F %T')] $*"; }

for case in "${CASES[@]}"; do
  case_dir="$CASES_DIR/$case"
  out_dir="$RESULT_DIR/$case"
  mkdir -p "$out_dir"
  for seed in "${SEEDS[@]}"; do
    seed_dir="$out_dir/seed${seed}"
    if [ -f "$seed_dir/layout.json" ]; then
      log "SKIP  case=$case seed=$seed"
      continue
    fi
    mkdir -p "$seed_dir"

    "$PYTHON" -c '
import json, sys
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src)); d["random_seed"] = seed
json.dump(d, open(dst, "w"), indent=2, ensure_ascii=False)
' "$case_dir/$PARAM_NAME" "$seed" "$out_dir/param.json"

    t0=$(date +%s)
    log "START case=$case seed=$seed"
    ATPLACE_TIMEOUT="$ATP_TIMEOUT" timeout -k 30 "$OUTER_TIMEOUT" \
      "$PYTHON" "$REPO_ROOT/reproduce.py" --case "$case" --mode "$MODE" \
      --case-dir "$case_dir" --param-file "$out_dir/param.json" --out-dir "$out_dir" \
      > "$out_dir/run_seed${seed}.log" 2>&1
    rc=$?
    elapsed=$(( $(date +%s) - t0 ))

    if [ -f "$seed_dir/layout.json" ]; then
      log "DONE  case=$case seed=$seed $((elapsed/60))m$((elapsed%60))s  $(grep -o 'Total time [0-9.]*' "$seed_dir/run.log" | tail -1)"
    else
      log "FAIL  case=$case seed=$seed $((elapsed/60))m$((elapsed%60))s rc=$rc"
    fi
  done
done
log "=== 三个 case 串行跑完 ==="
