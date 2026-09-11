#!/usr/bin/env bash
# run_my_test.sh — 8-case sweep for ATPlace2.5D placement (baseline).
#
# For each of the 8 converted cases (thermal mode): run reproduce.py for a full
# 1 hour. Inside that hour, reproduce.py repeatedly solves 100-iteration runs
# under a fresh random_seed each time (seed = 1, 2, 3, ...), recording every
# attempt (seed, hpwl, overlaps, legal) and never stopping early on a legal
# layout.  Total wall-clock: 8 cases x 1 hour = 8 hours.
#
# reproduce.py returns: 0 = completed (collected results over the full budget),
# 1 = error.  It always writes summary.json (with attempt_log) + layout.json
# (best/lowest-HPWL) + layout.png.
#
# Results layout under RESULT_DIR (default: <repo>/result):
#   result/<case>/{param.json, layout.json, layout.png, summary.json, run.log, status}
#   result/results.csv                  (one row per case, appended as it goes)
#
# Re-run safety: a case that already has a "done" status file is skipped.

set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CASES_DIR="$REPO_ROOT/cases"
RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/result}"

MODE="${MODE:-thermal}"            # thermal | wl
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}" # 1 hour per case (reproduce.py manages its own budget)
OUTER_TIMEOUT=$((TIMEOUT_SEC + 300))  # safety net: hard-kill if reproduce.py overruns
BASE_SEED=1                          # each case starts its seed sequence at 1

CASES=(acend910_bump cpu-dram_bump hp11_m_bump multigpu_bump syn1_bump syn4_bump syn6_bump xerox8_m_bump)

case "$MODE" in
  thermal) PARAM_NAME="Thermal-aware.json" ;;
  wl)      PARAM_NAME="WL-driven.json" ;;
  *) echo "Unknown MODE=$MODE" >&2; exit 2 ;;
esac

CSV="$RESULT_DIR/results.csv"
mkdir -p "$RESULT_DIR"
if [[ ! -f "$CSV" ]]; then
  echo "case,mode,status,rc,best_hpwl,twl_m,num_legal,num_illegal,attempts,runtime_s,layout_exists,out_dir" > "$CSV"
fi

log() { echo "[$(date '+%F %T')] $*"; }

merge_param() {
  # $1 = source param json, $2 = seed, $3 = dest param json
  local src="$1" seed="$2" dst="$3"
  "$PYTHON" -c '
import json, sys
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(src) as f:
    data = json.load(f)
data["random_seed"] = seed
with open(dst, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
' "$src" "$seed" "$dst"
}

json_field() {
  # $1 = summary.json path, $2 = key
  [[ -f "$1" ]] || { echo ""; return; }
  "$PYTHON" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get(sys.argv[2], ""))' "$1" "$2" 2>/dev/null
}

run_one() {
  local case="$1"
  local case_dir="$CASES_DIR/$case"
  local out_dir="$RESULT_DIR/$case"
  mkdir -p "$out_dir"

  # resume: skip cases already completed
  if [[ -f "$out_dir/status" ]] && grep -q '^done$' "$out_dir/status"; then
    log "SKIP  case=$case (already done)"
    return 0
  fi

  local param_src="$case_dir/$PARAM_NAME"
  local param_dst="$out_dir/param.json"
  merge_param "$param_src" "$BASE_SEED" "$param_dst"

  log "START case=$case (mode=$MODE timeout=${TIMEOUT_SEC}s base_seed=$BASE_SEED)"
  local t0; t0=$(date +%s)

  timeout -k 30 "$OUTER_TIMEOUT" \
    "$PYTHON" "$REPO_ROOT/reproduce.py" \
      --case "$case" --mode "$MODE" \
      --case-dir "$case_dir" \
      --param-file "$param_dst" \
      --out-dir "$out_dir" \
    > "$out_dir/run.log" 2>&1
  local rc=$?
  local t1; t1=$(date +%s)

  # safety net: clear any orphaned children left behind by a timeout kill
  pkill -f "reproduce.py --case ${case} " 2>/dev/null || true

  local status
  if [[ $rc -eq 0 ]]; then
    status="done"
  else
    status="error"
  fi
  echo "$status" > "$out_dir/status"

  local summary="$out_dir/summary.json"
  local hpwl twl runtime nlegal nillegal attempts layout_exists
  hpwl=$(json_field "$summary" best_hpwl)
  twl=$(json_field "$summary" twl_m)
  runtime=$(json_field "$summary" runtime_s)
  nlegal=$(json_field "$summary" num_legal)
  nillegal=$(json_field "$summary" num_illegal)
  attempts=$(json_field "$summary" attempts)
  layout_exists="no"
  [[ -f "$out_dir/layout.json" ]] && layout_exists="yes"

  echo "$case,$MODE,$status,$rc,$hpwl,$twl,$nlegal,$nillegal,$attempts,$runtime,$layout_exists,$out_dir" >> "$CSV"
  log "DONE  case=$case status=$status rc=$rc wall=$((t1-t0))s legal=$nlegal/$attempts"
}

log "=== 8-case sweep start: cases=${CASES[*]} mode=$MODE (1h each) ==="

for case in "${CASES[@]}"; do
  run_one "$case"
done

log "=== sweep finished. Results in $RESULT_DIR (see results.csv) ==="
