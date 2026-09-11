#!/usr/bin/env bash
# run_case10_65.sh — 只跑 Case10_bump, 65 迭代, seed ∈ {1,2,3,4,5}.
# 每个 seed 一次单次求解(只跑该 seed 一个 65-iter 尝试), 结果到 result/Case10_bump/seed<seed>/.
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO_ROOT/cases"
RESULT_DIR="$REPO_ROOT/result"

CASE=Case10_bump
MODE=thermal
PARAM_NAME="Thermal-aware.json"
ITER=65
ATP_TIMEOUT=30        # 内部 deadline=30s → 只跑第一个(唯一) seed 一次
OUTER_TIMEOUT=5400    # 1.5h 上限(65 iter ≈ 58min + 合法化, 留余量, 避免 1h 临界被掐)
SEEDS=(1 2 3 4 5)

log() { echo "[$(date '+%F %T')] $*"; }

case_dir="$CASES_DIR/$CASE"
out_dir="$RESULT_DIR/$CASE"
mkdir -p "$out_dir"
REPORT="$RESULT_DIR/case10_65_report.txt"
: > "$REPORT"

total=0; ok=0; fail=0

for seed in "${SEEDS[@]}"; do
  total=$((total+1))
  seed_dir="$out_dir/seed${seed}"

  "$PYTHON" -c '
import json, sys
src, seed, it, dst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = json.load(open(src))
d["random_seed"] = seed
d["floorplan_stages"][0]["iteration"] = it
json.dump(d, open(dst, "w"), indent=2, ensure_ascii=False)
' "$case_dir/$PARAM_NAME" "$seed" "$ITER" "$out_dir/param.json"

  t0=$(date +%s)
  log "START $CASE seed=$seed iter=$ITER"
  ATPLACE_TIMEOUT="$ATP_TIMEOUT" timeout -k 30 "$OUTER_TIMEOUT" \
    "$PYTHON" "$REPO_ROOT/reproduce.py" --case "$CASE" --mode "$MODE" \
    --case-dir "$case_dir" --param-file "$out_dir/param.json" --out-dir "$out_dir" \
    > "$out_dir/run_seed${seed}.log" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  elap_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))

  if [ -f "$seed_dir/layout.json" ]; then
    hpwl=$(grep -o '"hpwl": [0-9.e+-]*' "$seed_dir/layout.json" | head -1)
    ok=$((ok+1))
    log "DONE  $CASE seed=$seed ${elap_fmt} ($hpwl)"
  else
    fail=$((fail+1))
    reason=$(sed -n 's/.*\[placeflow\] error: //p' "$out_dir/run_seed${seed}.log" | tail -1)
    if [ -z "$reason" ]; then
      case "$rc" in
        124|137) reason="超时(1.5h 上限, rc=$rc)" ;;
        *)       reason="无 layout.json (rc=$rc)" ;;
      esac
    fi
    log "FAIL  $CASE seed=$seed ${elap_fmt} rc=$rc :: $reason"
    echo "FAIL  seed=$seed  耗时=${elap_fmt}  rc=$rc  原因=$reason" >> "$REPORT"
  fi
done

log "=== 结束: 共 $total 个 (成功 $ok, 失败 $fail) ==="
if [ -s "$REPORT" ]; then
  echo; echo "===== 未跑出来的报告: $REPORT ====="; cat "$REPORT"
else
  echo; echo "===== 全部 seed 都成功跑出 layout.json ====="
fi
