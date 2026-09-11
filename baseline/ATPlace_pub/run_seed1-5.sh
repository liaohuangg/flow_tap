#!/usr/bin/env bash
# run_seed1-5.sh — 13 个 _bump case × seed ∈ {1,2,3,4,5}, thermal mode.
#
# 规则:
#   - seed1: 若 result/<case>/seed1/layout.json 已存在(已成功跑出 seed1)则跳过, 否则重跑。
#   - 每个 seed 是一次"单次求解": 只跑该 seed 一个 100-iter 尝试, 结果写到 result/<case>/seed<seed>/。
#   - 每次求解 wall-clock 上限 1 小时; 没跑出来(无 layout.json)就记录耗时+原因, 跑出来就继续下一个 seed。
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO_ROOT/cases"
RESULT_DIR="$REPO_ROOT/result"

MODE=thermal
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT=30        # 内部 deadline=30s → 只跑第一个(唯一) seed 一次(100 iter 跑满后自然退出)
OUTER_TIMEOUT=3600    # 单次求解 wall-clock 上限 1 小时
SEEDS=(1 2 3 4 5)

CASES=(acend910_bump cpu-dram_bump hp11_m_bump multigpu_bump syn1_bump syn4_bump syn6_bump xerox8_m_bump \
       Case6_bump Case7_bump Case8_bump Case9_bump Case10_bump)

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/seeds1-5_report.txt"
: > "$REPORT"

total=0; ok=0; fail=0; skip=0

for case in "${CASES[@]}"; do
  case_dir="$CASES_DIR/$case"
  out_dir="$RESULT_DIR/$case"
  mkdir -p "$out_dir"
  for seed in "${SEEDS[@]}"; do
    total=$((total+1))
    seed_dir="$out_dir/seed${seed}"

    # seed1 已成功跑出 layout.json -> 跳过
    if [ "$seed" = "1" ] && [ -f "$seed_dir/layout.json" ]; then
      log "SKIP  case=$case seed=1 (已有 layout.json)"
      skip=$((skip+1))
      continue
    fi

    # 写 param.json (随机种子)
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
    elap_fmt=$(printf '%02d:%02d:%02d' $((elapsed/3600)) $(( (elapsed%3600)/60 )) $((elapsed%60)))

    if [ -f "$seed_dir/layout.json" ]; then
      hpwl=$(grep -o '"hpwl": [0-9.e+-]*' "$seed_dir/layout.json" | head -1)
      ok=$((ok+1))
      log "DONE  case=$case seed=$seed ${elap_fmt} ($hpwl)"
    else
      fail=$((fail+1))
      reason=$(sed -n 's/.*\[placeflow\] error: //p' "$out_dir/run_seed${seed}.log" | tail -1)
      if [ -z "$reason" ]; then
        case "$rc" in
          124|137) reason="超时(1h 上限, rc=$rc)" ;;
          *)       reason="无 layout.json (rc=$rc)" ;;
        esac
      fi
      log "FAIL  case=$case seed=$seed ${elap_fmt} rc=$rc :: $reason"
      echo "FAIL  case=$case  seed=$seed  耗时=${elap_fmt}  rc=$rc  原因=$reason" >> "$REPORT"
    fi
  done
done

log "=== 结束: 共 $total 个 seed 槽位 (跳过 $skip, 成功 $ok, 失败 $fail) ==="
if [ -s "$REPORT" ]; then
  echo; echo "===== 未跑出来的报告: $REPORT ====="; cat "$REPORT"
else
  echo; echo "===== 全部 seed 都成功跑出 layout.json, 无失败 ====="
fi
