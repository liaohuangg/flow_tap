#!/usr/bin/env bash
# run_seed1-5_0wl.sh — cases_0_wl 下 12 个 _bump case × seed ∈ {1,2,3,4,5}, thermal mode (wl_weight=0)。
#
# 与 run_seed1-5.sh 的差异:
#   - 输入用例目录: cases_0_wl  (wl_weight 已全部置 0)
#   - 结果目录:     result_0wl  (单独目录, 避免覆盖 cases/ 的旧结果)
#   - 遍历顺序:     seed 在外层、case 在内层 —— 先用 seed1 跑完所有 case, 再换 seed2, 依此类推。
#   - 不跳过任何 seed (1~5 全部重跑); 不含 syn6。
#
# 规则:
#   - 每个 seed 是一次"单次求解": 只跑该 seed 一个 100-iter 尝试, 结果写到 result_0wl/<case>/seed<seed>/。
#   - 每次求解 wall-clock 上限 1 小时; 没跑出来(无 layout.json)就记录耗时+原因, 跑出来就继续。
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/ATPlan39/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CASES_DIR="$REPO_ROOT/cases_0_wl"
RESULT_DIR="$REPO_ROOT/result_0wl"

MODE=thermal
PARAM_NAME="Thermal-aware.json"
ATP_TIMEOUT=30        # 内部 deadline=30s → 只跑第一个(唯一) seed 一次(100 iter 跑满后自然退出)
OUTER_TIMEOUT=3600    # 单次求解 wall-clock 上限 1 小时
SEEDS=(1 2 3 4 5)

CASES=(acend910_bump cpu-dram_bump hp11_m_bump multigpu_bump syn1_bump syn4_bump xerox8_m_bump \
       Case6_bump Case7_bump Case8_bump Case9_bump Case10_bump)

# 可用环境变量临时收窄范围, 例如:
#   CASES_OVERRIDE="Case6_bump Case7_bump" SEEDS_OVERRIDE="1" ./run_seed1-5_0wl.sh
if [ -n "${CASES_OVERRIDE:-}" ]; then read -ra CASES <<< "$CASES_OVERRIDE"; fi
if [ -n "${SEEDS_OVERRIDE:-}" ]; then read -ra SEEDS <<< "$SEEDS_OVERRIDE"; fi

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$RESULT_DIR"
REPORT="$RESULT_DIR/seeds1-5_report.txt"
: > "$REPORT"

total=0; ok=0; fail=0

for seed in "${SEEDS[@]}"; do
  log "===== seed=$seed 开始 ====="
  for case in "${CASES[@]}"; do
    case_dir="$CASES_DIR/$case"
    out_dir="$RESULT_DIR/$case"
    mkdir -p "$out_dir"
    total=$((total+1))
    seed_dir="$out_dir/seed${seed}"

    # 写 param.json (随机种子; 可选 ITER_OVERRIDE 覆盖迭代次数)
    ITER_OVERRIDE="${ITER_OVERRIDE:-}" "$PYTHON" -c '
import json, os, sys
src, seed, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(open(src)); d["random_seed"] = seed
it = os.environ.get("ITER_OVERRIDE", "")
if it:
    d.setdefault("floorplan_stages", [{}])[0]["iteration"] = int(it)
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

log "=== 结束: 共 $total 个 seed 槽位 (成功 $ok, 失败 $fail) ==="
if [ -s "$REPORT" ]; then
  echo; echo "===== 未跑出来的报告: $REPORT ====="; cat "$REPORT"
else
  echo; echo "===== 全部 seed 都成功跑出 layout.json, 无失败 ====="
fi
