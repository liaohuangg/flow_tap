#!/usr/bin/env bash
# run_ilp_cases.sh — ILP 布局基线的批跑包装。
#
# 用 chipdiffusion 环境 (Gurobi 13.0.1 学术许可证 + CPLEX 22.1.0 都装在这里)。
# 每个 worker 进程 Threads=1, 所以 workers 可以开到核数。
#
# 用法:
#   ./run_ilp_cases.sh                                  # 12 个 case x 5 个配置, quick
#   PROFILE=full ./run_ilp_cases.sh                     # 正式数字 (预算大很多)
#   CASES="hp6_m syn4" SEEDS="1" ./run_ilp_cases.sh     # 收窄范围
#   WORKERS=12 ./run_ilp_cases.sh
#
# 跑完之后评估 (注意必须清缓存, 见 README):
#   rm -f  ../../resultEval/ILP_result/eval_cache.json
#   rm -rf ../../resultEval/ILP_result/eval_out
#   ../../resultEval/eval_layout.py --method ILP --workers 8
set -uo pipefail

PYTHON="${PYTHON:-/root/anaconda3/envs/chipdiffusion/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PROFILE="${PROFILE:-quick}"
WORKERS="${WORKERS:-8}"
SEEDS="${SEEDS:-1}"   # MILP 确定性, 每个 case 只跑一次

case_sel=(--all)
if [ -n "${CASES:-}" ]; then read -ra _c <<< "$CASES"; case_sel=(--cases "${_c[@]}"); fi
read -ra _s <<< "$SEEDS"

log() { echo "[$(date '+%F %T')] $*"; }

log "ILP 基线: profile=$PROFILE workers=$WORKERS seeds=${_s[*]}"
"$PYTHON" run_ilp_cases.py "${case_sel[@]}" --seeds "${_s[@]}" \
  --profile "$PROFILE" --workers "$WORKERS"
rc=$?
log "求解结束 (rc=$rc)"

log "自检 (参考代价可复现 / 几何合法 / 目标一致性)"
"$PYTHON" check_objective.py --quiet
log "自检结束 (rc=$?)"
