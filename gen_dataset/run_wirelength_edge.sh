#!/usr/bin/env bash
# 用 CPLEX 重跑 340001..380000 的线长求解 (avg 变体),
# 除 total/avg 外, 额外输出每边路由距离标签 edge_wirelength/。
#
# workers=28: 已停热仿真, 全部 28 核给线长仿真。
#
# 用法:
#   nohup ./run_wirelength_edge.sh &          # 后台跑
# 进度查看:
#   tail -f ../Dataset/dataset/wirelength_dataset/wirelength_edge.log
#   grep -c "total=" ../Dataset/dataset/wirelength_dataset/wirelength_edge.log

set -uo pipefail

PY=/root/anaconda3/envs/chipdiffusion/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/gen_wirelength_dataset.py"
LOG_DIR="$SCRIPT_DIR/../Dataset/dataset/wirelength_dataset"
LOG="$LOG_DIR/wirelength_edge.log"
SUMMARY="$LOG_DIR/summary_340001_380000.csv"

START=340001
END=380000
BATCH=5000
WORKERS=28
VARIANT=avg

mkdir -p "$LOG_DIR"

{
  echo "=============================================================="
  echo "CPLEX 线长 + 每边距离标签任务启动  $(date '+%F %T')"
  echo "范围 ${START}..${END} (共 $((END - START + 1)) 布局), workers=${WORKERS}, variant=${VARIANT}, 每批 ${BATCH}"
  echo "=============================================================="
} >> "$LOG"

for s in $(seq "$START" "$BATCH" "$END"); do
  e=$((s + BATCH - 1))
  [ "$e" -gt "$END" ] && e="$END"

  echo "===== BATCH ${s}..${e} START $(date '+%F %T') =====" >> "$LOG"
  "$PY" "$GEN" --start "$s" --end "$e" --workers "$WORKERS" --variant "$VARIANT" \
       --summary "$SUMMARY" >> "$LOG" 2>&1
  rc=$?
  echo "===== BATCH ${s}..${e} END $(date '+%F %T') rc=${rc} =====" >> "$LOG"

  done_edge=$(ls "$LOG_DIR"/edge_wirelength/system_edge_wirelength_*.json 2>/dev/null | wc -l)
  echo "[进度] 每边标签已完成 ${done_edge} / $((END - START + 1))" >> "$LOG"
done

echo "===== 全部完成 $(date '+%F %T') =====" >> "$LOG"
