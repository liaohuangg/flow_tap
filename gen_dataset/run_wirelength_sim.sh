#!/usr/bin/env bash
# 跑 340001..360000 的线长求解 (avg 变体, 总线长 + 平均线长)。
# 分 4 批 x 5000, 进度实时写入 wirelength_dataset/wirelength_sim.log。
#
# 用法:
#   ./run_wirelength_sim.sh                # 前台跑
#   nohup ./run_wirelength_sim.sh &         # 后台跑
#
# 进度查看:  tail -f dataset/wirelength_dataset/wirelength_sim.log
#           grep -c "avg=" dataset/wirelength_dataset/wirelength_sim.log   # 已完成布局数

set -uo pipefail

PY=/root/anaconda3/envs/chipdiffusion/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/gen_wirelength_dataset.py"
LOG_DIR="$SCRIPT_DIR/../Dataset/dataset/wirelength_dataset"
LOG="$LOG_DIR/wirelength_sim.log"

START=340001
END=360000
BATCH=5000
WORKERS=28
VARIANT=avg

mkdir -p "$LOG_DIR"

{
  echo "=============================================================="
  echo "线长求解任务启动  $(date '+%F %T')"
  echo "范围 ${START}..${END} (共 $((END - START + 1)) 布局), workers=${WORKERS}, variant=${VARIANT}, 每批 ${BATCH}"
  echo "=============================================================="
} >> "$LOG"

for s in $(seq "$START" "$BATCH" "$END"); do
  e=$((s + BATCH - 1))
  [ "$e" -gt "$END" ] && e="$END"

  echo "===== BATCH ${s}..${e} START $(date '+%F %T') =====" >> "$LOG"
  "$PY" "$GEN" --start "$s" --end "$e" --workers "$WORKERS" --variant "$VARIANT" >> "$LOG" 2>&1
  rc=$?
  echo "===== BATCH ${s}..${e} END $(date '+%F %T') rc=${rc} =====" >> "$LOG"

  # 进度快照: 统计已生成的线长 CSV 数量
  done_n=$(ls "$LOG_DIR"/avg_wirelength/system_avg_wirelength_*.csv 2>/dev/null | wc -l)
  echo "[进度] 已完成 ${done_n} / $((END - START + 1))" >> "$LOG"
done

echo "===== 全部完成 $(date '+%F %T') =====" >> "$LOG"
