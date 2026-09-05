#!/usr/bin/env bash
# 跑 300001..320000 的热仿真 (2w 布局, 单功耗, 128x128)。
# 分 4 批 x 5000, 进度实时写入 thermal_dataset/thermal_sim.log。
#
# 用法:
#   ./run_thermal_sim.sh                # 前台跑 (不推荐, 太久)
#   nohup ./run_thermal_sim.sh &         # 后台跑
#
# 进度查看:  tail -f dataset/thermal_dataset/thermal_sim.log
#           grep -c "j0=ok" dataset/thermal_dataset/thermal_sim.log   # 已完成布局数

set -uo pipefail

PY=/root/anaconda3/envs/chipdiffusion/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN="$SCRIPT_DIR/gen_thermal_dataset.py"
LOG_DIR="$SCRIPT_DIR/dataset/thermal_dataset"
LOG="$LOG_DIR/thermal_sim.log"

START=300001
END=320000
BATCH=5000
WORKERS=16

mkdir -p "$LOG_DIR"

{
  echo "=============================================================="
  echo "热仿真任务启动  $(date '+%F %T')"
  echo "范围 ${START}..${END} (共 $((END - START + 1)) 布局), workers=${WORKERS}, 每批 ${BATCH}"
  echo "=============================================================="
} >> "$LOG"

for s in $(seq "$START" "$BATCH" "$END"); do
  e=$((s + BATCH - 1))
  [ "$e" -gt "$END" ] && e="$END"

  echo "===== BATCH ${s}..${e} START $(date '+%F %T') =====" >> "$LOG"
  "$PY" "$GEN" --start "$s" --end "$e" --workers "$WORKERS" >> "$LOG" 2>&1
  rc=$?
  echo "===== BATCH ${s}..${e} END $(date '+%F %T') rc=${rc} =====" >> "$LOG"

  # 进度快照: 统计已生成的 j=0 温度 CSV 数量
  done_n=$(ls "$LOG_DIR"/thermal_map/system_temp_*_0.csv 2>/dev/null | wc -l)
  echo "[进度] 已完成 ${done_n} / $((END - START + 1))" >> "$LOG"
done

echo "===== 全部完成 $(date '+%F %T') =====" >> "$LOG"
