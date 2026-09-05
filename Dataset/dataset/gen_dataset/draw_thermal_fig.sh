#!/usr/bin/env bash
# 批量绘制热图 / 功耗图, 用于检验 HotSpot 热仿真结果。
#
# 用法:
#   ./draw_thermal_fig.sh <start> <end> [j]
#   ./draw_thermal_fig.sh 300001 300012          # j=0 和 j=1 都画
#   ./draw_thermal_fig.sh 300001 300012 0        # 只画 j=0
#
# 输出到 dataset/thermal_dataset/figures/
#   system_<i>_<j>_thermal.png   128×128 芯片层温度(℃) + chiplet 边框
#   system_<i>_<j>_power.png     128×128 功耗图(W) + chiplet 边框
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$SCRIPT_DIR/dataset/thermal_dataset"
FIG="$DATA/figures"
PYTHON="${PYTHON:-/root/anaconda3/envs/chipdiffusion/bin/python}"
DRAW="$SCRIPT_DIR/draw_thermal_map.py"

START="${1:?用法: $0 <start> <end> [j]}"
END="${2:?}"
J="${3:-all}"

mkdir -p "$FIG"

for i in $(seq "$START" "$END"); do
  cfg="$DATA/config/system_${i}_config"
  flp="$cfg/system_${i}L4_ChipLayer.flp"
  [ -f "$flp" ] || { echo "[draw] 缺 $flp, 跳过 system_${i}"; continue; }

  for j in 0 1; do
    [ "$J" = "all" ] || [ "$J" = "$j" ] || continue

    t_csv="$DATA/thermal_map/system_temp_${i}_${j}.csv"
    if [ -f "$t_csv" ]; then
      "$PYTHON" "$DRAW" --input "$t_csv" --flp "$flp" --unit C \
        --out "$FIG/system_${i}_${j}_thermal.png" --title "system_${i} j=${j} thermal (°C)"
    fi

    p_csv="$DATA/power_map/system_power_${i}_${j}.csv"
    if [ -f "$p_csv" ]; then
      "$PYTHON" "$DRAW" --input "$p_csv" --flp "$flp" --unit W \
        --out "$FIG/system_${i}_${j}_power.png" --title "system_${i} j=${j} power (W)"
    fi
  done
done

echo "[draw] 完成 -> $FIG"
