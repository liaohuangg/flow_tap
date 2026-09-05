#!/usr/bin/env bash
# 生成 chiplet 系统 + 贪心布局,并增量打包到 dataset/placement_dataset/。
#
# 用法:
#   ./gendataset.sh <count>               # 从已打包的最后一个系统之后,再生成并打包 count 个
#   ./gendataset.sh <start> <end>         # 生成并打包 [start, end]
#   ./gendataset.sh <start> <end> --chunk 5000 --workers 8
#
# 所有路径均为相对路径(相对本脚本所在目录),从任意目录调用均可。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CHUNK=5000
WORKERS=8

# ---------- 解析参数 ----------
START=""
END=""
while [ $# -gt 0 ]; do
  case "$1" in
    --chunk)   CHUNK="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | head -20; exit 0 ;;
    *)
      if [ -z "$START" ]; then START="$1"; shift
      elif [ -z "$END" ]; then END="$1"; shift
      else echo "错误:多余参数 $1"; exit 1; fi ;;
  esac
done

if [ -z "$START" ]; then
  echo "用法: ./gendataset.sh <count> 或 ./gendataset.sh <start> <end>"
  exit 1
fi

# ---------- count 模式:自动从已打包末尾续起 ----------
if [ -z "$END" ]; then
  COUNT="$START"
  LAST=$(python3 - <<'PY'
import glob, json, re
files = sorted(
    glob.glob('dataset/placement_dataset/chiplet_dataset_*.json'),
    key=lambda p: int(re.search(r'chiplet_dataset_(\d+)\.json$', p).group(1)),
)
if not files:
    print(0)
else:
    d = json.load(open(files[-1]))
    print(max(int(k.split('_')[1]) for k in d))
PY
)
  START=$(( LAST + 1 ))
  END=$(( LAST + COUNT ))
fi

if [ "$END" -lt "$START" ]; then
  echo "错误:end < start"; exit 1
fi

echo "==> 范围:$START .. $END (共 $(( END - START + 1 )) 个 system)"

# ---------- 1/4 生成 cfg + input_test(临时生成器,逐 index seed,确定性) ----------
echo "==> [1/4] 生成 cfg + input_test"
mkdir -p dataset/input_test

GEN_PY="$SCRIPT_DIR/.gendataset_gen.py"
cat > "$GEN_PY" <<'PY'
import sys, random
from pathlib import Path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from input_preprocess import (
    _random_chiplet_dims, _generate_connected_graph_edges,
    _edges_to_connection_matrix, _format_cfg_like_cpu_dram, cfg_to_json,
)

lo, hi = int(sys.argv[1]), int(sys.argv[2])
config_dir = Path('config')
input_dir = Path('dataset/input_test')
for i in range(lo, hi + 1):
    random.seed(i)
    n = random.randint(3, 20)
    widths, heights = _random_chiplet_dims(n, (3, 30), (3, 30), 0.8, 1.25)
    powers = [float(random.randint(1, 200)) for _ in range(n)]
    edges = _generate_connected_graph_edges(n, extra_edge_prob=0.25)
    conn = _edges_to_connection_matrix(n, edges)
    cfg_text = _format_cfg_like_cpu_dram(n, widths, heights, powers, conn)
    cfg_path = config_dir / f'system_{i}.cfg'
    cfg_path.write_text(cfg_text, encoding='utf-8')
    cfg_to_json(str(cfg_path), str(input_dir))
print(f'[gen] {lo}..{hi} ok', file=sys.stderr)
PY

PER=$(( (END - START + WORKERS) / WORKERS ))
if [ "$PER" -lt 1 ]; then PER=1; fi

pids=()
lo="$START"
while [ "$lo" -le "$END" ]; do
  hi=$(( lo + PER - 1 ))
  if [ "$hi" -gt "$END" ]; then hi="$END"; fi
  python3 "$GEN_PY" "$lo" "$hi" > "/tmp/gendataset_gen_${lo}_${hi}.log" 2>&1 &
  pids+=("$!")
  lo=$(( hi + 1 ))
done
for p in "${pids[@]}"; do wait "$p"; done
rm -f "$GEN_PY"
echo "    cfg + input_test 生成完成"

# ---------- 2/4 贪心布局(并行,确定性,写同一 output 目录安全) ----------
echo "==> [2/4] 贪心布局生成"
mkdir -p dataset/output/placement
pids=()
lo="$START"
while [ "$lo" -le "$END" ]; do
  hi=$(( lo + PER - 1 ))
  if [ "$hi" -gt "$END" ]; then hi="$END"; fi
  (
    cd "$SCRIPT_DIR/dataset"
    python "$SCRIPT_DIR/gen_legal_pla_greedy.py" --start "$lo" --end "$hi" --output-dir output/placement
  ) > "/tmp/gendataset_pla_${lo}_${hi}.log" 2>&1 &
  pids+=("$!")
  lo=$(( hi + 1 ))
done
for p in "${pids[@]}"; do wait "$p"; done
echo "    布局生成完成"

# ---------- 3/4 校验布局覆盖 ----------
echo "==> [3/4] 校验布局覆盖 $START..$END"
python3 - <<PY
import os, re, sys
have = set()
for f in os.listdir("dataset/output/placement"):
    m = re.match(r"system_(\d+)\.json\$", f)
    if m:
        have.add(int(m.group(1)))
missing = [i for i in range($START, $END + 1) if i not in have]
print(f"    placement $START..$END 缺失: {len(missing)}")
if missing:
    print("    缺失示例:", missing[:10])
    sys.exit(1)
PY

# ---------- 4/4 增量打包 ----------
echo "==> [4/4] 打包 -> dataset/placement_dataset/"
python3 "$SCRIPT_DIR/pack_chiplet_dataset.py" --start "$START" --end "$END" \
    --chunk "$CHUNK" --workers "$WORKERS"

echo "ALL DONE ($START..$END)"
