#!/usr/bin/env bash
# 芯粒总线长 GNN 训练脚本
# 用法: bash train.sh
# 数据: placement_dataset_tw/chiplet_dataset_{69..72}.json (20000 个有标签 system)
# 模型: 消息传递 GNN,边求和读出 (total = Σ wireCount * d_edge)
set -euo pipefail

# ----------------------------------------------------------------------------
# 训练超参数(可在这里调整)
# ----------------------------------------------------------------------------
BATCH_SIZE=128        # 批大小
NUM_WORKERS=0         # DataLoader 进程数(WSL 下建议 0,避免多进程开销)
SEED=42               # 随机种子(决定 8:1:1 切分)
HIDDEN=256            # 隐层维数
NUM_LAYERS=6          # 消息传递层数
DROPOUT=0.0           # dropout 比例
LR=1e-3               # 学习率
WD=1e-5               # weight decay
EPOCHS=150            # 训练轮数
LOG_EVERY=5           # 每多少 epoch 打印一次
SAVE="checkpoint/best_wlmodel.pt"   # 最优模型保存路径

# ----------------------------------------------------------------------------
# 环境
# ----------------------------------------------------------------------------
cd "$(dirname "$0")"

# 激活 chipdiffusion 环境(如果当前不在该环境下)
if [[ "${CONDA_DEFAULT_ENV:-}" != "chipdiffusion" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate chipdiffusion 2>/dev/null || true
fi

# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
echo "=== 训练参数 ==="
echo "  batch_size=$BATCH_SIZE hidden=$HIDDEN num_layers=$NUM_LAYERS"
echo "  lr=$LR wd=$WD epochs=$EPOCHS seed=$SEED save=$SAVE"

python3 wlmodel.py \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --seed "$SEED" \
  --hidden "$HIDDEN" \
  --num_layers "$NUM_LAYERS" \
  --dropout "$DROPOUT" \
  --lr "$LR" \
  --wd "$WD" \
  --epochs "$EPOCHS" \
  --log_every "$LOG_EVERY" \
  --save "$SAVE"
