#!/usr/bin/env bash
# 芯粒总线长 GNN 测试集评估
#
# 加载训练好的 checkpoint, 在测试集上跑全量推理, 打印/写入完整指标
# (MAE/RMSE(log+mm) / MAPE / 相对误差中位数·最大·P90·P95·P99 / R^2 / 总和偏差),
# 结果写入 log/test.log (同路径覆盖)。
#
# 数据: placement_dataset_tw/chiplet_dataset_{69..80}.json (60000 个有标签 system),
#       按 split_records(seed=42) 复现训练时的 8:1:1 切分 -> 测试集 6000 个。
#
# 用法:
#   ./test.sh                 # 前台跑, 结果同时打到屏幕并写入 log/test.log
#   nohup ./test.sh &         # 后台跑
#
# 注意: 模型结构参数 (hidden/num_layers/heads/node_features) 必须与训练时一致,
#       否则 load_state_dict 会因 shape 不匹配报错。
set -euo pipefail

cd "$(dirname "$0")"

PY=/root/anaconda3/envs/chipdiffusion/bin/python

# ---------------------------------------------------------------------------
# 评估配置
# ---------------------------------------------------------------------------
BATCH_SIZE=8          # 推理批大小 (与历史 test.log 口径一致)
SEED=42               # 随机种子, 决定 train/val/test 切分 (与训练一致)
NUM_WORKERS=0         # DataLoader 进程数 (WSL 下建议 0)
CKPT="checkpoint/best_wlmodel_total_60k.pt"
NORM_CACHE="checkpoint/normalizer_total_60k.pt"
OUT_LOG="log/test.log"

mkdir -p log

# 传给内嵌 python 用
export WLM_CKPT="$CKPT"
export WLM_NORM="$NORM_CACHE"
export WLM_BS="$BATCH_SIZE"
export WLM_SEED="$SEED"
export WLM_WORKERS="$NUM_WORKERS"

echo "===== eval on TEST (ckpt=$CKPT, batch_size=$BATCH_SIZE, seed=$SEED) ====="
echo "     日志 -> $OUT_LOG"

"$PY" -u - <<'PY' 2>&1 | tee "$OUT_LOG"
import os
import time

import torch

from dataloader import (
    ChipletWirelengthDataset,
    Normalizer,
    collate_fn,
    load_labeled_systems,
    split_records,
)
from wlmodel import WirelengthGNN

CKPT_PATH = os.environ["WLM_CKPT"]
NORM_CACHE = os.environ["WLM_NORM"]
BATCH_SIZE = int(os.environ["WLM_BS"])
SEED = int(os.environ["WLM_SEED"])
NUM_WORKERS = int(os.environ["WLM_WORKERS"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# 1) 加载模型
# ---------------------------------------------------------------------------
t0 = time.time()
model = WirelengthGNN(
    node_dim=18, hidden=256, num_layers=6, heads=4, dropout=0.0,
    use_residual=True, use_global=True,
).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(state)
model.eval()
t_model = time.time() - t0

# ---------------------------------------------------------------------------
# 2) 数据: 复现训练时的切分, 取 test
# ---------------------------------------------------------------------------
t0 = time.time()
norm = torch.load(NORM_CACHE, map_location="cpu", weights_only=False)
normalizer = Normalizer(norm["node_mean"], norm["node_std"],
                        norm["edge_mean"], norm["edge_std"])

records = load_labeled_systems(use_congestion=True)
_, _, test_recs = split_records(records, seed=SEED)
test_ds = ChipletWirelengthDataset(test_recs, normalizer)

loader = torch.utils.data.DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, collate_fn=collate_fn,
)
t_data = time.time() - t0

# ---------------------------------------------------------------------------
# 3) 推理
# ---------------------------------------------------------------------------
preds, trues = [], []
t0 = time.time()
with torch.no_grad():
    for item in loader:
        x = item["x"].to(device)
        ei = item["edge_index"].to(device)
        ea = item["edge_attr"].to(device)
        ew = item["edge_weight"].to(device)
        ng = item["node_geom"].to(device)
        ga = item["global_attr"].to(device)
        cong = item["cong"].to(device)
        batch = item["batch"].to(device)
        y = item["y"].to(device)          # log(total)

        total_pred, _, _ = model(x, ei, ea, ew, ng, batch, ga, cong)
        preds.append(total_pred.double().cpu())
        trues.append(torch.exp(y).double().cpu())
t_infer = time.time() - t0

preds = torch.cat(preds)
trues = torch.cat(trues)
n = preds.numel()

# ---------------------------------------------------------------------------
# 4) 指标
# ---------------------------------------------------------------------------
eps = 1e-8
log_err = torch.log(preds + eps) - torch.log(trues + eps)
abs_err = (preds - trues).abs()
rel = abs_err / (trues + eps)

mae_log = log_err.abs().mean().item()
rmse_log = (log_err ** 2).mean().sqrt().item()
mae_mm = abs_err.mean().item()
rmse_mm = (abs_err ** 2).mean().sqrt().item()
mape = rel.mean().item()
med_rel = rel.median().item()
max_rel = rel.max().item()

q = lambda p: torch.quantile(rel, p).item()
p90, p95, p99 = q(0.90), q(0.95), q(0.99)

ss_res = ((preds - trues) ** 2).sum().item()
ss_tot = ((trues - trues.mean()) ** 2).sum().item()
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

sum_pred = preds.sum().item()
sum_true = trues.sum().item()
sum_dev = (sum_pred - sum_true) / sum_true if sum_true else float("nan")

# ---------------------------------------------------------------------------
# 5) 报告
# ---------------------------------------------------------------------------
bar = "=" * 78
print(bar)
print("wlmodel 测试集评估")
print(bar)
print(f"checkpoint : {os.path.abspath(CKPT_PATH)}")
print(f"batch_size : {BATCH_SIZE}")
print(f"seed       : {SEED} (train/val/test = 8:1:1)")
print(f"device     : {device}")
print(f"测试集规模 : {n} 个 system")
print(f"模型加载耗时 : {t_model:.2f}s   数据加载耗时 : {t_data:.1f}s   推理耗时 : {t_infer:.2f}s")
print()
print("[线长预测指标 (总线长 total wirelength)]")
print()
print(f"  MAE(log)          : {mae_log:.6f}")
print(f"  RMSE(log)         : {rmse_log:.6f}")
print(f"  MAE(mm)           : {mae_mm:.1f}")
print(f"  RMSE(mm)          : {rmse_mm:.1f}")
print(f"  MAPE              : {mape:.4%}")
print(f"  相对误差中位数      : {med_rel:.4%}")
print(f"  最大相对误差        : {max_rel:.4%}")
print(f"  相对误差 P90       : {p90:.4%}")
print(f"  相对误差 P95       : {p95:.4%}")
print(f"  相对误差 P99       : {p99:.4%}")
print(f"  R^2 (总线长)       : {r2:.6f}")
print(f"  总线长预测总和      : {sum_pred:.1f} mm")
print(f"  总线长真实总和      : {sum_true:.1f} mm")
print(f"  总和相对偏差        : {sum_dev:.4%}")
print("RESULT test_mape={:.6f} test_med_rel={:.6f} test_r2={:.6f}".format(mape, med_rel, r2))
PY
