"""芯粒总线长预测模型:纯 PyTorch 手写消息传递 GNN。

设计要点:
  - 图 = 芯粒(节点,带空间/物理特征) + 互联(边,带 wireCount 和 min_clump_manhattan)
  - 消息传递聚合邻居信息(捕捉容量/拥塞),边级读出给出每根线的"实际布线距离"
  - 总线长 = Σ_边 wireCount × d_edge(边求和结构,是问题本身的结构)
  - 平均线长 avg = total / (2 * sum(wireCount)),由总线长解析推出
  - 总线长动态范围大(约 800 ~ 5.5e6),损失在 log 空间计算

用法:
  python wlmodel.py --hidden 256 --num_layers 5 --lr 1e-3 --epochs 150
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import get_dataloaders


# ----------------------------------------------------------------------------
# 消息传递层(纯 PyTorch,不用 torch_scatter)
# ----------------------------------------------------------------------------
class MessagePassingLayer(nn.Module):
    """带边特征的图卷积:聚合每条边的 message 到目标节点。"""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index  # [E]
        msg_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)  # [E, 2N+F]
        msg = self.msg_net(msg_in)                                # [E, D]

        agg = torch.zeros(x.size(0), self.out_dim, device=x.device, dtype=x.dtype)
        agg = agg.index_add_(0, dst, msg)  # 高效 scatter-sum

        out = self.update_net(torch.cat([x, agg], dim=-1))        # [N, D]
        out = self.norm(out + x)                                   # 残差 + LayerNorm
        return out


# ----------------------------------------------------------------------------
# 图池化
# ----------------------------------------------------------------------------
def scatter_mean(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out = out.index_add_(0, index, src)
    cnt = torch.bincount(index, minlength=dim_size).clamp_min(1).unsqueeze(1)
    return out / cnt


def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    return out.index_add_(0, index, src)


# ----------------------------------------------------------------------------
# 主模型
# ----------------------------------------------------------------------------
class WirelengthGNN(nn.Module):
    def __init__(self, node_dim: int = 14, edge_dim: int = 2, global_dim: int = 4,
                 hidden: int = 128, num_layers: int = 4, dropout: float = 0.0,
                 use_residual: bool = True, use_global: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.use_global = use_global
        in_dim = node_dim + (global_dim if use_global else 0)
        self.node_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList(
            [MessagePassingLayer(hidden, hidden, hidden, dropout) for _ in range(num_layers)]
        )

        # 边级读出:每根线的"实际布线距离" d_edge > 0
        self.edge_score = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # 节点级读出:每个 chiplet 4 个侧边(左/上/右/下)的 bump 流量 (原始计数, >0)
        self.node_flow_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def _min_clump_manhattan(self, node_geom, edge_index):
        """从原始几何 (x,y,w,h,hubump) 在线计算每条边两端 4×4 clump 的最小曼哈顿距离。

        与 dataloader._clumps 的 4 个 clump(左/上/右/下)定义一致, 且对 x/y 可微
        (min 把次梯度传到 argmin 的 clump 对)。这是 flow-matching 引导项位置可微的关键。
        """
        src, dst = edge_index
        x = node_geom[:, 0]
        y = node_geom[:, 1]
        w = node_geom[:, 2]
        h = node_geom[:, 3]
        hu = node_geom[:, 4]
        cx = x + w / 2.0
        cy = y + h / 2.0
        clumps = torch.stack([
            torch.stack([x - hu / 2.0, cy], dim=-1),          # 左
            torch.stack([cx, y + h + hu / 2.0], dim=-1),      # 上
            torch.stack([x + w + hu / 2.0, cy], dim=-1),      # 右
            torch.stack([cx, y - hu / 2.0], dim=-1),          # 下
        ], dim=1)  # [N, 4, 2]
        cs = clumps[src]  # [E, 4, 2]
        cd = clumps[dst]  # [E, 4, 2]
        manhattan = (cs.unsqueeze(2) - cd.unsqueeze(1)).abs().sum(-1)  # [E, 4, 4]
        return manhattan.min(dim=-1).values.min(dim=-1).values  # [E]

    def forward(self, x, edge_index, edge_attr, edge_weight, node_geom, batch, global_attr):
        num_graphs = int(batch.max().item()) + 1

        # 图级标量拼进每个节点(全局拥塞尺度),再做节点编码
        if self.use_global:
            x = torch.cat([x, global_attr[batch]], dim=-1)
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, e)

        src, dst = edge_index
        correction = F.softplus(
            self.edge_score(torch.cat([e, h[src], h[dst]], dim=-1))).squeeze(-1)
        # 残差读出: d_edge = dmin(无容量下界, 从几何在线算, 位置可微) + 容量膨胀修正(≥0)
        dmin = self._min_clump_manhattan(node_geom, edge_index)
        d_edge = dmin + correction if self.use_residual else correction
        weighted = edge_weight * d_edge  # [E]

        edge_batch = batch[src]
        total = scatter_sum(weighted.unsqueeze(1), edge_batch, num_graphs).squeeze(-1)  # [num_graphs]

        # 节点级:每 chiplet 每侧边 bump 流量 (原始计数)
        flow_pred = F.softplus(self.node_flow_head(h))  # [N, 4]
        return total, d_edge, flow_pred

    def predict_wirelength(self, total, wcount):
        """由总线长推出平均线长。"""
        avg = total / (2.0 * wcount.clamp_min(1e-6))
        return total, avg


# ----------------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    total_log_err = 0.0
    total_abs_err = 0.0
    total_rel_err = 0.0
    n = 0
    preds, trues = [], []
    with torch.no_grad():
        for item in loader:
            x = item["x"].to(device)
            ei = item["edge_index"].to(device)
            ea = item["edge_attr"].to(device)
            ew = item["edge_weight"].to(device)
            ng = item["node_geom"].to(device)
            ga = item["global_attr"].to(device)
            batch = item["batch"].to(device)
            y = item["y"].to(device)  # log(total)

            total_pred, _, _ = model(x, ei, ea, ew, ng, batch, ga)
            log_pred = torch.log(total_pred + 1e-8)
            total_true = torch.exp(y)

            total_log_err += (log_pred - y).abs().sum().item()
            total_abs_err += (total_pred - total_true).abs().sum().item()
            total_rel_err += ((total_pred - total_true).abs() / total_true).sum().item()
            n += y.size(0)
            preds.append(total_pred.cpu())
            trues.append(total_true.cpu())

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    med_rel = (preds - trues).abs().div(trues).median().item()
    return {
        "mae_log": total_log_err / n,
        "mae_mm": total_abs_err / n,
        "mape": total_rel_err / n,
        "med_rel": med_rel,
    }


def train(config: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    use_congestion = config.get("node_features", 14) == 14
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        seed=config["seed"], use_congestion=use_congestion)

    model = WirelengthGNN(node_dim=config.get("node_features", 14),
                          hidden=config["hidden"], num_layers=config["num_layers"],
                          dropout=config.get("dropout", 0.0),
                          use_residual=not config.get("no_residual", False),
                          use_global=not config.get("no_global", False)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])

    best_val = float("inf")
    best_path = config.get("save", os.path.join(os.path.dirname(__file__), "checkpoint", "best_wlmodel.pt"))
    os.makedirs(os.path.dirname(os.path.abspath(best_path)), exist_ok=True)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        nb = 0
        for item in train_loader:
            x = item["x"].to(device)
            ei = item["edge_index"].to(device)
            ea = item["edge_attr"].to(device)
            ew = item["edge_weight"].to(device)
            ng = item["node_geom"].to(device)
            ga = item["global_attr"].to(device)
            batch = item["batch"].to(device)
            y = item["y"].to(device)  # log(total)
            edge_label = item["edge_label"]

            total_pred, d_edge, flow_pred = model(x, ei, ea, ew, ng, batch, ga)
            loss = F.mse_loss(torch.log(total_pred + 1e-8), y)
            if edge_label is not None:
                el = edge_label.to(device)
                loss_edge = F.mse_loss(torch.log(d_edge + 1e-8), torch.log(el + 1e-8))
                loss = loss + config.get("edge_loss_weight", 1.0) * loss_edge
            node_flow = item["node_flow"]
            if node_flow is not None:
                nf = node_flow.to(device)
                loss_side = F.mse_loss(torch.log1p(flow_pred), torch.log1p(nf))
                loss = loss + config.get("side_loss_weight", 1.0) * loss_side

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            nb += 1
        sched.step()

        val_metrics = evaluate(model, val_loader, device)
        train_metrics = evaluate(model, train_loader, device)

        if val_metrics["mae_log"] < best_val:
            best_val = val_metrics["mae_log"]
            torch.save(model.state_dict(), best_path)

        if epoch % config["log_every"] == 0 or epoch == 1:
            print(f"[epoch {epoch:3d}] loss={epoch_loss / nb:.4f}  "
                  f"train(mae_log={train_metrics['mae_log']:.4f}, med_rel={train_metrics['med_rel']:.3%})  "
                  f"val(mae_log={val_metrics['mae_log']:.4f}, med_rel={val_metrics['med_rel']:.3%})  "
                  f"({time.time() - t0:.1f}s)")

    model.load_state_dict(torch.load(best_path))
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== 测试集(总线长) ===")
    print(f"  MAE(log)          : {test_metrics['mae_log']:.4f}")
    print(f"  MAE (mm)          : {test_metrics['mae_mm']:.1f}")
    print(f"  MAPE              : {test_metrics['mape']:.3%}")
    print(f"  相对误差中位数     : {test_metrics['med_rel']:.3%}")
    print(f"  最优模型已保存: {best_path}")
    print(f"RESULT best_val_mae_log={best_val:.4f} test_mae_log={test_metrics['mae_log']:.4f} "
          f"test_mape={test_metrics['mape']:.4f} test_med_rel={test_metrics['med_rel']:.4f}")
    return model, test_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Chiplet wirelength GNN training")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--node_features", type=int, default=14, choices=[7, 14],
                   help="节点特征维数: 14=含拥塞特征(需求/容量/每侧边容量), 7=不含")
    p.add_argument("--no_residual", action="store_true", help="关闭残差读出(d_edge 退化为纯 softplus)")
    p.add_argument("--no_global", action="store_true", help="关闭 global_attr 注入")
    p.add_argument("--edge_loss_weight", type=float, default=1.0,
                   help="每边监督损失权重 λ(0 = 仅总线长监督, 回到 baseline)")
    p.add_argument("--side_loss_weight", type=float, default=1.0,
                   help="每侧边流量监督损失权重 μ(0 = 关闭 side flow 监督)")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--save", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    if not config["save"]:
        config["save"] = os.path.join(os.path.dirname(__file__), "checkpoint", "best_wlmodel.pt")
    print("训练参数:", config)
    train(config)
