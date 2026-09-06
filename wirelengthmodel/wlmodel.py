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
# 图注意力层(GAT,纯 PyTorch,不用 torch_scatter)
# ----------------------------------------------------------------------------
class GATLayer(nn.Module):
    """带边特征的多头图注意力层。

    注意力 = 软选择"该看哪个邻居/哪条边"(soft-argmax),天然适合"走哪个侧、哪个邻居拥塞"
    这类离散选择;相比 sum 聚合, 更能建模超载在邻居间的级联传播。
    """

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int,
                 heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert out_dim % heads == 0, "out_dim 必须能被 heads 整除"
        self.heads = heads
        self.head_dim = out_dim // heads
        self.out_dim = out_dim

        self.q = nn.Linear(node_dim, out_dim, bias=False)      # 目标侧 query
        self.k = nn.Linear(node_dim, out_dim, bias=False)      # 源侧 key
        self.v = nn.Linear(node_dim, out_dim, bias=False)      # 源侧 value
        self.e_att = nn.Linear(edge_dim, heads, bias=False)    # 边特征 → 每头注意力 logit
        self.e_msg = nn.Linear(edge_dim, out_dim, bias=False)  # 边特征 → message 增量
        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.update = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        H, d = self.heads, self.head_dim

        q = self.q(x).view(-1, H, d)            # [N, H, d]
        k = self.k(x).view(-1, H, d)
        v = self.v(x).view(-1, H, d)

        # 注意力 logits: 缩放点积 + 边特征贡献
        a = (q[dst] * k[src]).sum(-1) / (d ** 0.5)   # [E, H]
        a = a + self.e_att(edge_attr)                 # [E, H]
        a = self.leaky(a)

        alpha = scatter_softmax(a, dst, x.size(0))    # [E, H] 按目标节点 softmax

        # message = value + 边特征增量,按注意力加权
        msg = v[src] + self.e_msg(edge_attr).view(-1, H, d)   # [E, H, d]
        msg = self.dropout(alpha.unsqueeze(-1) * msg)         # [E, H, d]

        agg = torch.zeros(x.size(0), H, d, device=x.device, dtype=x.dtype)
        agg = agg.index_add_(0, dst, msg).view(x.size(0), self.out_dim)  # [N, out_dim]

        out = self.update(torch.cat([x, agg], dim=-1))        # [N, out_dim]
        return self.norm(out + x)                              # 残差 + LayerNorm


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


def scatter_softmax(logits, index, dim_size):
    """按目标节点做 softmax(logits)。logits: [E, H], index: [E] 目标节点 id。"""
    H = logits.size(1)
    idx = index.unsqueeze(1).expand(-1, H)
    maxes = torch.full((dim_size, H), -1e9, device=logits.device, dtype=logits.dtype)
    maxes = maxes.scatter_reduce(0, idx, logits, reduce="amax")
    exp = torch.exp(logits - maxes[index])
    sums = torch.zeros(dim_size, H, device=logits.device, dtype=logits.dtype)
    sums = sums.scatter_add(0, idx, exp)
    return exp / (sums[index] + 1e-8)


# ----------------------------------------------------------------------------
# 主模型
# ----------------------------------------------------------------------------
class WirelengthGNN(nn.Module):
    def __init__(self, node_dim: int = 18, edge_dim: int = 2, global_dim: int = 4,
                 cong_dim: int = 8, hidden: int = 256, num_layers: int = 6,
                 heads: int = 4, dropout: float = 0.1,
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
        # 每侧拥塞原始信号(每侧容量+偏好需求)单独编码, 供 correction 直接使用
        self.cong_encoder = nn.Sequential(
            nn.Linear(cong_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.convs = nn.ModuleList(
            [GATLayer(hidden, hidden, hidden, heads, dropout) for _ in range(num_layers)]
        )

        # 边级读出: correction = f(e, h_src, h_dst, cong_src, cong_dst)
        self.edge_score = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
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

    def forward(self, x, edge_index, edge_attr, edge_weight, node_geom, batch, global_attr, cong):
        num_graphs = int(batch.max().item()) + 1

        # 图级标量拼进每个节点(全局拥塞尺度),再做节点编码
        if self.use_global:
            x = torch.cat([x, global_attr[batch]], dim=-1)
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, e)

        src, dst = edge_index
        # 每侧拥塞信号(容量+偏好需求)单独编码, 让 correction 直接看到"两端哪侧超载"
        c = self.cong_encoder(cong)  # [N, hidden]
        correction = F.softplus(
            self.edge_score(torch.cat([e, h[src], h[dst], c[src], c[dst]], dim=-1))).squeeze(-1)
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
            cong = item["cong"].to(device)
            batch = item["batch"].to(device)
            y = item["y"].to(device)  # log(total)

            total_pred, _, _ = model(x, ei, ea, ew, ng, batch, ga, cong)
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

    use_congestion = config.get("node_features", 18) == 18
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        batch_size=config["batch_size"], num_workers=config["num_workers"],
        seed=config["seed"], use_congestion=use_congestion)

    model = WirelengthGNN(node_dim=config.get("node_features", 18),
                          hidden=config["hidden"], num_layers=config["num_layers"],
                          heads=config.get("heads", 4),
                          dropout=config.get("dropout", 0.1),
                          use_residual=not config.get("no_residual", False),
                          use_global=not config.get("no_global", False)).to(device)
    if config.get("resume"):
        resume_path = os.path.abspath(config["resume"])
        model.load_state_dict(torch.load(resume_path, map_location=device))
        print(f"warm start: 已加载 {resume_path}")
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config["epochs"])

    best_val = float("inf")
    best_path = os.path.abspath(config.get("save", os.path.join(os.path.dirname(__file__), "checkpoint", "best_wlmodel.pt")))
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    snapshot_every = config.get("snapshot_every", 0)

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
            cong = item["cong"].to(device)
            batch = item["batch"].to(device)
            y = item["y"].to(device)  # log(total)
            edge_label = item["edge_label"]

            total_pred, d_edge, flow_pred = model(x, ei, ea, ew, ng, batch, ga, cong)
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

        if snapshot_every and epoch % snapshot_every == 0:
            snap_path = best_path.replace(".pt", f"_epoch{epoch}.pt")
            torch.save(model.state_dict(), snap_path)

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
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4, help="GAT 注意力头数")
    p.add_argument("--node_features", type=int, default=18, choices=[7, 18],
                   help="节点特征维数: 18=含拥塞特征(需求/容量/每侧容量/每侧偏好需求), 7=不含")
    p.add_argument("--no_residual", action="store_true", help="关闭残差读出(d_edge 退化为纯 softplus)")
    p.add_argument("--no_global", action="store_true", help="关闭 global_attr 注入")
    p.add_argument("--edge_loss_weight", type=float, default=1.0,
                   help="每边监督损失权重 λ(0 = 仅总线长监督, 回到 baseline)")
    p.add_argument("--side_loss_weight", type=float, default=1.0,
                   help="每侧边流量监督损失权重 μ(0 = 关闭 side flow 监督)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--resume", type=str, default="", help="warm start: 加载 checkpoint 权重后继续训练")
    p.add_argument("--snapshot_every", type=int, default=50, help="每 N 个 epoch 保存一次模型快照(0=关闭)")
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
