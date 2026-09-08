"""
GNN + HRNet 组合热预测模型 (gnnhrnet.py)

动机
----
纯 GNN + U-Net 已把热图 RMSE 压到 ~0.55°C (8w 数据), 但 U-Net 的瓶颈在 8×8 会丢失
尖锐热点的细节。HRNet 的核心优势是「全程保留高分辨率 64×64 分支 + 多尺度交换单元」,
既能有全局感受野 (16×16 分支), 又不抹平峰值细节。

本模型把两者结合:

    GNN  (GATv2)   → 节点嵌入 → 全局图向量 global_cond (chiplet 级物理/交互信息)
    HRNet 场头      → 多尺度 64/32/16 分支, 用 global_cond 做 FiLM 调制 (取代 HRNet
                      原来只喂 total_power 单标量的弱条件)
    peak head      → 每节点峰值回归 + hard max 池化 (直接打热点短板)

输入:
  - 图: node 特征 (功率/尺寸/位置/hubump 环宽与面积), 全连接边 (中心距离)
  - 场栅格: [功率密度, chiplet 掩码, hubump 环掩码] 3 通道 64×64

输出: heatmap [B,1,64,64], peak [B,1], node_peak [N,1]
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv
from scipy.ndimage import zoom as _scipy_zoom

import dataLoader as dl

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJ, "Dataset/dataset/thermal_dataset_64")
CFG_ROOT = os.path.join(DATA_ROOT, "config")

# 温度归一化 (与 HRNet 全量统计一致)
TEMP_MIN = 45.22
TEMP_MAX = 276.12

# 特征归一化常数
POWER_SCALE = 200.0          # 单个 chiplet 功率 (W)
HUBUMP_SCALE = 2.0           # hubump 环宽 (mm)
POWER_GRID_SCALE = 10.0      # 功率密度 (W/mm²)


# --------------------------------------------------------------------------- #
# GNN 编码器 + 峰值头
# --------------------------------------------------------------------------- #
class GNNEncoder(nn.Module):
    """GATv2 堆叠编码器,每层带残差 + LayerNorm。"""

    def __init__(self, in_dim, hidden=128, heads=4, num_layers=3, edge_dim=1, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "gat": GATv2Conv(hidden, hidden, heads=heads, edge_dim=edge_dim,
                                 concat=False, dropout=dropout),
                "norm": nn.LayerNorm(hidden),
            }))
        self.out_norm = nn.LayerNorm(hidden)

    def forward(self, x, edge_index, edge_attr=None):
        h = F.relu(self.in_proj(x))
        for blk in self.layers:
            h = blk["norm"](h + F.relu(blk["gat"](h, edge_index, edge_attr)))
        return self.out_norm(h)


def _global_pool(node_emb, batch, num_graphs):
    """mean+max 池化得到每个图的全局向量。"""
    B, H = num_graphs, node_emb.size(1)
    mean = torch.zeros(B, H, device=node_emb.device, dtype=node_emb.dtype)
    mean.index_add_(0, batch, node_emb)
    counts = torch.zeros(B, 1, device=node_emb.device, dtype=node_emb.dtype)
    counts.index_add_(0, batch, torch.ones_like(node_emb[:, :1]))
    mean = mean / counts.clamp_min(1.0)
    maxv = torch.full((B, H), float("-inf"), device=node_emb.device, dtype=node_emb.dtype)
    maxv.index_reduce_(0, batch, node_emb, "amax", include_self=False)
    return torch.cat([mean, maxv], dim=-1)  # [B, 2H]


class PeakHead(nn.Module):
    """每节点峰值回归 + hard max 池化得全局峰值。"""

    def __init__(self, hidden, pooling="hard"):
        super().__init__()
        self.node_peak = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # 峰值位置回归分支: 每个 chiplet 体内热点坐标 (x,y) ∈ [-1,1]
        self.node_pos = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.pooling = pooling

    def forward(self, node_emb, batch, num_graphs):
        node_peak = self.node_peak(node_emb).squeeze(-1)  # [N]
        node_pos = self.node_pos(node_emb)  # [N, 2]
        B = num_graphs
        maxv = torch.full((B,), float("-inf"), device=node_emb.device, dtype=node_peak.dtype)
        maxv.index_reduce_(0, batch, node_peak, "amax", include_self=False)
        if self.pooling == "hard":
            global_peak = maxv
        elif self.pooling == "lse":
            tau = 0.1
            e = ((node_peak - maxv[batch]) / tau).exp()
            s = torch.zeros(B, device=node_emb.device, dtype=node_peak.dtype)
            s.index_add_(0, batch, e)
            global_peak = maxv + tau * (s + 1e-8).log()
        else:  # softmean
            tau = 0.1
            w = ((node_peak - maxv[batch]) / tau).exp()
            denom = torch.zeros(B, device=node_emb.device, dtype=node_peak.dtype)
            denom.index_add_(0, batch, w)
            global_peak = torch.zeros(B, device=node_emb.device, dtype=node_peak.dtype)
            global_peak.index_add_(0, batch, (w / (denom + 1e-8)) * node_peak)
        return global_peak.view(-1, 1), node_peak.view(-1, 1), node_pos


class FiLM(nn.Module):
    """对特征图做 FiLM 调制 (用条件向量生成 per-channel gamma/beta)。"""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, channels)
        self.beta = nn.Linear(cond_dim, channels)

    def forward(self, feat, cond):
        g = self.gamma(cond).view(-1, feat.size(1), 1, 1)
        b = self.beta(cond).view(-1, feat.size(1), 1, 1)
        return feat * (1.0 + g) + b


# --------------------------------------------------------------------------- #
# HRNet 场头 (多尺度 64/32/16 分支 + 交换单元)
# --------------------------------------------------------------------------- #
def _group_norm(ch):
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.GroupNorm(num_groups=1, num_channels=ch)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class LiteInvertedResidual(nn.Module):
    """MobileNetV3 风格轻量倒残差块 (SiLU + GroupNorm)。"""

    def __init__(self, in_ch, out_ch, stride, expand_ratio=2):
        super().__init__()
        assert stride in (1, 2)
        mid = int(in_ch * expand_ratio)
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if mid != in_ch:
            layers.append(ConvGNAct(in_ch, mid, k=1, s=1, p=0))
        layers.append(nn.Conv2d(mid, mid, kernel_size=3, stride=stride, padding=1, groups=mid, bias=False))
        layers.append(_group_norm(mid))
        layers.append(nn.SiLU(inplace=True))
        layers.append(nn.Conv2d(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(_group_norm(out_ch))
        self.net = nn.Sequential(*layers)
        self.out_act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.net(x)
        if self.use_res:
            y = y + x
        return self.out_act(y)


class HRBranch(nn.Module):
    """单分辨率分支: (C,H,W) -> (C,H,W)。"""

    def __init__(self, ch, n_blocks=2, expand_ratio=2):
        super().__init__()
        self.blocks = nn.Sequential(*[
            LiteInvertedResidual(ch, ch, stride=1, expand_ratio=expand_ratio)
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)


class ExchangeUnit(nn.Module):
    """64/32/16 分支间的多尺度融合。"""

    def __init__(self, c64, c32, c16):
        super().__init__()
        self.up_32_to_64 = ConvGNAct(c32, c64, k=1, s=1, p=0)
        self.up_16_to_64 = ConvGNAct(c16, c64, k=1, s=1, p=0)
        self.up_16_to_32 = ConvGNAct(c16, c32, k=1, s=1, p=0)
        self.down_64_to_32 = ConvGNAct(c64, c32, k=3, s=2, p=1)
        self.down_32_to_16 = ConvGNAct(c32, c16, k=3, s=2, p=1)
        self.down_64_to_16 = nn.Sequential(
            ConvGNAct(c64, c32, k=3, s=2, p=1),
            ConvGNAct(c32, c16, k=3, s=2, p=1),
        )

    def forward(self, x64, x32, x16):
        f64 = x64 + self.up_32_to_64(F.interpolate(x32, scale_factor=2, mode="bilinear", align_corners=False)) \
                   + self.up_16_to_64(F.interpolate(x16, scale_factor=4, mode="bilinear", align_corners=False))
        f32 = x32 + self.down_64_to_32(x64) \
                   + self.up_16_to_32(F.interpolate(x16, scale_factor=2, mode="bilinear", align_corners=False))
        f16 = x16 + self.down_32_to_16(x32) + self.down_64_to_16(x64)
        return f64, f32, f16


def _make_coord_maps(h, w, device, dtype):
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(1, 1, h, w)
    return xs, ys


class HRNetFieldHead(nn.Module):
    """HRNet 多尺度场头: 场栅格 -> 64×64 温度场, 用 global_cond 做 FiLM 调制。

    与 U-Net 的区别: 全程保留 64×64 高分辨率分支 (不抹平热点细节), 同时通过 16×16
    低分辨率分支获得全局感受野; 每阶段做密集多尺度交换。cond_dim = 2*hidden (GNN 全局向量)。
    """

    def __init__(self, in_channels=3, base=64, cond_dim=256, stages=4,
                 blocks_per_stage=2, expand_ratio=2):
        super().__init__()
        c64, c32, c16 = base, base * 2, base * 4
        self._c64, self._c32, self._c16 = c64, c32, c16
        self.in_channels = in_channels

        # stem: 输入已在 64×64, stride 1 (+2 坐标图)
        self.stem = nn.Sequential(
            ConvGNAct(in_channels + 2, c64, k=3, s=1, p=1),
            LiteInvertedResidual(c64, c64, stride=1, expand_ratio=expand_ratio),
        )
        self.down_64_to_32_init = ConvGNAct(c64, c32, k=3, s=2, p=1)
        self.down_32_to_16_init = ConvGNAct(c32, c16, k=3, s=2, p=1)

        # 每个分支用 global_cond 做 FiLM
        self.film64 = FiLM(cond_dim, c64)
        self.film32 = FiLM(cond_dim, c32)
        self.film16 = FiLM(cond_dim, c16)

        self.stages = nn.ModuleList()
        for _ in range(stages):
            self.stages.append(nn.ModuleDict({
                "b64": HRBranch(c64, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                "b32": HRBranch(c32, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                "b16": HRBranch(c16, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                "ex": ExchangeUnit(c64=c64, c32=c32, c16=c16),
            }))

        # 头部融合: 上采样到 64×64 + concat 原始栅格作 hint
        head_in = c64 + c32 + c16 + in_channels
        self.head_fuse = ConvGNAct(head_in, c64, k=3, s=1, p=1)
        self.head_out = nn.Conv2d(c64, 1, kernel_size=1)

        self._coord_hw = None
        self._coord_xy = None

    def _coords(self, b, h, w, device, dtype):
        if (self._coord_hw != (h, w) or self._coord_xy is None
                or self._coord_xy[0].device != device or self._coord_xy[0].dtype != dtype):
            self._coord_hw = (h, w)
            self._coord_xy = _make_coord_maps(h, w, device=device, dtype=dtype)
        x, y = self._coord_xy
        return x.expand(b, -1, -1, -1), y.expand(b, -1, -1, -1)

    def forward(self, raster, global_cond):
        b = raster.size(0)
        h, w = raster.size(2), raster.size(3)
        xmap, ymap = self._coords(b, h, w, raster.device, raster.dtype)
        x = torch.cat([raster, xmap, ymap], dim=1)  # [B, in_channels+2, H, W]

        x64 = self.stem(x)
        x32 = self.down_64_to_32_init(x64)
        x16 = self.down_32_to_16_init(x32)

        x64 = self.film64(x64, global_cond)
        x32 = self.film32(x32, global_cond)
        x16 = self.film16(x16, global_cond)

        for st in self.stages:
            x64 = st["b64"](x64)
            x32 = st["b32"](x32)
            x16 = st["b16"](x16)
            x64, x32, x16 = st["ex"](x64, x32, x16)

        u32 = F.interpolate(x32, scale_factor=2, mode="bilinear", align_corners=False)
        u16 = F.interpolate(x16, scale_factor=4, mode="bilinear", align_corners=False)
        feat = torch.cat([x64, u32, u16, raster], dim=1)
        feat = self.head_fuse(feat)
        return self.head_out(feat)  # [B,1,H,W]


class PeakRefineHead(nn.Module):
    """峰值专属高分辨率残差分支 (小网络专门修峰值)。

    动机: 64×64 场头里, 尖锐热点只有 1~3 格, 被多尺度下采样/上采样抹平了峰值。
    本分支把粗 heatmap [B,1,H,W] 上采样到 hi×hi (128/256), 用一个很小的 CNN
    (ch 通道) 学一个残差 Δ, 专门恢复被抹平的尖锐峰值, 而不改动主场的分辨率。

    输出 [B,1,hi,hi] = 上采样粗图 + Δ。每层用 global_cond 做 FiLM 调制。
    """

    def __init__(self, cond_dim, ch=16, n_blocks=2, expand_ratio=2):
        super().__init__()
        self.in_conv = ConvGNAct(1 + 2, ch, k=3, s=1, p=1)  # 上采样粗图 + xy 坐标图
        self.film0 = FiLM(cond_dim, ch)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "res": LiteInvertedResidual(ch, ch, stride=1, expand_ratio=expand_ratio),
                "film": FiLM(cond_dim, ch),
            }) for _ in range(n_blocks)
        ])
        self.out_conv = nn.Conv2d(ch, 1, kernel_size=1)

    def forward(self, coarse, global_cond, hi):
        b = coarse.size(0)
        up = F.interpolate(coarse, size=(hi, hi), mode="bilinear", align_corners=False)
        xmap, ymap = _make_coord_maps(hi, hi, coarse.device, coarse.dtype)
        x = torch.cat([up, xmap.expand(b, -1, -1, -1), ymap.expand(b, -1, -1, -1)], dim=1)
        h = self.film0(self.in_conv(x), global_cond)
        for blk in self.blocks:
            h = blk["film"](blk["res"](h), global_cond)
        return up + self.out_conv(h)


class GNNHRNetModel(nn.Module):
    """GNN (图编码 + 峰值) + HRNet (多尺度场解码)。"""

    def __init__(self, node_dim=8, hidden=128, heads=4, num_layers=3, edge_dim=1,
                 grid=64, hi_grid=None, base=64, stages=4, blocks_per_stage=2,
                 expand_ratio=2, dropout=0.1, peak_branch=False,
                 peak_branch_ch=16, peak_branch_blocks=2):
        super().__init__()
        self.grid = grid
        self.hi_grid = hi_grid if hi_grid else grid
        self.encoder = GNNEncoder(node_dim, hidden=hidden, heads=heads, num_layers=num_layers,
                                  edge_dim=edge_dim, dropout=dropout)
        self.field_head = HRNetFieldHead(in_channels=3, base=base, cond_dim=2 * hidden,
                                         stages=stages, blocks_per_stage=blocks_per_stage,
                                         expand_ratio=expand_ratio)
        self.peak_refine = (PeakRefineHead(cond_dim=2 * hidden, ch=peak_branch_ch,
                                           n_blocks=peak_branch_blocks, expand_ratio=expand_ratio)
                            if peak_branch else None)

    def forward(self, x, edge_index, batch, edge_attr=None, field_raster=None):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        node_emb = self.encoder(x, edge_index, edge_attr)
        global_cond = _global_pool(node_emb, batch, num_graphs)

        heatmap = self.field_head(field_raster, global_cond)
        if self.peak_refine is not None:
            heatmap = self.peak_refine(heatmap, global_cond, self.hi_grid)
        return {"heatmap": heatmap}


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def _spatial_gradient_loss(pred, target):
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    px = F.conv2d(pred, kx, padding=1)
    py = F.conv2d(pred, ky, padding=1)
    gx = F.conv2d(target, kx, padding=1)
    gy = F.conv2d(target, ky, padding=1)
    return torch.mean(torch.abs(px - gx) + torch.abs(py - gy))


def _laplacian_loss(pred, target):
    """二阶曲率损失: 匹配拉普拉斯 (∇²T)。

    峰值处曲率大且为负, 匹配它 = 锐化峰值; 平滑区曲率小, 匹配它 = 约束平滑。
    结构先验, 不针对单格加权。
    """
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                     device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)
    lp = F.conv2d(pred, k, padding=1)
    lt = F.conv2d(target, k, padding=1)
    return torch.mean(torch.abs(lp - lt))


def gnn_thermal_loss(out, target_heatmap, target_peak, *, heatmap_w=1.0, grad_w=0.1,
                     peak_w=1.0, peak_loc=None, laplace_w=0.0):
    """热图损失: MSE + 梯度差 + 二阶曲率 + 峰值位置监督。

    峰值位置监督: 只在「真实峰值所在单元」监督热图值逼近真实峰值温度。
    位置用 argmax 的真实坐标 (不是预测 argmax, 避免漂移), 只动一个单元。
    二阶曲率 (laplace_w): 匹配拉普拉斯, 锐化峰值且约束平滑, 不单格加权。
    """
    hm = F.mse_loss(out["heatmap"], target_heatmap)
    grad = _spatial_gradient_loss(out["heatmap"], target_heatmap)
    loss = heatmap_w * hm + grad_w * grad
    info = {"hm": float(hm.detach()), "grad": float(grad.detach()),
            "loss": float(loss.detach())}

    if laplace_w > 0:
        lap = _laplacian_loss(out["heatmap"], target_heatmap)
        loss = loss + laplace_w * lap
        info["lap"] = float(lap.detach())
        info["loss"] = float(loss.detach())

    if peak_w > 0 and peak_loc is not None:
        B = out["heatmap"].size(0)
        hm_peak = out["heatmap"][torch.arange(B, device=out["heatmap"].device), 0,
                                 peak_loc[:, 0], peak_loc[:, 1]].view(B, 1)
        pl = F.mse_loss(hm_peak, target_peak)
        loss = loss + peak_w * pl
        info["peak"] = float(pl.detach())
        info["loss"] = float(loss.detach())
    return loss, info


# --------------------------------------------------------------------------- #
# 图构建
# --------------------------------------------------------------------------- #
from dataclasses import dataclass, field  # noqa: E402


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    cell_node: torch.Tensor
    side_mm: float


def _build_graph_from_rects(rects, powers, side_mm, grid, hubump_mm=None):
    N = len(rects)
    if hubump_mm is None:
        hubump_mm = [0.0] * N
    feats = []
    for (x, y, w, h), p, hb in zip(rects, powers, hubump_mm):
        xc, yc = x + w / 2.0, y + h / 2.0
        area = w * h
        hb_area = 2.0 * (w + h) * hb
        feats.append([
            p / POWER_SCALE,
            w / side_mm,
            h / side_mm,
            area / (side_mm * side_mm),
            xc / side_mm,
            yc / side_mm,
            hb / HUBUMP_SCALE,
            hb_area / (side_mm * side_mm),
        ])
    x = torch.tensor(feats, dtype=torch.float32)

    centers = [((rx + rw / 2.0) / side_mm, (ry + rh / 2.0) / side_mm) for rx, ry, rw, rh in rects]
    src, dst, dists = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
            dists.append(math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(dists, dtype=torch.float32).view(-1, 1)

    cell = side_mm / grid
    cell_node = torch.full((grid, grid), -1, dtype=torch.long)
    for nid, (rx, ry, rw, rh) in enumerate(rects):
        ix0 = max(0, min(grid, int(math.floor(rx / cell))))
        iy0 = max(0, min(grid, int(math.floor(ry / cell))))
        ix1 = max(0, min(grid, int(math.ceil((rx + rw) / cell))))
        iy1 = max(0, min(grid, int(math.ceil((ry + rh) / cell))))
        if ix1 > ix0 and iy1 > iy0:
            cell_node[iy0:iy1, ix0:ix1] = nid

    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr,
                     cell_node=cell_node, side_mm=side_mm)


def build_graph_from_rects(rects_mm, powers_w, side_mm, grid=64, hubump_mm=None):
    return _build_graph_from_rects(rects_mm, powers_w, side_mm, grid, hubump_mm)


def collate_graphs(graphs):
    grid = graphs[0].cell_node.shape[0]
    xs, eis, eas, batches, cell_nodes = [], [], [], [], []
    node_offset = 0
    for b, g in enumerate(graphs):
        n = g.x.size(0)
        xs.append(g.x)
        eis.append(g.edge_index + node_offset)
        eas.append(g.edge_attr)
        batches.append(torch.full((n,), b, dtype=torch.long))
        cn = g.cell_node.clone()
        cn[cn >= 0] += node_offset
        cell_nodes.append(cn)
        node_offset += n
    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(eis, dim=1),
        "edge_attr": torch.cat(eas, dim=0),
        "batch": torch.cat(batches, dim=0),
        "cell_node": torch.stack(cell_nodes, dim=0),
    }


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def read_ptrace_powers(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path} 行数不足")
    names = lines[0].split()
    powers = lines[1].split()
    return {n: float(p) for n, p in zip(names, powers)}


def parse_l4_thermal(l4_path):
    blk = []
    with open(l4_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = re.split(r"\s+", s)
            if len(parts) < 5:
                continue
            try:
                blk.append((parts[0], float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4])))
            except ValueError:
                continue
    chips = [b for b in blk if b[0].startswith("Chiplet")]
    ubumps = [b for b in blk if b[0].startswith("Ubump")]
    eps = 1e-6
    out = {}
    for name, cw, ch, cx, cy in chips:
        x0, x1, y0, y1 = cx, cx + cw, cy, cy + ch
        ws = []
        for _, uw, uh, ux, uy in ubumps:
            ux0, ux1, uy0, uy1 = ux, ux + uw, uy, uy + uh
            if (abs(uy1 - y0) < eps or abs(uy0 - y1) < eps) and ux1 > x0 + eps and ux0 < x1 - eps:
                ws.append(uh)
            elif (abs(ux1 - x0) < eps or abs(ux0 - x1) < eps) and uy1 > y0 + eps and uy0 < y1 - eps:
                ws.append(uw)
        out[name] = float(np.median(ws)) if ws else 0.0
    ubump_rects_mm = [(ux * 1000.0, uy * 1000.0, uw * 1000.0, uh * 1000.0)
                      for _, uw, uh, ux, uy in ubumps]
    return out, ubump_rects_mm


def rasterize_rects_mm(rects_mm, side_mm, grid, values=None):
    cell = side_mm / grid
    out = np.zeros((grid, grid), dtype=np.float32)
    for k, (x, y, w, h) in enumerate(rects_mm):
        v = 1.0 if values is None else values[k]
        if v == 0.0:
            continue
        ix0 = max(0, int(math.floor(x / cell)))
        iy0 = max(0, int(math.floor(y / cell)))
        ix1 = min(grid - 1, int(math.ceil((x + w) / cell) - 1))
        iy1 = min(grid - 1, int(math.ceil((y + h) / cell) - 1))
        if ix1 >= ix0 and iy1 >= iy0:
            out[iy0:iy1 + 1, ix0:ix1 + 1] = v
    return out


def load_case(i, j, grid=64, hi_grid=None):
    # grid    = 场头/场栅格输入分辨率
    # hi_grid = 目标(输出)温度场分辨率; None 则同 grid (峰值专属高分辨率分支用它)
    hi = hi_grid if hi_grid else grid
    cfg = os.path.join(CFG_ROOT, f"system_{i}_config")
    flp = os.path.join(cfg, "system.flp")
    l4 = os.path.join(cfg, f"system_{i}L4_ChipLayer.flp")
    ptrace = os.path.join(cfg, f"system_{i}_{j}.ptrace")
    temp_csv = os.path.join(DATA_ROOT, "thermal_map", f"system_temp_{i}_{j}.csv")
    peak_csv = os.path.join(DATA_ROOT, "max_temp", f"system_maxtemp_{i}_{j}.csv")

    rects = [(x, y, w, h, name) for (x, y, w, h, name) in dl.parse_flp_rects(flp)
             if name.startswith("Chiplet")]
    pw = read_ptrace_powers(ptrace)
    side_mm = dl.interposer_side_m(l4) * 1000.0

    rects_mm = [(x * 1000.0, y * 1000.0, w * 1000.0, h * 1000.0) for (x, y, w, h, _) in rects]
    powers_w = [pw[name] for (_, _, _, _, name) in rects]
    hubump_w, ubump_rects_mm = parse_l4_thermal(l4)
    hubump_mm = [hubump_w.get(name, 0.0) * 1000.0 for (_, _, _, _, name) in rects]
    graph = build_graph_from_rects(rects_mm, powers_w, side_mm, grid=grid, hubump_mm=hubump_mm)

    # 场栅格: [功率密度, chiplet 掩码, hubump 环掩码]
    areas_mm2 = [w * h for (_, _, w, h) in rects_mm]
    p_density = [p / a if a > 0 else 0.0 for p, a in zip(powers_w, areas_mm2)]
    p_grid = rasterize_rects_mm(rects_mm, side_mm, grid,
                                values=[d / POWER_GRID_SCALE for d in p_density])
    mask_grid = rasterize_rects_mm(rects_mm, side_mm, grid)
    hubump_grid = rasterize_rects_mm(ubump_rects_mm, side_mm, grid)
    field_raster = np.stack([p_grid, mask_grid, hubump_grid], axis=0).astype(np.float32)

    t_raw = dl.read_index_value_csv(temp_csv).reshape(64, 64)
    if hi != 64:
        # 原生 64×64, 立方插值上采样到 hi (插值过原始采样点, 峰值精确保留)
        t_raw = _scipy_zoom(t_raw.astype(np.float64), hi / 64.0, order=3)
    t01 = dl.minmax_scale(t_raw, TEMP_MIN, TEMP_MAX)
    peak_raw = dl.read_scalar_csv(peak_csv)
    peak01 = np.asarray([(peak_raw - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)], dtype=np.float32)

    # 真实峰值位置 (温度场 argmax 的 grid 坐标 [row, col])
    pr, pc = np.unravel_index(int(np.argmax(t_raw)), t_raw.shape)
    peak_loc = np.array([pr, pc], dtype=np.int64)

    temp_t = torch.from_numpy(t01).unsqueeze(0)
    peak_t = torch.tensor(peak01, dtype=torch.float32).view(1)
    peak_loc_t = torch.from_numpy(peak_loc)
    field_t = torch.from_numpy(field_raster)
    return graph, temp_t, peak_t, peak_loc_t, field_t


class GNNThermalDataset(Dataset):
    def __init__(self, cases, grid=64, hi_grid=None):
        self.cases = cases
        self.grid = grid
        self.hi_grid = hi_grid if hi_grid else grid

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        i, j = self.cases[idx]
        graph, temp_t, peak_t, peak_loc_t, field_t = load_case(i, j, self.grid, self.hi_grid)
        return {"graph": graph, "temp": temp_t, "peak": peak_t,
                "peak_loc": peak_loc_t, "field": field_t, "i": i, "j": j}


def collate(batch):
    out = collate_graphs([b["graph"] for b in batch])
    out["temp"] = torch.stack([b["temp"] for b in batch])
    out["peak"] = torch.stack([b["peak"] for b in batch])
    out["peak_loc"] = torch.stack([b["peak_loc"] for b in batch])
    out["field"] = torch.stack([b["field"] for b in batch])
    out["i"] = torch.tensor([b["i"] for b in batch], dtype=torch.long)
    out["j"] = torch.tensor([b["j"] for b in batch], dtype=torch.long)
    return out


def to_device(d, device):
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


@torch.no_grad()
def evaluate(model, loader, device, temp_span):
    model.eval()
    hm_sq, hmax_ae, hmax_se, n, ncase = 0.0, 0.0, 0.0, 0, 0
    for batch in loader:
        b = to_device(batch, device)
        out = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
        hm_sq += float(((out["heatmap"] - b["temp"]) ** 2).sum())
        pred_max = out["heatmap"].amax(dim=(2, 3))
        hmax_ae += float((pred_max - b["peak"]).abs().sum())
        hmax_se += float((pred_max - b["peak"]).sum())
        n += b["temp"].numel()
        ncase += b["peak"].numel()
    hm_rmse_c = (hm_sq / n) ** 0.5 * temp_span
    hmax_mae_c = hmax_ae / max(ncase, 1) * temp_span
    hmax_bias_c = hmax_se / max(ncase, 1) * temp_span
    return hm_rmse_c, hmax_mae_c, hmax_bias_c


def benchmark_speed(model, loader, device, iters=50):
    model.eval()
    batch = to_device(next(iter(loader)), device)
    bs = batch["temp"].size(0)
    for _ in range(10):
        model(batch["x"], batch["edge_index"], batch["batch"], batch["edge_attr"], batch["field"])
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(batch["x"], batch["edge_index"], batch["batch"], batch["edge_attr"], batch["field"])
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return bs / dt, dt * 1000.0 / bs


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_train", type=int, default=64000)
    ap.add_argument("--num_val", type=int, default=8000)
    ap.add_argument("--num_workers", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--hi_grid", type=int, default=None,
                    help="峰值高分辨率分支输出分辨率 (默认同 grid; 128/256 让尖峰不被抹平)")
    ap.add_argument("--peak_branch", action="store_true",
                    help="峰值专属高分辨率残差分支 (小网络专门修峰值)")
    ap.add_argument("--peak_branch_ch", type=int, default=16)
    ap.add_argument("--peak_branch_blocks", type=int, default=2)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--blocks_per_stage", type=int, default=2)
    ap.add_argument("--expand_ratio", type=int, default=2)
    ap.add_argument("--grad_w", type=float, default=0.1)
    ap.add_argument("--laplace_w", type=float, default=0.0, help="二阶曲率(拉普拉斯)监督权重")
    ap.add_argument("--peak_w", type=float, default=0.0, help="峰值位置监督权重 (实测会伤 RMSE, 默认关)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--amp", action="store_true", help="混合精度 (bf16 autocast) 加速训练")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    temp_span = TEMP_MAX - TEMP_MIN

    all_cases = dl.list_cases(os.path.join(DATA_ROOT, "power_map"))
    train_cases, val_cases, _ = dl.split_cases_by_i(all_cases, seed=args.seed)
    train_cases = train_cases[:args.num_train]
    val_cases = val_cases[:args.num_val]

    train_ds = GNNThermalDataset(train_cases, grid=args.grid, hi_grid=args.hi_grid)
    val_ds = GNNThermalDataset(val_cases, grid=args.grid, hi_grid=args.hi_grid)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate)

    model = GNNHRNetModel(node_dim=8, hidden=args.hidden, heads=args.heads,
                          num_layers=args.num_layers, grid=args.grid, hi_grid=args.hi_grid,
                          base=args.base, stages=args.stages,
                          blocks_per_stage=args.blocks_per_stage,
                          expand_ratio=args.expand_ratio,
                          peak_branch=args.peak_branch,
                          peak_branch_ch=args.peak_branch_ch,
                          peak_branch_blocks=args.peak_branch_blocks).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[setup] train={len(train_cases)} val={len(val_cases)} "
          f"params={nparams/1e6:.2f}M device={device} "
          f"grid={args.grid} hi_grid={model.hi_grid} peak_branch={args.peak_branch}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    use_amp = bool(args.amp and device.type == "cuda")

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "checkpoints", "gnnhrnet")
    os.makedirs(out_dir, exist_ok=True)
    best_rmse = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, steps = 0.0, 0
        for batch in train_loader:
            b = to_device(batch, device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
                    loss, info = gnn_thermal_loss(
                        out, b["temp"], b["peak"], grad_w=args.grad_w, peak_w=args.peak_w,
                        peak_loc=b["peak_loc"], laplace_w=args.laplace_w)
            else:
                out = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
                loss, info = gnn_thermal_loss(
                    out, b["temp"], b["peak"], grad_w=args.grad_w, peak_w=args.peak_w,
                    peak_loc=b["peak_loc"], laplace_w=args.laplace_w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += info["loss"]
            steps += 1
        sched.step()
        hm_rmse, hmax_mae, hmax_bias = evaluate(model, val_loader, device, temp_span)
        dt = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]
        print(f"[epoch {ep}/{args.epochs}] loss={tot_loss/max(steps,1):.5f} "
              f"val_hm_rmse={hm_rmse:.3f}C  val_peak_mae={hmax_mae:.3f}C  "
              f"val_peak_bias={hmax_bias:+.3f}C  lr={lr_now:.2e}  ({dt:.1f}s)")

        if hm_rmse < best_rmse:
            best_rmse = hm_rmse
            torch.save({"epoch": ep, "model": model.state_dict(), "best_rmse": best_rmse},
                       os.path.join(out_dir, "best.pth"))
            print(f"[ckpt] best saved (val_hm_rmse={best_rmse:.3f}C)")

    sps, ms = benchmark_speed(model, val_loader, device)
    print(f"[speed] {sps:.1f} samples/s  ({ms:.2f} ms/sample, GPU)")
    print(f"[done] best_val_hm_rmse={best_rmse:.3f}C")


if __name__ == "__main__":
    main()
