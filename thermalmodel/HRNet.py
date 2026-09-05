"""ThermalGuidanceHRNet model and losses (self-contained, torch only)."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_coord_maps(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return x/y coordinate maps in [-1,1], shape (1,1,H,W)."""
    xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, 1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(1, 1, h, w)
    return xs, ys


def _group_norm(ch: int) -> nn.GroupNorm:
    for g in (16, 8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(num_groups=g, num_channels=ch)
    return nn.GroupNorm(num_groups=1, num_channels=ch)


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.gn = _group_norm(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class LiteInvertedResidual(nn.Module):
    """Lite inverted residual block (MobileNetV3-style) using SiLU + GroupNorm."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int = 2):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.use_res:
            y = y + x
        return self.out_act(y)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, feat_ch: int, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, feat_ch * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.mlp(cond)
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta


class HRBranch(nn.Module):
    """A single-resolution branch: (C,H,W)->(C,H,W) with several LiteInvertedResidual blocks."""

    def __init__(self, ch: int, n_blocks: int = 2, expand_ratio: int = 2):
        super().__init__()
        self.blocks = nn.Sequential(*[LiteInvertedResidual(ch, ch, stride=1, expand_ratio=expand_ratio) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ExchangeUnit(nn.Module):
    """Multi-scale fusion among high/mid/low branches (e.g. 128/64/32).

    Up: bilinear + 1x1 conv
    Down: strided 3x3 conv
    """

    def __init__(self, c_hi: int, c_mid: int, c_lo: int):
        super().__init__()

        # upsample (low -> high)
        self.up_mid_to_hi = ConvGNAct(c_mid, c_hi, k=1, s=1, p=0)
        self.up_lo_to_hi = ConvGNAct(c_lo, c_hi, k=1, s=1, p=0)
        self.up_lo_to_mid = ConvGNAct(c_lo, c_mid, k=1, s=1, p=0)

        # downsample (high -> low)
        self.down_hi_to_mid = ConvGNAct(c_hi, c_mid, k=3, s=2, p=1)
        self.down_mid_to_lo = ConvGNAct(c_mid, c_lo, k=3, s=2, p=1)

        # cross downsample (hi -> lo) via two strided convs
        self.down_hi_to_lo = nn.Sequential(
            ConvGNAct(c_hi, c_mid, k=3, s=2, p=1),
            ConvGNAct(c_mid, c_lo, k=3, s=2, p=1),
        )

    def forward(self, x_hi: torch.Tensor, x_mid: torch.Tensor, x_lo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # high: from hi + up(mid) + up(lo)
        f_hi = x_hi
        f_hi = f_hi + self.up_mid_to_hi(F.interpolate(x_mid, scale_factor=2, mode="bilinear", align_corners=False))
        f_hi = f_hi + self.up_lo_to_hi(F.interpolate(x_lo, scale_factor=4, mode="bilinear", align_corners=False))

        # mid: from mid + down(hi) + up(lo)
        f_mid = x_mid
        f_mid = f_mid + self.down_hi_to_mid(x_hi)
        f_mid = f_mid + self.up_lo_to_mid(F.interpolate(x_lo, scale_factor=2, mode="bilinear", align_corners=False))

        # low: from lo + down(mid) + down(hi)
        f_lo = x_lo
        f_lo = f_lo + self.down_mid_to_lo(x_mid)
        f_lo = f_lo + self.down_hi_to_lo(x_hi)

        return f_hi, f_mid, f_lo


class ThermalGuidanceHRNet(nn.Module):
    """HRNet-style thermal guidance model.

    Inputs:
      - power_grid:  (B,1,128,128)
      - layout_mask: (B,1,128,128)
      - total_power: (B,1)

    Model input tensor: concat [power, mask, x_coord, y_coord] -> (B,4,128,128)

    Outputs:
      - temp_grid: (B,1,128,128)
      - avg_temp:  (B,1)

    Branches: 128x128 (c128=base) / 64x64 (c64=base*2) / 32x32 (c32=base*4),
    with multi-scale exchange units at each stage; power_grid hint concatenated before head.

    Note:
      - Model predicts normalized temperature in [0,1]; denorm to Celsius is handled in eval.
    """

    def __init__(
        self,
        base: int = 32,
        cond_dim: int = 1,
        *,
        stages: int = 4,
        blocks_per_stage: int = 2,
        expand_ratio: int = 2,
        mean_calib: bool = False,
    ):
        super().__init__()

        self.mean_calib = bool(mean_calib)  # kept for checkpoint/flag compatibility; not used

        # widths for 128/64/32 branches
        c128 = int(base)
        c64 = int(base * 2)
        c32 = int(base * 4)
        self._c128, self._c64, self._c32 = c128, c64, c32

        # stem: 128x128 -> 128x128 feature for high-res branch
        self.stem = nn.Sequential(
            ConvGNAct(4, c128, k=3, s=1, p=1),  # 128 -> 128
            LiteInvertedResidual(c128, c128, stride=1, expand_ratio=expand_ratio),
        )

        # generate lower-res branches from high-res
        self.down_128_to_64_init = ConvGNAct(c128, c64, k=3, s=2, p=1)  # 128 -> 64
        self.down_64_to_32_init = ConvGNAct(c64, c32, k=3, s=2, p=1)  # 64 -> 32

        # Per-branch FiLM (sync inject total_power into ALL resolutions)
        self.film128 = FiLM(cond_dim=cond_dim, feat_ch=c128)
        self.film64 = FiLM(cond_dim=cond_dim, feat_ch=c64)
        self.film32 = FiLM(cond_dim=cond_dim, feat_ch=c32)

        # HRNet stages: branch blocks + exchange
        self.stages = nn.ModuleList()
        for _ in range(int(stages)):
            self.stages.append(
                nn.ModuleDict(
                    {
                        "b128": HRBranch(c128, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "b64": HRBranch(c64, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "b32": HRBranch(c32, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "ex": ExchangeUnit(c_hi=c128, c_mid=c64, c_lo=c32),
                    }
                )
            )

        # avg head (aux) from lowest-res features (global context)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_head = nn.Sequential(
            nn.Linear(c32, base * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

        # power hint injection before output head
        # power_hint is (B,1,128,128) from power_grid directly
        head_in = c128 + c64 + c32 + 1
        self.head_fuse = ConvGNAct(head_in, c128, k=3, s=1, p=1)
        self.head_out = nn.Conv2d(c128, 1, kernel_size=1)

        # coord cache
        self._coord_hw: Optional[Tuple[int, int]] = None
        self._coord_xy: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _coords(self, b: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._coord_hw != (h, w)
            or self._coord_xy is None
            or self._coord_xy[0].device != device
            or self._coord_xy[0].dtype != dtype
        ):
            self._coord_hw = (h, w)
            self._coord_xy = _make_coord_maps(h, w, device=device, dtype=dtype)
        x, y = self._coord_xy
        return x.expand(b, -1, -1, -1), y.expand(b, -1, -1, -1)

    def forward(self, power_grid: torch.Tensor, layout_mask: torch.Tensor, total_power: Optional[torch.Tensor] = None):
        b, _, h, w = power_grid.shape
        xmap, ymap = self._coords(b, h, w, device=power_grid.device, dtype=power_grid.dtype)

        x = torch.cat([power_grid, layout_mask, xmap, ymap], dim=1)

        x128 = self.stem(x)  # (B,c128,128,128)
        x64 = self.down_128_to_64_init(x128)  # (B,c64,64,64)
        x32 = self.down_64_to_32_init(x64)  # (B,c32,32,32)

        if total_power is None:
            cond = torch.zeros((b, 1), device=x128.device, dtype=x128.dtype)
        else:
            cond = total_power.view(b, 1).to(device=x128.device, dtype=x128.dtype)

        # condition injection on all branches (synchronized)
        x128 = self.film128(x128, cond)
        x64 = self.film64(x64, cond)
        x32 = self.film32(x32, cond)

        for st in self.stages:
            x128 = st["b128"](x128)
            x64 = st["b64"](x64)
            x32 = st["b32"](x32)
            x128, x64, x32 = st["ex"](x128, x64, x32)

        # aux avg head from lowest-res branch
        pooled = self.avg_pool(x32).flatten(1)
        avg = self.avg_head(pooled)

        # Head fusion: upsample all to 128x128 + concat + power hint
        u128 = x128  # already (B,c128,128,128)
        u64 = F.interpolate(x64, scale_factor=2, mode="bilinear", align_corners=False)
        u32 = F.interpolate(x32, scale_factor=4, mode="bilinear", align_corners=False)

        power_hint = power_grid  # already (B,1,128,128)
        feat = torch.cat([u128, u64, u32, power_hint], dim=1)

        feat = self.head_fuse(feat)
        out = self.head_out(feat)

        return out, avg


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    return kx, ky


def spatial_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kx, ky = _sobel_filters(pred.device, pred.dtype)
    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)
    return F.mse_loss(gx_p, gx_t) + F.mse_loss(gy_p, gy_t)


def guidance_loss(
    pred_grid: torch.Tensor,
    target_grid: torch.Tensor,
    *,
    pred_avg: Optional[torch.Tensor] = None,
    target_avg: Optional[torch.Tensor] = None,
    grad_w: float = 0.01,
    avg_w: float = 0.1,
    mean_consistency_w: float = 0.1,
    under_w: float = 1.0,
    hotspot_mode: str = "linear",
    hotspot_alpha: float = 3.0,
    hotspot_beta: float = 3.0,
    hotspot_pow: float = 4.0,
    maxpool_w: float = 0.0,
    maxpool_ks: int = 4,
    topk_w: float = 0.0,
    topk_k: int = 0,
    peak_w: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    err = pred_grid - target_grid
    err2 = err * err
    if under_w != 1.0:
        err2 = torch.where(err < 0, err2 * float(under_w), err2)

    tmin = target_grid.amin(dim=(2, 3), keepdim=True)
    tmax = target_grid.amax(dim=(2, 3), keepdim=True)
    norm_t = (target_grid - tmin) / (tmax - tmin + 1e-8)

    if hotspot_mode == "exp":
        weight = torch.exp(norm_t * float(hotspot_alpha))
    elif hotspot_mode == "pow":
        weight = 1.0 + float(hotspot_beta) * (norm_t ** float(hotspot_pow))
    else:
        weight = 1.0 + 3.0 * norm_t

    weighted_mse = (err2 * weight).mean()
    grad = spatial_gradient_loss(pred_grid, target_grid)

    loss = weighted_mse + grad_w * grad
    out: Dict[str, float] = {
        "mse": float(weighted_mse.detach().cpu()),
        "grad": float(grad.detach().cpu()),
        "loss": float(loss.detach().cpu()),
    }

    if topk_w and topk_w > 0 and topk_k and topk_k > 0:
        b, _, hh, ww = target_grid.shape
        k = int(min(int(topk_k), hh * ww))
        flat_t = target_grid.view(b, -1)
        flat_e = err2.view(b, -1)
        idx = torch.topk(flat_t, k=k, dim=1, largest=True, sorted=False).indices
        topk_mse = flat_e.gather(1, idx).mean()
        loss = loss + float(topk_w) * topk_mse
        out["topk_mse"] = float(topk_mse.detach().cpu())
        out["loss"] = float(loss.detach().cpu())

    if maxpool_w and maxpool_w > 0:
        ks = int(max(1, maxpool_ks))
        p_max = F.max_pool2d(pred_grid, kernel_size=ks, stride=ks)
        t_max = F.max_pool2d(target_grid, kernel_size=ks, stride=ks)
        maxpool_mse = F.mse_loss(p_max, t_max)
        loss = loss + float(maxpool_w) * maxpool_mse
        out["maxpool_mse"] = float(maxpool_mse.detach().cpu())
        out["loss"] = float(loss.detach().cpu())

    if peak_w and peak_w > 0:
        pred_peak = pred_grid.amax(dim=(2, 3)).view(-1, 1)
        tgt_peak = target_grid.amax(dim=(2, 3)).view(-1, 1)
        peak_mse = F.mse_loss(pred_peak, tgt_peak)
        loss = loss + float(peak_w) * peak_mse
        out["peak_mse"] = float(peak_mse.detach().cpu())
        out["loss"] = float(loss.detach().cpu())

    if pred_avg is not None and target_avg is not None:
        pred_avg = pred_avg.view(-1, 1)
        target_avg = target_avg.view(-1, 1)
        avg_mse = F.mse_loss(pred_avg, target_avg)
        loss = loss + avg_w * avg_mse
        out["avg_mse"] = float(avg_mse.detach().cpu())
        out["loss"] = float(loss.detach().cpu())

        grid_mean = pred_grid.mean(dim=(2, 3), keepdim=False).view(-1, 1)
        mean_cons = F.l1_loss(grid_mean, pred_avg)
        loss = loss + mean_consistency_w * mean_cons
        out["mean_cons"] = float(mean_cons.detach().cpu())
        out["loss"] = float(loss.detach().cpu())

    return loss, out


def _spatial_gradient_abs_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kx, ky = _sobel_filters(pred.device, pred.dtype)
    px = F.conv2d(pred, kx, padding=1)
    py = F.conv2d(pred, ky, padding=1)
    gx = F.conv2d(target, kx, padding=1)
    gy = F.conv2d(target, ky, padding=1)
    return torch.mean(torch.abs(px - gx) + torch.abs(py - gy))
