import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running `python HRNet.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset, compute_minmax, split_cases_by_i
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay


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
    """Multi-scale fusion among 64/32/16 branches.

    Up: bilinear + 1x1 conv
    Down: strided 3x3 conv
    """

    def __init__(self, c64: int, c32: int, c16: int):
        super().__init__()

        # upsample (low -> high)
        self.up_32_to_64 = ConvGNAct(c32, c64, k=1, s=1, p=0)
        self.up_16_to_64 = ConvGNAct(c16, c64, k=1, s=1, p=0)
        self.up_16_to_32 = ConvGNAct(c16, c32, k=1, s=1, p=0)

        # downsample (high -> low)
        self.down_64_to_32 = ConvGNAct(c64, c32, k=3, s=2, p=1)
        self.down_32_to_16 = ConvGNAct(c32, c16, k=3, s=2, p=1)

        # cross downsample (64 -> 16) via two strided convs
        self.down_64_to_16 = nn.Sequential(
            ConvGNAct(c64, c32, k=3, s=2, p=1),
            ConvGNAct(c32, c16, k=3, s=2, p=1),
        )

    def forward(self, x64: torch.Tensor, x32: torch.Tensor, x16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # high (64): from 64 + up(32) + up(16)
        f64 = x64
        f64 = f64 + self.up_32_to_64(F.interpolate(x32, scale_factor=2, mode="bilinear", align_corners=False))
        f64 = f64 + self.up_16_to_64(F.interpolate(x16, scale_factor=4, mode="bilinear", align_corners=False))

        # mid (32): from 32 + down(64) + up(16)
        f32 = x32
        f32 = f32 + self.down_64_to_32(x64)
        f32 = f32 + self.up_16_to_32(F.interpolate(x16, scale_factor=2, mode="bilinear", align_corners=False))

        # low (16): from 16 + down(32) + down(64)
        f16 = x16
        f16 = f16 + self.down_32_to_16(x32)
        f16 = f16 + self.down_64_to_16(x64)

        return f64, f32, f16


class ThermalGuidanceHRNet(nn.Module):
    """HRNet-style thermal guidance model.

    Inputs (same as guidance_model.py):
      - power_grid:  (B,1,128,128)
      - layout_mask: (B,1,128,128)
      - total_power: (B,1)

    Model input tensor: concat [power, mask, x_coord, y_coord] -> (B,4,128,128)

    Outputs:
      - temp_grid: (B,1,64,64)
      - avg_temp:  (B,1)

    Key differences vs old UNet:
      - maintains a persistent 64x64 high-res branch
      - multi-scale exchange units at each stage
      - removes hard mean calibration; uses power_grid hint concatenated before head

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

        # widths for 64/32/16 branches
        c64 = int(base)
        c32 = int(base * 2)
        c16 = int(base * 4)
        self._c64, self._c32, self._c16 = c64, c32, c16

        # stem: 128x128 -> 64x64 feature for high-res branch
        self.stem = nn.Sequential(
            ConvGNAct(4, c64, k=3, s=2, p=1),  # 128 -> 64
            LiteInvertedResidual(c64, c64, stride=1, expand_ratio=expand_ratio),
        )

        # generate lower-res branches from high-res
        self.down_64_to_32_init = ConvGNAct(c64, c32, k=3, s=2, p=1)  # 64 -> 32
        self.down_32_to_16_init = ConvGNAct(c32, c16, k=3, s=2, p=1)  # 32 -> 16

        # Per-branch FiLM (sync inject total_power into ALL resolutions)
        self.film64 = FiLM(cond_dim=cond_dim, feat_ch=c64)
        self.film32 = FiLM(cond_dim=cond_dim, feat_ch=c32)
        self.film16 = FiLM(cond_dim=cond_dim, feat_ch=c16)

        # HRNet stages: branch blocks + exchange
        self.stages = nn.ModuleList()
        for _ in range(int(stages)):
            self.stages.append(
                nn.ModuleDict(
                    {
                        "b64": HRBranch(c64, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "b32": HRBranch(c32, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "b16": HRBranch(c16, n_blocks=blocks_per_stage, expand_ratio=expand_ratio),
                        "ex": ExchangeUnit(c64=c64, c32=c32, c16=c16),
                    }
                )
            )

        # avg head (aux) from lowest-res features (global context)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_head = nn.Sequential(
            nn.Linear(c16, base * 2),
            nn.SiLU(inplace=True),
            nn.Linear(base * 2, 1),
        )

        # power hint injection before output head
        # power_hint is (B,1,64,64) from downsample power_grid 128->64
        head_in = c64 + c32 + c16 + 1
        self.head_fuse = ConvGNAct(head_in, c64, k=3, s=1, p=1)
        self.head_out = nn.Conv2d(c64, 1, kernel_size=1)

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

        x64 = self.stem(x)  # (B,c64,64,64)
        x32 = self.down_64_to_32_init(x64)  # (B,c32,32,32)
        x16 = self.down_32_to_16_init(x32)  # (B,c16,16,16)

        if total_power is None:
            cond = torch.zeros((b, 1), device=x64.device, dtype=x64.dtype)
        else:
            cond = total_power.view(b, 1).to(device=x64.device, dtype=x64.dtype)

        # condition injection on all branches (synchronized)
        x64 = self.film64(x64, cond)
        x32 = self.film32(x32, cond)
        x16 = self.film16(x16, cond)

        for st in self.stages:
            x64 = st["b64"](x64)
            x32 = st["b32"](x32)
            x16 = st["b16"](x16)
            x64, x32, x16 = st["ex"](x64, x32, x16)

        # aux avg head from lowest-res branch
        pooled = self.avg_pool(x16).flatten(1)
        avg = self.avg_head(pooled)

        # Head fusion: upsample all to 64x64 + concat + power hint
        u32 = F.interpolate(x32, scale_factor=2, mode="bilinear", align_corners=False)
        u16 = F.interpolate(x16, scale_factor=4, mode="bilinear", align_corners=False)

        power_hint = F.interpolate(power_grid, size=(64, 64), mode="bilinear", align_corners=False)
        feat = torch.cat([x64, u32, u16, power_hint], dim=1)

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


@dataclass
class _Stats:
    temp_min: float
    temp_max: float


def _maybe_temp_stats(ckpt: dict) -> Optional[_Stats]:
    s = ckpt.get("stats")
    if not isinstance(s, dict):
        return None
    try:
        return _Stats(temp_min=float(s["temp_min"]), temp_max=float(s["temp_max"]))
    except Exception:
        return None


def _denorm_temp_c(x01: torch.Tensor, st: _Stats) -> torch.Tensor:
    return x01 * (st.temp_max - st.temp_min) + st.temp_min


@torch.no_grad()
def _eval_val_metrics_denorm(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    *,
    stats: "MinMaxStats",
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    st = _Stats(temp_min=float(stats.temp_min), temp_max=float(stats.temp_max))

    rmse_sum = 0.0
    rmse_min = float("inf")
    rmse_max = 0.0
    grad_sum = 0.0
    max_ae = 0.0
    n = 0
    n_batches = 0

    eps = 1e-6
    rmspe_sum = 0.0

    for batch in val_loader:
        totalp = batch.get("total_power")

        power = batch["power"].to(device)
        layout = batch["layout"].to(device)
        temp = batch["temp"].to(device)
        totalp = totalp.to(device) if totalp is not None else None

        pred_grid, _pred_avg = model(power, layout, totalp)

        pred_eval = pred_grid.detach().cpu()
        temp_eval = temp.detach().cpu()

        pred_t = _denorm_temp_c(pred_eval, st)
        gt_t = _denorm_temp_c(temp_eval, st)

        diff = pred_t - gt_t
        mse = torch.mean(diff * diff, dim=(1, 2, 3))
        rmse = torch.sqrt(mse)

        grad = _spatial_gradient_abs_metric(pred_t, gt_t)

        batch_max_ae = float(diff.abs().max().item())
        max_ae = max(max_ae, batch_max_ae)

        pe2 = torch.mean(((diff / (gt_t.abs() + eps)) ** 2), dim=(1, 2, 3))
        rmspe = torch.sqrt(pe2) * 100.0
        rmspe_sum += float(rmspe.mean().item())

        for bb in range(rmse.shape[0]):
            r = float(rmse[bb].item())
            rmse_sum += r
            rmse_min = min(rmse_min, r)
            rmse_max = max(rmse_max, r)
            n += 1

        grad_sum += float(grad.item())
        n_batches += 1

    return {
        "units": "C",
        "n_cases": int(n),
        "mean_rmse": float(rmse_sum / max(n, 1)),
        "min_rmse": float(rmse_min if n > 0 else 0.0),
        "max_rmse": float(rmse_max),
        "mean_grad": float(grad_sum / max(n_batches, 1)),
        "max_ae": float(max_ae),
        "mean_rmspe_pct": float(rmspe_sum / max(n_batches, 1)),
    }


def main_train(args) -> None:
    torch.manual_seed(args.seed)
    device = _device()

    dataset_all = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    all_cases = dataset_all.cases
    train_cases, val_cases, _test_cases = split_cases_by_i(all_cases, seed=args.seed, train_ratio=0.8, val_ratio=0.1)

    if args.limit_train and args.limit_train > 0:
        train_cases = train_cases[: int(args.limit_train)]
    if args.limit_val and args.limit_val > 0:
        val_cases = val_cases[: int(args.limit_val)]

    train_stats = compute_minmax(dataset_all.data_root, grid_size=dataset_all.temp_grid_size, cases=train_cases)

    train_set = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        stats=train_stats,
        cases=train_cases,
    )
    val_set = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        stats=train_stats,
        cases=val_cases,
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    n_total = len(dataset_all)
    n_train = len(train_set)
    n_val = len(val_set)

    model = ThermalGuidanceHRNet(
        base=args.base,
        stages=args.stages,
        blocks_per_stage=args.blocks_per_stage,
        expand_ratio=args.expand_ratio,
        mean_calib=(not args.disable_mean_calib),
    ).to(device)

    if getattr(args, "qat", False):
        raise SystemExit("--qat is no longer supported; quantization/QAT has been removed")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    resume_ckpt = getattr(args, "resume_ckpt", "")
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] ckpt={resume_ckpt} | start_epoch={start_epoch} | target_epochs={args.epochs}")

    out_dir = args.out_dir if hasattr(args, "out_dir") and args.out_dir else os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_tag = args.ckpt_tag if hasattr(args, "ckpt_tag") else ""

    print("==== Thermal Guidance HRNet Training Config ====")
    print(f"device: {device} (cuda_available={torch.cuda.is_available()})")
    if torch.cuda.is_available():
        try:
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    print("power_grid: 128 -> temp_grid: 64")
    print(f"dataset_total: {n_total} | train: {n_train} | val: {n_val}")
    print(f"batch_size: {args.batch_size} | train_batches: {len(train_loader)} | val_batches: {len(val_loader)}")
    print(f"seed: {args.seed}")
    if getattr(args, "amp", False):
        print(f"amp: True ({args.amp_dtype})")
    else:
        print("amp: False")
    if ckpt_tag:
        print(f"ckpt_tag: {ckpt_tag}")
    print(f"out_dir: {out_dir}")
    print("==============================================")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        use_amp = bool(getattr(args, "amp", False) and device.type == "cuda")
        amp_dtype = torch.float16 if getattr(args, "amp_dtype", "fp16") == "fp16" else torch.bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

        for it, batch in enumerate(train_loader):
            totalp = batch.get("total_power")
            avg = batch.get("avg_temp")

            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = totalp.to(device) if totalp is not None else None
            avg = avg.to(device) if avg is not None else None

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred_grid, pred_avg = model(power, layout, totalp)
                    loss, m = guidance_loss(
                        pred_grid,
                        temp,
                        pred_avg=pred_avg,
                        target_avg=avg,
                        grad_w=args.grad_w,
                        avg_w=args.avg_w,
                        mean_consistency_w=args.mean_consistency_w,
                        under_w=args.under_w,
                        hotspot_mode=args.hotspot_mode,
                        hotspot_alpha=args.hotspot_alpha,
                        hotspot_beta=args.hotspot_beta,
                        hotspot_pow=args.hotspot_pow,
                        maxpool_w=args.maxpool_w,
                        maxpool_ks=args.maxpool_ks,
                        topk_w=args.topk_w,
                        topk_k=args.topk_k,
                        peak_w=args.peak_w,
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred_grid, pred_avg = model(power, layout, totalp)
                loss, m = guidance_loss(
                    pred_grid,
                    temp,
                    pred_avg=pred_avg,
                    target_avg=avg,
                    grad_w=args.grad_w,
                    avg_w=args.avg_w,
                    mean_consistency_w=args.mean_consistency_w,
                    under_w=args.under_w,
                    hotspot_mode=args.hotspot_mode,
                    hotspot_alpha=args.hotspot_alpha,
                    hotspot_beta=args.hotspot_beta,
                    hotspot_pow=args.hotspot_pow,
                    maxpool_w=args.maxpool_w,
                    maxpool_ks=args.maxpool_ks,
                    topk_w=args.topk_w,
                    topk_k=args.topk_k,
                    peak_w=args.peak_w,
                )
                loss.backward()
                opt.step()

            if epoch == start_epoch and it == 0:
                with torch.no_grad():
                    b0_pred = pred_grid[0].detach().cpu()
                    b0_gt = temp[0].detach().cpu()
                    b0_pred_avg = float(pred_avg[0].detach().cpu().item())
                    b0_gt_avg = float(avg[0].detach().cpu().item()) if avg is not None else float("nan")
                    b0_totalp = float(totalp[0].detach().cpu().item()) if totalp is not None else float("nan")
                    print(
                        f"[one-batch] ep {epoch:04d} it {it:04d} "
                        f"loss {m['loss']:.6f} mse {m['mse']:.6f} grad {m['grad']:.6f} "
                        f"avg_pred {b0_pred_avg:.4f} avg_gt {b0_gt_avg:.4f} totalp {b0_totalp:.4f} "
                        f"pred(min/max/avg) {float(b0_pred.min()):.4f}/{float(b0_pred.max()):.4f}/{float(b0_pred.mean()):.4f} "
                        f"gt(min/max/avg) {float(b0_gt.min()):.4f}/{float(b0_gt.max()):.4f}/{float(b0_gt.mean()):.4f}"
                    )

            if it % args.print_every == 0:
                with torch.no_grad():
                    pmax = float(pred_grid.max().item())
                    pavg = float(pred_grid.mean().item())
                    tmax = float(temp.max().item())
                    tavg = float(temp.mean().item())
                print(
                    f"[train] ep {epoch:04d} it {it:04d} loss {m['loss']:.6f} mse {m['mse']:.6f} grad {m['grad']:.6f} "
                    f"pred(max/avg) {pmax:.4f}/{pavg:.4f} gt(max/avg) {tmax:.4f}/{tavg:.4f}"
                )

        model.eval()
        vm = {"loss": 0.0, "mse": 0.0, "grad": 0.0, "avg_mse": 0.0, "mean_cons": 0.0}
        steps = 0
        with torch.no_grad():
            for batch in val_loader:
                totalp = batch.get("total_power")
                avg = batch.get("avg_temp")

                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                temp = batch["temp"].to(device)
                totalp = totalp.to(device) if totalp is not None else None
                avg = avg.to(device) if avg is not None else None

                pred_grid, pred_avg = model(power, layout, totalp)
                _, m = guidance_loss(
                    pred_grid,
                    temp,
                    pred_avg=pred_avg,
                    target_avg=avg,
                    grad_w=args.grad_w,
                    avg_w=args.avg_w,
                    mean_consistency_w=args.mean_consistency_w,
                    under_w=args.under_w,
                    hotspot_mode=args.hotspot_mode,
                    hotspot_alpha=args.hotspot_alpha,
                    hotspot_beta=args.hotspot_beta,
                    hotspot_pow=args.hotspot_pow,
                    maxpool_w=args.maxpool_w,
                    maxpool_ks=args.maxpool_ks,
                )
                for k in vm:
                    vm[k] += m.get(k, 0.0)
                steps += 1
        for k in vm:
            vm[k] /= max(steps, 1)
        print(
            f"Epoch {epoch:04d} | val loss {vm['loss']:.6f} mse {vm['mse']:.6f} grad {vm['grad']:.6f} "
            f"avg_mse {vm['avg_mse']:.6f} mean_cons {vm['mean_cons']:.6f}"
        )

        if epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "stats": train_stats.to_dict(),
                "grid_size": 128,
                "base": args.base,
                "batch_size": args.batch_size,
                "lr": float(args.lr),
                "grad_w": float(args.grad_w),
                "avg_w": float(args.avg_w),
                "mean_consistency_w": float(args.mean_consistency_w),
                "under_w": float(args.under_w),
                "hotspot_mode": str(args.hotspot_mode),
                "hotspot_alpha": float(args.hotspot_alpha),
                "hotspot_beta": float(args.hotspot_beta),
                "hotspot_pow": float(args.hotspot_pow),
                "maxpool_w": float(args.maxpool_w),
                "maxpool_ks": int(args.maxpool_ks),
                "topk_w": float(args.topk_w),
                "topk_k": int(args.topk_k),
                "peak_w": float(args.peak_w),
                "disable_mean_calib": bool(args.disable_mean_calib),
                "mean_calib": bool(not args.disable_mean_calib),
                "seed": int(args.seed),
                "ckpt_tag": ckpt_tag,
                "stages": int(args.stages),
                "blocks_per_stage": int(args.blocks_per_stage),
                "expand_ratio": int(args.expand_ratio),
                "limit_train": int(args.limit_train),
                "limit_val": int(args.limit_val),
            }

            lr_tok = f"{float(args.lr):.0e}"
            gw_tok = f"{float(args.grad_w):g}".replace(".", "p")
            prefix = "hrnet" + (f"_{ckpt_tag}" if ckpt_tag else "")

            name = (
                f"{prefix}_ep{epoch:04d}_seed{args.seed}_bs{args.batch_size}_lr{lr_tok}_base{args.base}_gw{gw_tok}"
                f"_s{args.stages}_b{args.blocks_per_stage}_er{args.expand_ratio}"
                f"_tr{args.limit_train}_va{args.limit_val}.pth"
            )
            path = os.path.join(out_dir, name)
            torch.save(ckpt, path)
            print(f"[ckpt] saved: {path}")

            if getattr(args, "eval_val_on_ckpt", False):
                t0 = time.time()
                mval = _eval_val_metrics_denorm(model, val_loader, stats=train_stats, device=device)
                dt = time.time() - t0
                print(
                    "[ckpt-val] "
                    f"ep={epoch:04d} metrics units={mval['units']} n_cases={int(mval['n_cases'])} "
                    f"mean_rmse={mval['mean_rmse']:.8f} min_rmse={mval['min_rmse']:.8f} max_rmse={mval['max_rmse']:.8f} "
                    f"mean_grad={mval['mean_grad']:.8f} max_ae={mval['max_ae']:.8f} mean_rmspe_pct={mval['mean_rmspe_pct']:.4f} "
                    f"time_s={dt:.2f}"
                )

            # QAT/quantization removed: no int8 export.


def _stats_line(tag: str, x: torch.Tensor) -> str:
    # x: (1,H,W) or (H,W) or (1,1,H,W)
    if hasattr(x, "detach"):
        x = x.detach()
    t = x
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    t = t.float()
    return f"{tag} min={float(t.min().item()):.6f} max={float(t.max().item()):.6f} mean={float(t.mean().item()):.6f}"


def main_test(args) -> None:
    device = _device()

    dataset_all = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
    )

    all_cases = dataset_all.cases
    train_cases, val_cases, test_cases = split_cases_by_i(all_cases, seed=args.seed, train_ratio=0.8, val_ratio=0.1)

    train_stats = compute_minmax(dataset_all.data_root, grid_size=dataset_all.temp_grid_size, cases=train_cases)

    split = str(getattr(args, "split", "test"))
    if split == "val":
        eval_cases = val_cases
    else:
        split = "test"
        eval_cases = test_cases

    if args.limit_test and args.limit_test > 0:
        eval_cases = eval_cases[: int(args.limit_test)]

    eval_set = ThermalDataset(
        thermal_map_rel="Dataset/dataset/output/thermal/thermal_map",
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        stats=train_stats,
        cases=eval_cases,
    )

    loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    mean_calib = bool(ckpt.get("mean_calib", ckpt.get("mean_calibration", False)))
    if "disable_mean_calib" in ckpt:
        mean_calib = not bool(ckpt.get("disable_mean_calib"))

    model = ThermalGuidanceHRNet(
        base=int(ckpt.get("base", 32)),
        stages=int(ckpt.get("stages", 4)),
        blocks_per_stage=int(ckpt.get("blocks_per_stage", 2)),
        expand_ratio=int(ckpt.get("expand_ratio", 2)),
        mean_calib=mean_calib,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    st = _maybe_temp_stats(ckpt)
    if st is None:
        st = _Stats(temp_min=float(train_stats.temp_min), temp_max=float(train_stats.temp_max))

    print(
        "[debug_scale] "
        f"split={split} ckpt={args.ckpt} "
        f"temp_min={st.temp_min:.6f} temp_max={st.temp_max:.6f}"
    )

    out_dir = os.path.join(os.path.dirname(__file__), "test_result")
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, f"{split}_hrnet_fig")
    os.makedirs(fig_dir, exist_ok=True)

    if hasattr(args, "out_fig_dir") and args.out_fig_dir:
        fig_dir = args.out_fig_dir
        os.makedirs(fig_dir, exist_ok=True)

    mse_sum = 0.0
    rmse_sum = 0.0
    rmse_min = float("inf")
    rmse_max = 0.0
    grad_sum = 0.0
    max_ae = 0.0
    n = 0

    block_size = 100
    in_block = 0
    block_idx = 0
    worst = None

    def flush() -> None:
        nonlocal worst, block_idx
        if worst is None:
            return
        i = int(worst["i"])
        j = int(worst["j"])
        score = float(worst["rmse"])

        flp = os.path.join(
            _PROJECT_ROOT,
            "Dataset/dataset/output/thermal/hotspot_config",
            f"system_{i}_config",
            "system.flp",
        )

        pred = worst["pred"]
        gt = worst["gt"]
        units = "C" if st is not None else "norm"

        vmin = float(min(pred.min().item(), gt.min().item()))
        vmax = float(max(pred.max().item(), gt.max().item()))

        try:
            print(
                "[debug_plot] "
                f"block={block_idx:03d} i={i} j={j} rmse={score:.6f} units={units} vmin={vmin:.6f} vmax={vmax:.6f} "
                + _stats_line(f"pred_{units}", pred)
                + " "
                + _stats_line(f"gt_{units}", gt)
            )
        except Exception:
            pass

        plot_thermal_grid_overlay(
            flp,
            pred,
            os.path.join(fig_dir, f"hrnet_block{block_idx:03d}_i{i}_j{j}_pred_rmse{score:.6f}.png"),
            title=f"Pred block {block_idx} i={i} j={j} RMSE={score:.6f}",
            vmin=vmin,
            vmax=vmax,
        )
        plot_thermal_grid_overlay(
            flp,
            gt,
            os.path.join(fig_dir, f"hrnet_block{block_idx:03d}_i{i}_j{j}_gt.png"),
            title=f"GT block {block_idx} i={i} j={j}",
            vmin=vmin,
            vmax=vmax,
        )

        worst = None

    with torch.no_grad():
        for batch in loader:
            totalp = batch.get("total_power")

            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = totalp.to(device) if totalp is not None else None

            pred_grid, _pred_avg = model(power, layout, totalp)

            # Debug: confirm model/dataset are still normalized (0..1)
            if n == 0:
                try:
                    print(
                        "[debug_scale_batch0] "
                        + _stats_line("pred01", pred_grid.detach().cpu())
                        + " "
                        + _stats_line("gt01", temp.detach().cpu())
                    )
                except Exception:
                    pass

            pred_eval_norm = pred_grid.detach().cpu()
            temp_eval_norm = temp.detach().cpu()

            pred_eval = pred_eval_norm
            temp_eval = temp_eval_norm
            if st is not None:
                pred_eval = _denorm_temp_c(pred_eval, st)
                temp_eval = _denorm_temp_c(temp_eval, st)

            if n == 0 and st is not None:
                try:
                    print(
                        "[debug_rmse_batch0] "
                        + _stats_line("pred_C", pred_eval)
                        + " "
                        + _stats_line("gt_C", temp_eval)
                    )
                except Exception:
                    pass

            diff = pred_eval - temp_eval
            mse = torch.mean(diff * diff, dim=(1, 2, 3))
            rmse = torch.sqrt(mse)
            grad = spatial_gradient_loss(pred_eval, temp_eval)

            batch_max_ae = float(diff.abs().max().item())
            max_ae = max(max_ae, batch_max_ae)

            for bb in range(mse.shape[0]):
                r = float(rmse[bb].item())
                mse_sum += float(mse[bb].item())
                rmse_sum += r
                rmse_min = min(rmse_min, r)
                rmse_max = max(rmse_max, r)
                grad_sum += float(grad.item())
                n += 1

                if worst is None or r > float(worst["rmse"]):
                    worst = {
                        "rmse": r,
                        "i": int(batch["i"][bb]),
                        "j": int(batch["j"][bb]),
                        # Store denormalized grids once; plotting uses the same scale as RMSE.
                        "pred": pred_eval[bb],
                        "gt": temp_eval[bb],
                    }

                in_block += 1
                if in_block >= block_size:
                    flush()
                    block_idx += 1
                    in_block = 0

    if worst is not None:
        flush()

    units = "C" if st is not None else "norm"
    mean_rmse = float(rmse_sum / max(n, 1))
    print(f"==== HRNet {split.upper()} Metrics ({units}) ====")
    print(f"mse:  {mse_sum / max(n, 1):.8f}")
    print(f"rmse: {mean_rmse:.8f}")
    print(f"grad: {grad_sum / max(n, 1):.8f}")
    print(f"max_ae: {max_ae:.8f}")
    print(f"fig_dir: {fig_dir}")
    print(
        "[metrics] "
        f"split={split} units={units} n_cases={int(n)} "
        f"mean_rmse={mean_rmse:.8f} min_rmse={float(rmse_min if n>0 else 0.0):.8f} max_rmse={float(rmse_max):.8f} "
        f"max_ae={float(max_ae):.8f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch_size", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--base", type=int, default=32)

    t.add_argument("--stages", type=int, default=4)
    t.add_argument("--blocks_per_stage", type=int, default=2)
    t.add_argument("--expand_ratio", type=int, default=2)

    t.add_argument("--grad_w", type=float, default=0.1)
    t.add_argument("--avg_w", type=float, default=0.1)
    t.add_argument("--mean_consistency_w", type=float, default=0.1)

    t.add_argument("--under_w", type=float, default=1.0)
    t.add_argument("--hotspot_mode", type=str, default="linear", choices=["linear", "exp", "pow"])
    t.add_argument("--hotspot_alpha", type=float, default=3.0)
    t.add_argument("--hotspot_beta", type=float, default=3.0)
    t.add_argument("--hotspot_pow", type=float, default=4.0)
    t.add_argument("--maxpool_w", type=float, default=0.0)
    t.add_argument("--maxpool_ks", type=int, default=4)

    t.add_argument("--topk_w", type=float, default=0.0)
    t.add_argument("--topk_k", type=int, default=0)
    t.add_argument("--peak_w", type=float, default=0.0)

    t.add_argument("--disable_mean_calib", action="store_true")
    t.add_argument("--eval_val_on_ckpt", action="store_true")
    t.add_argument("--ckpt_every", type=int, default=1)
    t.add_argument("--print_every", type=int, default=10)
    # QAT/quantization removed (kept arg for backward compatibility; now errors out)
    t.add_argument("--qat", action="store_true", help=argparse.SUPPRESS)
    t.add_argument("--ckpt_tag", type=str, default="")
    t.add_argument("--out_dir", type=str, default="")
    t.add_argument("--resume_ckpt", type=str, default="")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--amp", action="store_true")
    t.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    # dataset slicing for quick experiments
    t.add_argument("--limit_train", type=int, default=0, help="if >0, only use first N train cases")
    t.add_argument("--limit_val", type=int, default=0, help="if >0, only use first N val cases")

    te = sub.add_parser("test")
    te.add_argument("--ckpt", type=str, required=True)
    te.add_argument("--batch_size", type=int, default=8)
    te.add_argument("--out_fig_dir", type=str, default="")
    te.add_argument("--seed", type=int, default=0)
    te.add_argument("--limit_test", type=int, default=0, help="if >0, only use first N eval cases")
    te.add_argument("--split", type=str, default="test", choices=["test", "val"], help="which split to evaluate")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.cmd == "train":
        main_train(args)
    elif args.cmd == "test":
        main_test(args)