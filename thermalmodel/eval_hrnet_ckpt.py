import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Allow running `python eval_hrnet_ckpt.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.dataLoader import ThermalDataset, MinMaxStats, compute_minmax, split_cases_by_i
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay
from thermalmodel.HRNet import ThermalGuidanceHRNet


def _device(prefer: str = "cuda") -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_temp_stats(ckpt: Dict[str, Any]) -> Optional[Dict[str, float]]:
    st = ckpt.get("stats")
    if isinstance(st, dict) and ("temp_min" in st) and ("temp_max" in st):
        try:
            return {"temp_min": float(st["temp_min"]), "temp_max": float(st["temp_max"])}
        except Exception:
            return None
    return None


def _denorm_temp_c(t01: torch.Tensor, st: Dict[str, float]) -> torch.Tensor:
    return t01 * (st["temp_max"] - st["temp_min"]) + st["temp_min"]


def _spatial_gradient_abs_metric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    # pred/gt: (B,1,H,W)
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)

    px = F.conv2d(pred, sobel_x, padding=1)
    py = F.conv2d(pred, sobel_y, padding=1)
    gx = F.conv2d(gt, sobel_x, padding=1)
    gy = F.conv2d(gt, sobel_y, padding=1)
    return torch.mean(torch.abs(px - gx) + torch.abs(py - gy))


@dataclass
class Metrics:
    units: str
    n_cases: int
    mean_rmse: float
    min_rmse: float
    max_rmse: float
    mean_mae: float
    max_mae: float
    mean_mse: float

    # Existing percentage metric (per-pixel |e|/|gt| averaged, *100)
    mean_mape_pct: float

    # New metrics (normalize by per-sample mean(gt))
    mean_abs_rel: float
    mean_abs_rel_pct: float
    mean_rel: float
    mean_rel_pct: float

    # Peak temperature metrics (per-sample max over grid)
    mean_peak_ae: float
    mean_peak_abs_rel: float
    mean_peak_abs_rel_pct: float
    mean_peak_rel: float
    mean_peak_rel_pct: float

    mean_grad: float
    max_ae: float


def _extract_ckpt_meta(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "epoch",
        "base",
        "batch_size",
        "lr",
        "grad_w",
        "avg_w",
        "mean_consistency_w",
        "under_w",
        "hotspot_mode",
        "hotspot_alpha",
        "hotspot_beta",
        "hotspot_pow",
        "maxpool_w",
        "maxpool_ks",
        "topk_w",
        "topk_k",
        "peak_w",
        "seed",
        "ckpt_tag",
        "stages",
        "blocks_per_stage",
        "expand_ratio",
        "limit_train",
        "limit_val",
        "grid_size",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in ckpt:
            v = ckpt.get(k)
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[k] = v
            else:
                out[k] = str(v)
    return out


@torch.no_grad()
def _tensor_stats_2d01(x: torch.Tensor) -> Dict[str, float]:
    # x: (B,1,H,W) or (1,H,W) or (H,W)
    if hasattr(x, "detach"):
        x = x.detach()
    t = x
    if t.dim() == 4:
        t = t[:, 0]  # (B,H,W)
    if t.dim() == 3 and t.shape[0] == 1:
        t = t[0]
    t = t.float()
    return {
        "min": float(t.min().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
    }


def _format_stats(tag: str, st: Dict[str, float]) -> str:
    return f"{tag} min={st['min']:.6f} max={st['max']:.6f} mean={st['mean']:.6f}"


def eval_hrnet_ckpt(
    ckpt_path: str,
    *,
    split: str,
    batch_size: int,
    seed: int,
    prefer_device: str,
    out_fig_dir: str,
    topk: int,
    limit_cases: int,
) -> Tuple[Metrics, Dict[str, Any]]:
    device = _device(prefer=prefer_device)

    dataset_all = ThermalDataset(
        thermal_map_rel="Dataset/dataset/thermal_dataset_64",
        hotspot_cfg_rel="Dataset/dataset/thermal_dataset_64/config",
        power_grid_size=64,
        temp_grid_size=64,
        stats=MinMaxStats(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),  # skip full minmax; recomputed on train split below
    )

    all_cases = dataset_all.cases
    train_cases, val_cases, test_cases = split_cases_by_i(all_cases, seed=seed, train_ratio=0.8, val_ratio=0.1)
    train_stats = compute_minmax(dataset_all.data_root, grid_size=dataset_all.temp_grid_size, cases=train_cases)

    if split == "val":
        eval_cases = val_cases
    elif split == "test":
        eval_cases = test_cases
    else:
        raise ValueError(f"Unknown split={split!r}; expected 'val' or 'test'")

    if limit_cases and limit_cases > 0:
        eval_cases = eval_cases[: int(limit_cases)]

    eval_set = ThermalDataset(
        thermal_map_rel="Dataset/dataset/thermal_dataset_64",
        hotspot_cfg_rel="Dataset/dataset/thermal_dataset_64/config",
        power_grid_size=64,
        temp_grid_size=64,
        stats=train_stats,
        cases=eval_cases,
    )

    loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=0)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    base = int(ckpt.get("base", 64))
    stages = int(ckpt.get("stages", 4))
    blocks_per_stage = int(ckpt.get("blocks_per_stage", 2))
    expand_ratio = int(ckpt.get("expand_ratio", 2))

    model = ThermalGuidanceHRNet(
        base=base,
        stages=stages,
        blocks_per_stage=blocks_per_stage,
        expand_ratio=expand_ratio,
        mean_calib=bool(ckpt.get("mean_calib", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    st = _maybe_temp_stats(ckpt)
    if st is None:
        st = {"temp_min": float(train_stats.temp_min), "temp_max": float(train_stats.temp_max)}

    units = "C"

    print(
        "[debug_scale] "
        f"split={split} ckpt={ckpt_path} "
        f"temp_min={st['temp_min']:.6f} temp_max={st['temp_max']:.6f}"
    )

    eps = 1e-6

    # Aggregates
    rmse_sum = 0.0
    rmse_min = float("inf")
    rmse_max = 0.0

    mae_sum = 0.0
    mae_max = 0.0

    mse_sum = 0.0
    mape_sum = 0.0

    # New metrics: normalize by per-sample mean(gt)
    abs_rel_sum = 0.0
    rel_sum = 0.0

    # Peak metrics
    peak_ae_sum = 0.0
    peak_abs_rel_sum = 0.0
    peak_rel_sum = 0.0

    grad_sum = 0.0
    max_ae = 0.0

    n = 0

    # save best/worst/topk
    entries: List[Dict[str, Any]] = []

    for batch in loader:
        totalp = batch.get("total_power")

        power = batch["power"].to(device)
        layout = batch["layout"].to(device)
        temp = batch["temp"].to(device)
        totalp = totalp.to(device) if totalp is not None else None

        pred01, _pred_avg = model(power, layout, totalp)

        if n == 0:
            try:
                p01s = _tensor_stats_2d01(pred01.detach().cpu())
                g01s = _tensor_stats_2d01(temp.detach().cpu())
                print("[debug_scale_batch0] " + _format_stats("pred01", p01s) + " " + _format_stats("gt01", g01s))
            except Exception:
                pass

        pred_c = _denorm_temp_c(pred01.detach().cpu(), st)
        gt_c = _denorm_temp_c(temp.detach().cpu(), st)

        if n == 0:
            try:
                pcs = _tensor_stats_2d01(pred_c)
                gcs = _tensor_stats_2d01(gt_c)
                print("[debug_scale_batch0_denorm] " + _format_stats("pred_C", pcs) + " " + _format_stats("gt_C", gcs))
            except Exception:
                pass

        diff = pred_c - gt_c

        per_mse = torch.mean(diff * diff, dim=(1, 2, 3))
        per_rmse = torch.sqrt(per_mse)
        per_mae = torch.mean(torch.abs(diff), dim=(1, 2, 3))
        per_mape = torch.mean(torch.abs(diff) / (torch.abs(gt_c) + eps), dim=(1, 2, 3)) * 100.0

        # New: per-sample mean(gt) normalized errors
        per_gt_mean = torch.mean(gt_c, dim=(1, 2, 3))
        per_abs_rel = per_mae / (per_gt_mean + eps)  # mean(|e|)/mean(gt)
        per_rel = torch.mean(diff, dim=(1, 2, 3)) / (per_gt_mean + eps)  # mean(e)/mean(gt)

        # Peak metrics (evaluate prediction at the GT peak location)
        # Find argmax on GT grid, then compare values at that same location.
        gt_flat = gt_c.view(gt_c.shape[0], -1)
        pred_flat = pred_c.view(pred_c.shape[0], -1)
        peak_idx = torch.argmax(gt_flat, dim=1)  # (B,)
        # Convert flat idx to (row, col) in HxW
        h_gt, w_gt = int(gt_c.shape[2]), int(gt_c.shape[3])
        peak_r = torch.div(peak_idx, w_gt, rounding_mode="floor")
        peak_c = peak_idx % w_gt
        per_gt_peak = gt_flat.gather(1, peak_idx.view(-1, 1)).squeeze(1)
        per_pred_peak = pred_flat.gather(1, peak_idx.view(-1, 1)).squeeze(1)
        per_peak_ae = torch.abs(per_pred_peak - per_gt_peak)
        per_peak_abs_rel = per_peak_ae / (per_gt_peak + eps)
        per_peak_rel = (per_pred_peak - per_gt_peak) / (per_gt_peak + eps)

        grad = _spatial_gradient_abs_metric(pred_c, gt_c)
        grad_sum += float(grad.item())

        batch_max_ae = float(torch.max(torch.abs(diff)).item())
        max_ae = max(max_ae, batch_max_ae)

        for b in range(per_rmse.shape[0]):
            r = float(per_rmse[b].item())
            m = float(per_mse[b].item())
            a = float(per_mae[b].item())
            p = float(per_mape[b].item())
            ar = float(per_abs_rel[b].item())
            rr = float(per_rel[b].item())
            pk_ae = float(per_peak_ae[b].item())
            pk_ar = float(per_peak_abs_rel[b].item())
            pk_rr = float(per_peak_rel[b].item())
            pk_r = int(peak_r[b].item())
            pk_c = int(peak_c[b].item())
            pk_gt = float(per_gt_peak[b].item())
            pk_pred = float(per_pred_peak[b].item())

            rmse_sum += r
            mse_sum += m
            mae_sum += a
            mape_sum += p
            abs_rel_sum += ar
            rel_sum += rr
            peak_ae_sum += pk_ae
            peak_abs_rel_sum += pk_ar
            peak_rel_sum += pk_rr

            rmse_min = min(rmse_min, r)
            rmse_max = max(rmse_max, r)
            mae_max = max(mae_max, a)

            entries.append(
                {
                    "rmse": r,
                    "mse": m,
                    "mae": a,
                    "mape_pct": p,
                    "abs_rel": ar,
                    "rel": rr,
                    "peak_ae": pk_ae,
                    "peak_abs_rel": pk_ar,
                    "peak_rel": pk_rr,
                    "gt_peak_r": pk_r,
                    "gt_peak_c": pk_c,
                    "gt_peak": pk_gt,
                    "pred_at_gt_peak": pk_pred,
                    "i": int(batch["i"][b]),
                    "j": int(batch["j"][b]),
                    "pred": pred_c[b],
                    "gt": gt_c[b],
                }
            )
            n += 1

    mean_rmse = float(rmse_sum / max(n, 1))
    mean_mse = float(mse_sum / max(n, 1))
    mean_mae = float(mae_sum / max(n, 1))
    mean_mape = float(mape_sum / max(n, 1))

    mean_abs_rel = float(abs_rel_sum / max(n, 1))
    mean_rel = float(rel_sum / max(n, 1))

    mean_peak_ae = float(peak_ae_sum / max(n, 1))
    mean_peak_abs_rel = float(peak_abs_rel_sum / max(n, 1))
    mean_peak_rel = float(peak_rel_sum / max(n, 1))

    mean_grad = float(grad_sum / max(len(loader), 1))

    metrics = Metrics(
        units=units,
        n_cases=int(n),
        mean_rmse=mean_rmse,
        min_rmse=float(rmse_min if n > 0 else 0.0),
        max_rmse=float(rmse_max),
        mean_mae=mean_mae,
        max_mae=float(mae_max),
        mean_mse=mean_mse,
        mean_mape_pct=mean_mape,
        mean_abs_rel=mean_abs_rel,
        mean_abs_rel_pct=mean_abs_rel * 100.0,
        mean_rel=mean_rel,
        mean_rel_pct=mean_rel * 100.0,
        mean_peak_ae=mean_peak_ae,
        mean_peak_abs_rel=mean_peak_abs_rel,
        mean_peak_abs_rel_pct=mean_peak_abs_rel * 100.0,
        mean_peak_rel=mean_peak_rel,
        mean_peak_rel_pct=mean_peak_rel * 100.0,
        mean_grad=mean_grad,
        max_ae=float(max_ae),
    )

    # Pick best/worst or topk
    entries.sort(key=lambda d: float(d["rmse"]))
    k = int(topk) if topk is not None else 0

    best_list = entries[: max(1, k)] if entries else []
    worst_list = list(reversed(entries[-max(1, k) :])) if entries else []

    meta: Dict[str, Any] = {
        "best": [{kk: vv for kk, vv in e.items() if kk not in ("pred", "gt")} for e in best_list],
        "worst": [{kk: vv for kk, vv in e.items() if kk not in ("pred", "gt")} for e in worst_list],
        "temp_min": float(st["temp_min"]),
        "temp_max": float(st["temp_max"]),
    }

    if out_fig_dir:
        os.makedirs(out_fig_dir, exist_ok=True)

        def _save_one(tag: str, rank: int, e: Dict[str, Any]) -> None:
            i = int(e["i"])
            j = int(e["j"])
            rmse = float(e["rmse"])
            mae = float(e["mae"])
            mape = float(e["mape_pct"])
            abs_rel = float(e.get("abs_rel", 0.0))
            rel = float(e.get("rel", 0.0))
            peak_ae = float(e.get("peak_ae", 0.0))
            peak_abs_rel = float(e.get("peak_abs_rel", 0.0))
            peak_rel = float(e.get("peak_rel", 0.0))
            peak_r = int(e.get("gt_peak_r", -1))
            peak_c = int(e.get("gt_peak_c", -1))
            gt_peak = float(e.get("gt_peak", 0.0))
            pred_at_peak = float(e.get("pred_at_gt_peak", 0.0))

            flp = os.path.join(
                _PROJECT_ROOT,
                "Dataset/dataset/thermal_dataset_64/config",
                f"system_{i}_config",
                "system.flp",
            )

            pred_w = e["pred"]
            gt_w = e["gt"]

            vmin = float(min(pred_w.min().item(), gt_w.min().item()))
            vmax = float(max(pred_w.max().item(), gt_w.max().item()))

            try:
                ps = _tensor_stats_2d01(pred_w)
                gs = _tensor_stats_2d01(gt_w)
                print(
                    "[debug_plot] "
                    f"tag={tag} r={rank:02d} i={i} j={j} units={units} rmse={rmse:.6f} mae={mae:.6f} "
                    f"abs_rel={abs_rel:.6f} rel={rel:.6f} "
                    f"peak_ae={peak_ae:.6f} peak_abs_rel={peak_abs_rel:.6f} peak_rel={peak_rel:.6f} "
                    f"GTpeak(r={peak_r},c={peak_c}) GTpeak={gt_peak:.2f}{units} Pred@GTpeak={pred_at_peak:.2f}{units} "
                    f"vmin={vmin:.6f} vmax={vmax:.6f} "
                    + _format_stats("pred", ps)
                    + " "
                    + _format_stats("gt", gs)
                )
            except Exception:
                pass

            pred_path = os.path.join(out_fig_dir, f"{tag}_r{rank:02d}_i{i}_j{j}_pred_rmse{rmse:.6f}_mae{mae:.6f}.png")
            gt_path = os.path.join(out_fig_dir, f"{tag}_r{rank:02d}_i{i}_j{j}_gt.png")

            plot_thermal_grid_overlay(
                flp,
                pred_w,
                pred_path,
                title=(
                    f"Pred {tag} r={rank:02d} i={i} j={j} "
                    f"RMSE={rmse:.6f} MAE={mae:.6f} MAPE%={mape:.4f} "
                    f"AbsRel%={abs_rel*100.0:.4f} Rel%={rel*100.0:.4f} "
                    f"PeakAE={peak_ae:.6f} PeakAbsRel%={peak_abs_rel*100.0:.4f} PeakRel%={peak_rel*100.0:.4f} "
                    f"GTpeak(r={peak_r},c={peak_c}) GTpeak={gt_peak:.2f}{units} Pred@GTpeak={pred_at_peak:.2f}{units}"
                ),
                vmin=vmin,
                vmax=vmax,
            )
            plot_thermal_grid_overlay(
                flp,
                gt_w,
                gt_path,
                title=f"GT {tag} r={rank:02d} i={i} j={j} GTpeak(r={peak_r},c={peak_c})",
                vmin=vmin,
                vmax=vmax,
            )

        for idx, e in enumerate(best_list, start=1):
            _save_one("best", idx, e)
        for idx, e in enumerate(worst_list, start=1):
            _save_one("worst", idx, e)

    return metrics, meta


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_fig_dir", type=str, default="")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--topk", type=int, default=1, help="Save top-k best and top-k worst cases by per-sample RMSE")
    ap.add_argument("--limit_cases", type=int, default=0, help="If >0, only evaluate first N cases in split")
    ap.add_argument("--out_log", type=str, default="", help="If set, write full log to this path")
    ap.add_argument("--append", action="store_true", help="append to --out_log instead of overwrite")
    return ap


def _write_log(*, out_log: str, append: bool, args: argparse.Namespace, metrics: Metrics, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_log), exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_meta = _extract_ckpt_meta(ckpt)

    lines: List[str] = []
    lines.append("# hrnet checkpoint eval")
    lines.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("env=conda:chipdiffusion")
    lines.append(f"project_root={_PROJECT_ROOT}")
    lines.append("")

    lines.append("[eval_params]")
    lines.append(f"ckpt={args.ckpt}")
    lines.append(f"split={args.split}")
    lines.append(f"seed={args.seed}")
    lines.append(f"eval_bs={args.eval_bs}")
    lines.append(f"device={args.device}")
    lines.append(f"limit_cases={args.limit_cases}")
    lines.append(f"topk={args.topk}")
    lines.append(f"out_fig_dir={args.out_fig_dir}")
    lines.append("")

    lines.append("[ckpt_meta]")
    lines.append(str(ckpt_meta))
    lines.append("")
    lines.append("[debug_scale]")
    if "temp_min" in meta and "temp_max" in meta:
        lines.append(f"temp_min={float(meta['temp_min']):.6f} temp_max={float(meta['temp_max']):.6f}")
    lines.append(f"units={metrics.units}")
    lines.append("")

    lines.append("[metrics]")
    lines.append(
        " ".join(
            [
                f"units={metrics.units}",
                f"n_cases={metrics.n_cases}",
                f"Mean_RMSE_C={metrics.mean_rmse:.6f}",
                f"Min_RMSE_C={metrics.min_rmse:.6f}",
                f"Max_RMSE_C={metrics.max_rmse:.6f}",
                f"Mean_MAE_C={metrics.mean_mae:.6f}",
                f"Max_MAE_C={metrics.max_mae:.6f}",
                f"Mean_MSE_C2={metrics.mean_mse:.6f}",
                f"Mean_MAPE_pct={metrics.mean_mape_pct:.6f}",
                f"Mean_AbsRel={metrics.mean_abs_rel:.6f}",
                f"Mean_AbsRel_pct={metrics.mean_abs_rel_pct:.6f}",
                f"Mean_Rel={metrics.mean_rel:.6f}",
                f"Mean_Rel_pct={metrics.mean_rel_pct:.6f}",
                f"Mean_Peak_AE_C={metrics.mean_peak_ae:.6f}",
                f"Mean_Peak_AbsRel={metrics.mean_peak_abs_rel:.6f}",
                f"Mean_Peak_AbsRel_pct={metrics.mean_peak_abs_rel_pct:.6f}",
                f"Mean_Peak_Rel={metrics.mean_peak_rel:.6f}",
                f"Mean_Peak_Rel_pct={metrics.mean_peak_rel_pct:.6f}",
                f"mean_grad={metrics.mean_grad:.6f}",
                f"Max_Absolute_Error_C={metrics.max_ae:.6f}",
            ]
        )
    )
    lines.append("")

    best0 = meta.get("best")[0] if isinstance(meta.get("best"), list) and meta.get("best") else {}
    worst0 = meta.get("worst")[0] if isinstance(meta.get("worst"), list) and meta.get("worst") else {}

    lines.append("[best_case_by_rmse]")
    if best0:
        lines.append(
            f"best_i={best0.get('i')} best_j={best0.get('j')} best_rmse={best0.get('rmse')} best_mae={best0.get('mae')} best_mape_pct={best0.get('mape_pct')} "
            f"best_gt_peak_r={best0.get('gt_peak_r')} best_gt_peak_c={best0.get('gt_peak_c')} best_gt_peak={best0.get('gt_peak')} best_pred_at_gt_peak={best0.get('pred_at_gt_peak')}"
        )
    lines.append("")

    lines.append("[worst_case_by_rmse]")
    if worst0:
        lines.append(
            f"worst_i={worst0.get('i')} worst_j={worst0.get('j')} worst_rmse={worst0.get('rmse')} worst_mae={worst0.get('mae')} worst_mape_pct={worst0.get('mape_pct')} "
            f"worst_gt_peak_r={worst0.get('gt_peak_r')} worst_gt_peak_c={worst0.get('gt_peak_c')} worst_gt_peak={worst0.get('gt_peak')} worst_pred_at_gt_peak={worst0.get('pred_at_gt_peak')}"
        )
    lines.append("")

    mode = "a" if append else "w"
    with open(out_log, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write("\n".join(lines))


def main() -> None:
    args = build_argparser().parse_args()

    metrics, meta = eval_hrnet_ckpt(
        args.ckpt,
        split=args.split,
        batch_size=args.eval_bs,
        seed=args.seed,
        prefer_device=args.device,
        out_fig_dir=args.out_fig_dir,
        topk=args.topk,
        limit_cases=args.limit_cases,
    )

    # Always print a one-line summary (for scripts)
    best0 = meta.get("best")[0] if isinstance(meta.get("best"), list) and meta.get("best") else {}
    worst0 = meta.get("worst")[0] if isinstance(meta.get("worst"), list) and meta.get("worst") else {}

    print(
        "metrics"
        f" units={metrics.units}"
        f" n_cases={metrics.n_cases}"
        f" mean_rmse={metrics.mean_rmse:.6f}"
        f" min_rmse={metrics.min_rmse:.6f}"
        f" max_rmse={metrics.max_rmse:.6f}"
        f" mean_mae={metrics.mean_mae:.6f}"
        f" mean_mse={metrics.mean_mse:.6f}"
        f" mean_mape_pct={metrics.mean_mape_pct:.6f}"
        f" mean_abs_rel={metrics.mean_abs_rel:.6f}"
        f" mean_abs_rel_pct={metrics.mean_abs_rel_pct:.6f}"
        f" mean_rel={metrics.mean_rel:.6f}"
        f" mean_rel_pct={metrics.mean_rel_pct:.6f}"
        f" mean_peak_ae={metrics.mean_peak_ae:.6f}"
        f" mean_peak_abs_rel={metrics.mean_peak_abs_rel:.6f}"
        f" mean_peak_abs_rel_pct={metrics.mean_peak_abs_rel_pct:.6f}"
        f" mean_peak_rel={metrics.mean_peak_rel:.6f}"
        f" mean_peak_rel_pct={metrics.mean_peak_rel_pct:.6f}"
        f" max_ae={metrics.max_ae:.6f}"
        f" best_i={best0.get('i','')} best_j={best0.get('j','')} best_rmse={best0.get('rmse','')}"
        f" best_gt_peak_r={best0.get('gt_peak_r','')} best_gt_peak_c={best0.get('gt_peak_c','')}"
        f" best_gt_peak={best0.get('gt_peak','')} best_pred_at_gt_peak={best0.get('pred_at_gt_peak','')}"
        f" worst_i={worst0.get('i','')} worst_j={worst0.get('j','')} worst_rmse={worst0.get('rmse','')}"
        f" worst_gt_peak_r={worst0.get('gt_peak_r','')} worst_gt_peak_c={worst0.get('gt_peak_c','')}"
        f" worst_gt_peak={worst0.get('gt_peak','')} worst_pred_at_gt_peak={worst0.get('pred_at_gt_peak','')}"
        f" ckpt={args.ckpt}"
    )

    if args.out_log:
        _write_log(out_log=args.out_log, append=bool(args.append), args=args, metrics=metrics, meta=meta)
        print(f"WROTE {args.out_log}")


if __name__ == "__main__":
    main()