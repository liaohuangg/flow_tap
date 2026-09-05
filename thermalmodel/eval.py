"""Evaluate a ThermalGuidanceHRNet checkpoint (concise entrypoint).

Example:
    python eval.py --ckpt checkpoints/hrnet_base32_seed0_ep0200.pth --split test --topk 50
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from thermalmodel.HRNet import ThermalGuidanceHRNet, _spatial_gradient_abs_metric
from thermalmodel.dataLoader import ThermalDataset, MinMaxStats, compute_minmax, split_cases_by_i
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay

DATA = "Dataset/dataset/thermal_dataset"
CFG = "Dataset/dataset/thermal_dataset/config"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0, help="export top-k best and worst figures")
    ap.add_argument("--fig_dir", type=str, default="")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = ThermalGuidanceHRNet(
        base=int(ckpt.get("base", 32)),
        stages=int(ckpt.get("stages", 4)),
        blocks_per_stage=int(ckpt.get("blocks_per_stage", 2)),
        expand_ratio=int(ckpt.get("expand_ratio", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    seed = int(ckpt.get("seed", 0))
    all_ds = ThermalDataset(DATA, CFG, stats=MinMaxStats(0.0, 1.0, 0.0, 1.0, 0.0, 1.0))  # 只取 cases
    train_cases, val_cases, test_cases = split_cases_by_i(all_ds.cases, seed=seed)
    cases = val_cases if args.split == "val" else test_cases
    if args.limit:
        cases = cases[: args.limit]

    stats = ckpt.get("stats")
    if isinstance(stats, dict):
        stats = MinMaxStats(**stats)
    else:
        stats = compute_minmax(all_ds.data_root, grid_size=128, cases=train_cases)

    eval_set = ThermalDataset(DATA, CFG, stats=stats, cases=cases)
    loader = torch.utils.data.DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    tmin, tmax = stats.temp_min, stats.temp_max
    denorm = lambda x: x * (tmax - tmin) + tmin

    n = 0
    rmse_sum, rmse_min, rmse_max = 0.0, float("inf"), 0.0
    mae_sum, mae_max = 0.0, 0.0
    mape_sum = 0.0
    grad_sum = 0.0
    max_ae = 0.0
    ranked = []  # (rmse, i, j, pred_c, gt_c) if --topk

    with torch.no_grad():
        for batch in loader:
            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            totalp = batch["total_power"].to(device)
            temp = batch["temp"].to(device)

            pred, _ = model(power, layout, totalp)
            pred_c = denorm(pred.cpu())
            gt_c = denorm(temp.cpu())

            diff = pred_c - gt_c
            rmse = diff.square().mean(dim=(1, 2, 3)).sqrt()
            mae = diff.abs().mean(dim=(1, 2, 3))
            mape = (diff.abs() / (gt_c.abs() + 1e-6)).mean(dim=(1, 2, 3)) * 100.0

            grad_sum += float(_spatial_gradient_abs_metric(pred_c, gt_c).item())
            max_ae = max(max_ae, float(diff.abs().max().item()))

            for b in range(pred.shape[0]):
                r = float(rmse[b])
                n += 1
                rmse_sum += r
                rmse_min = min(rmse_min, r)
                rmse_max = max(rmse_max, r)
                mae_sum += float(mae[b])
                mae_max = max(mae_max, float(mae[b]))
                mape_sum += float(mape[b])
                if args.topk:
                    ranked.append((r, int(batch["i"][b]), int(batch["j"][b]), pred_c[b], gt_c[b]))

    print(f"==== {args.split.upper()} metrics (C, n={n}) ====")
    print(f"mean_rmse={rmse_sum / n:.6f}  min_rmse={rmse_min:.6f}  max_rmse={rmse_max:.6f}")
    print(f"mean_mae={mae_sum / n:.6f}  max_mae={mae_max:.6f}  mean_mape%={mape_sum / n:.4f}")
    print(f"max_ae={max_ae:.6f}  mean_grad={grad_sum / max(n, 1):.6f}")

    if args.topk and ranked:
        ranked.sort()
        fig_dir = args.fig_dir or os.path.join(os.path.dirname(__file__), "figs", args.split)
        os.makedirs(fig_dir, exist_ok=True)
        flp_root = all_ds.hotspot_root
        for tag, items in (("best", ranked[: args.topk]), ("worst", ranked[-args.topk:])):
            for rmse, i, j, pred_c, gt_c in items:
                flp = os.path.join(flp_root, f"system_{i}_config", "system.flp")
                for name, grid in (("pred", pred_c), ("gt", gt_c)):
                    plot_thermal_grid_overlay(
                        flp, grid,
                        os.path.join(fig_dir, f"{tag}_i{i}_j{j}_{name}_rmse{rmse:.6f}.png"),
                        title=f"{tag} i={i} j={j} {name} RMSE={rmse:.6f}",
                        vmin=float(pred_c.min()), vmax=float(pred_c.max()),
                    )
        print(f"figures -> {fig_dir}")


if __name__ == "__main__":
    main()
