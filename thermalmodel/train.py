"""Train ThermalGuidanceHRNet (concise entrypoint).

Example:
    python train.py --epochs 200 --batch_size 32 --lr 2e-4 --base 32
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from thermalmodel.HRNet import ThermalGuidanceHRNet, guidance_loss
from thermalmodel.dataLoader import MinMaxStats, ThermalDataset, compute_minmax, split_cases_by_i

DATA = "Dataset/dataset/thermal_dataset"
CFG = "Dataset/dataset/thermal_dataset/config"
GRID = 128
_DUMMY = MinMaxStats(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)  # 仅用于跳过全量 minmax


def make_dataset(cases, stats=None) -> ThermalDataset:
    return ThermalDataset(DATA, CFG, stats=stats, cases=cases)


def build_model(args) -> ThermalGuidanceHRNet:
    return ThermalGuidanceHRNet(
        base=args.base,
        stages=args.stages,
        blocks_per_stage=args.blocks_per_stage,
        expand_ratio=args.expand_ratio,
    )


def loss_args(args) -> dict:
    return dict(
        grad_w=args.grad_w,
        avg_w=args.avg_w,
        mean_consistency_w=args.mean_consistency_w,
        under_w=args.under_w,
        peak_w=args.peak_w,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--blocks_per_stage", type=int, default=2)
    ap.add_argument("--expand_ratio", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad_w", type=float, default=0.1)
    ap.add_argument("--avg_w", type=float, default=0.1)
    ap.add_argument("--mean_consistency_w", type=float, default=0.1)
    ap.add_argument("--under_w", type=float, default=1.0)
    ap.add_argument("--peak_w", type=float, default=0.0)
    ap.add_argument("--ckpt_every", type=int, default=5)
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # splits + normalization stats (from train split only)
    all_ds = ThermalDataset(DATA, CFG, stats=_DUMMY)  # 只取 cases + data_root
    train_cases, val_cases, _ = split_cases_by_i(all_ds.cases, seed=args.seed)
    if args.limit_train:
        train_cases = train_cases[: args.limit_train]
    if args.limit_val:
        val_cases = val_cases[: args.limit_val]
    stats = compute_minmax(all_ds.data_root, grid_size=GRID, cases=train_cases)

    train_set = make_dataset(train_cases, stats=stats)
    val_set = make_dataset(val_cases, stats=stats)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] {args.resume} -> start_epoch={start_epoch}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print(f"device={device} train={len(train_set)} val={len(val_set)} "
          f"base={args.base} stages={args.stages} lr={args.lr}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        for it, batch in enumerate(train_loader):
            power = batch["power"].to(device)
            layout = batch["layout"].to(device)
            temp = batch["temp"].to(device)
            totalp = batch["total_power"].to(device)
            avg = batch["avg_temp"].to(device)

            opt.zero_grad(set_to_none=True)
            pred, pred_avg = model(power, layout, totalp)
            loss, m = guidance_loss(pred, temp, pred_avg=pred_avg, target_avg=avg, **loss_args(args))
            loss.backward()
            opt.step()

            if it % args.print_every == 0:
                print(f"[train] ep{epoch:04d} it{it:04d} "
                      f"loss={m['loss']:.6f} mse={m['mse']:.6f} grad={m['grad']:.6f}")

        # validation loss
        model.eval()
        vm = {k: 0.0 for k in ("loss", "mse", "grad", "avg_mse", "mean_cons")}
        with torch.no_grad():
            for batch in val_loader:
                power = batch["power"].to(device)
                layout = batch["layout"].to(device)
                temp = batch["temp"].to(device)
                totalp = batch["total_power"].to(device)
                avg = batch["avg_temp"].to(device)
                pred, pred_avg = model(power, layout, totalp)
                _, m = guidance_loss(pred, temp, pred_avg=pred_avg, target_avg=avg, **loss_args(args))
                for k in vm:
                    vm[k] += m.get(k, 0.0)
        n = max(len(val_loader), 1)
        vm = {k: v / n for k, v in vm.items()}
        print(f"[val] ep{epoch:04d} loss={vm['loss']:.6f} mse={vm['mse']:.6f} grad={vm['grad']:.6f} "
              f"avg_mse={vm['avg_mse']:.6f}")

        if epoch % args.ckpt_every == 0:
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "stats": stats.to_dict(),
                "grid_size": GRID,
                "base": args.base,
                "stages": args.stages,
                "blocks_per_stage": args.blocks_per_stage,
                "expand_ratio": args.expand_ratio,
                "lr": args.lr,
                "seed": args.seed,
                "batch_size": args.batch_size,
            }
            name = f"hrnet_base{args.base}_seed{args.seed}_ep{epoch:04d}.pth"
            path = os.path.join(out_dir, name)
            torch.save(ckpt, path)
            print(f"[ckpt] saved {path}")


if __name__ == "__main__":
    main()
