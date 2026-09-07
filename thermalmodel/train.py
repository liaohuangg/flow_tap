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
    ap.add_argument("--max_cases", type=int, default=0,
                    help="只用前 max_cases 个 case(按 (i,j) 排序)再切分; 0=全部")
    ap.add_argument("--num_workers", type=int, default=0,
                    help="DataLoader 并行加载进程数")
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--compile", action="store_true", help="torch.compile 融合小 kernel 以减少 CPU 派发开销")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # TF32 tensor cores: 明显加速 float32 matmul(标准做法,回归任务数值影响可忽略)
        torch.set_float32_matmul_precision("high")

    # splits + normalization stats (from train split only)
    all_ds = ThermalDataset(DATA, CFG, stats=_DUMMY)  # 只取 cases + data_root
    cases = all_ds.cases
    if args.max_cases > 0:
        cases = cases[: args.max_cases]
    train_cases, val_cases, _ = split_cases_by_i(cases, seed=args.seed)
    if args.limit_train:
        train_cases = train_cases[: args.limit_train]
    if args.limit_val:
        val_cases = val_cases[: args.limit_val]
    stats = compute_minmax(all_ds.data_root, grid_size=GRID, cases=train_cases)

    train_set = make_dataset(train_cases, stats=stats)
    val_set = make_dataset(val_cases, stats=stats)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_orig = build_model(args).to(device)

    # 先加载 resume 权重到未编译的原始模型(避免 torch.compile 的 _orig_mod. 前缀污染 state_dict)
    start_epoch = 1
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state):
            state = {k[len("_orig_mod."):]: v for k, v in state.items()}
        model_orig.load_state_dict(state, strict=False)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[resume] {args.resume} -> start_epoch={start_epoch}", flush=True)

    if args.compile:
        model = torch.compile(model_orig)
        print("[compile] torch.compile enabled", flush=True)
    else:
        model = model_orig

    opt = torch.optim.Adam(model_orig.parameters(), lr=args.lr)
    if args.resume and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    out_dir = args.out_dir or os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print(f"device={device} train={len(train_set)} val={len(val_set)} "
          f"base={args.base} stages={args.stages} lr={args.lr}", flush=True)

    best_val = float("inf")
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
                      f"loss={float(m['loss']):.6f} mse={float(m['mse']):.6f} grad={float(m['grad']):.6f}", flush=True)

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
                    vm[k] += float(m.get(k, 0.0))
        n = max(len(val_loader), 1)
        vm = {k: v / n for k, v in vm.items()}
        print(f"[val] ep{epoch:04d} loss={vm['loss']:.6f} mse={vm['mse']:.6f} grad={vm['grad']:.6f} "
              f"avg_mse={vm['avg_mse']:.6f}", flush=True)

        ckpt = {
            "epoch": epoch,
            "model": model_orig.state_dict(),
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
        if epoch % args.ckpt_every == 0:
            name = f"hrnet_base{args.base}_seed{args.seed}_ep{epoch:04d}.pth"
            path = os.path.join(out_dir, name)
            torch.save(ckpt, path)
            print(f"[ckpt] saved {path}", flush=True)

        # 保存当前 val loss 最优的模型
        if vm["loss"] < best_val:
            best_val = vm["loss"]
            best_path = os.path.join(out_dir, f"hrnet_base{args.base}_seed{args.seed}_best.pth")
            torch.save(ckpt, best_path)
            print(f"[best] ep{epoch:04d} val_loss={best_val:.6f} -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
