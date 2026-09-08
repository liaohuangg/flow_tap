"""fp32_case_time.py

Time the HRNet FP32 checkpoint including:
  1) per-case end-to-end time (includes DataLoader + H2D copy) and
  2) per-case model forward time (excludes DataLoader/H2D).

Writes timing results to a log file.

Usage (chipdiffusion env):
  python fp32_case_time.py --n_cases 1000 --eval_bs 8 --device cuda
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import torch

# Allow running from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _list_first_n_cases(thermal_map_root: str, n: int) -> List[Tuple[int, int]]:
    power_dir = os.path.join(thermal_map_root, "powercsv")
    pat = re.compile(r"system_power_(\d+)_(\d+)\.csv$")
    cases: List[Tuple[int, int]] = []
    for fn in os.listdir(power_dir):
        m = pat.match(fn)
        if not m:
            continue
        cases.append((int(m.group(1)), int(m.group(2))))
    cases.sort()
    return cases[:n]


def _mk_dataset(*, thermal_map_root: str, cases):
    from thermalmodel.dataLoader import ThermalDataset

    return ThermalDataset(
        thermal_map_rel=os.path.relpath(thermal_map_root, start=_PROJECT_ROOT),
        hotspot_cfg_rel="Dataset/dataset/output/thermal/hotspot_config",
        power_grid_size=128,
        temp_grid_size=64,
        cases=cases,
    )


@torch.no_grad()
def time_fp32_ckpt(
    ckpt_path: str,
    *,
    n_cases: int,
    eval_bs: int,
    seed: int,
    device: str,
    thermal_map_root: str,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, float]:
    from thermalmodel.HRNet import ThermalGuidanceHRNet

    torch.manual_seed(seed)

    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = ThermalGuidanceHRNet(
        base=int(ckpt.get("base", 64)),
        stages=int(ckpt.get("stages", 4)),
        blocks_per_stage=int(ckpt.get("blocks_per_stage", 2)),
        expand_ratio=int(ckpt.get("expand_ratio", 2)),
        mean_calib=bool(ckpt.get("mean_calib", False)),
    ).to(dev)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    cases = _list_first_n_cases(thermal_map_root, n_cases)
    ds_t0 = time.perf_counter()
    dataset = _mk_dataset(thermal_map_root=thermal_map_root, cases=cases)
    ds_build_s = time.perf_counter() - ds_t0

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_wall0 = time.perf_counter()
    t_dl_s = 0.0
    t_fwd_s = 0.0
    n_seen = 0

    for batch in loader:
        dl0 = time.perf_counter()

        power = batch["power"].to(dev, non_blocking=False)
        layout = batch["layout"].to(dev, non_blocking=False)
        totalp = batch.get("total_power")
        totalp = totalp.to(dev, non_blocking=False) if totalp is not None else None

        if dev.type == "cuda":
            torch.cuda.synchronize()

        dl1 = time.perf_counter()
        t_dl_s += (dl1 - dl0)

        if dev.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(power, layout, totalp)
            end.record()
            torch.cuda.synchronize()
            t_fwd_s += start.elapsed_time(end) / 1e3
        else:
            f0 = time.perf_counter()
            _ = model(power, layout, totalp)
            t_fwd_s += time.perf_counter() - f0

        n_seen += int(power.shape[0])

    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_total_s = time.perf_counter() - t_wall0

    per_case_total_ms = t_total_s * 1e3 / max(n_seen, 1)
    per_case_dl_ms = t_dl_s * 1e3 / max(n_seen, 1)
    per_case_total_minus_dl_ms = (t_total_s - t_dl_s) * 1e3 / max(n_seen, 1)

    return {
        "ckpt": ckpt_path,
        "device": str(dev),
        "n_cases": int(n_seen),
        "eval_bs": int(eval_bs),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "dataset_build_s": float(ds_build_s),
        "total_wall_s": float(t_total_s),
        "forward_only_s": float(t_fwd_s),
        "dataloader_like_s": float(t_dl_s),
        "per_case_total_ms": float(per_case_total_ms),
        "per_case_forward_ms": float(t_fwd_s * 1e3 / max(n_seen, 1)),
        "per_case_dataloader_like_ms": float(per_case_dl_ms),
        "per_case_total_minus_dataloader_ms": float(per_case_total_minus_dl_ms),
        "total_minus_forward_s": float(t_total_s - t_fwd_s),
        "total_minus_forward_minus_dl_s": float(t_total_s - t_fwd_s - t_dl_s),
        "model_param_dtype": str(next(model.parameters()).dtype),
    }


def _replace_fp32_block(*, log_path: str, new_block: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write(new_block)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fp32_ckpt",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/checkpoints/fp32/fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_20260507_171818/hrnet_fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_ep0175_seed0_bs32_lr2e-04_base96_gw0p1_s4_b2_er2_tr0_va0.pth",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n_cases", type=int, default=1000)
    ap.add_argument(
        "--thermal_map_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument(
        "--out_time_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/logs/model_per_case_time.log",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    t = time_fp32_ckpt(
        args.fp32_ckpt,
        n_cases=args.n_cases,
        eval_bs=args.eval_bs,
        seed=args.seed,
        device=args.device,
        thermal_map_root=args.thermal_map_root,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    new_block = (
        "[hrnet_fp32_per_case_time]\n"
        f"date={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "env=conda:chipdiffusion\n"
        f"ckpt={t['ckpt']}\n"
        f"device={t['device']}\n"
        f"n_cases={t['n_cases']} eval_bs={t['eval_bs']} num_workers={t['num_workers']} pin_memory={t['pin_memory']}\n"
        f"model_param_dtype={t['model_param_dtype']}\n"
        f"dataset_build_s={t['dataset_build_s']:.6f}\n"
        f"total_wall_s={t['total_wall_s']:.6f}\n"
        f"forward_only_s={t['forward_only_s']:.6f}\n"
        f"dataloader_like_s={t['dataloader_like_s']:.6f}\n"
        f"per_case_total_ms={t['per_case_total_ms']:.6f}\n"
        f"per_case_forward_ms={t['per_case_forward_ms']:.6f}\n"
        f"per_case_dataloader_like_ms={t['per_case_dataloader_like_ms']:.6f}\n"
        f"per_case_total_minus_dataloader_ms={t['per_case_total_minus_dataloader_ms']:.6f}\n"
        f"total_minus_forward_s={t['total_minus_forward_s']:.6f}\n"
        f"total_minus_forward_minus_dl_s={t['total_minus_forward_minus_dl_s']:.6f}\n"
    )

    _replace_fp32_block(log_path=args.out_time_log, new_block=new_block)
    print(f"WROTE {args.out_time_log}")


if __name__ == "__main__":
    main()

'''
python /root/placement/flow_GCN/thermalmodel/fp32_case_time.py  --fp32_ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/fp32/fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_20260507_171818/hrnet_fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_ep0175_seed0_bs32_lr2e-04_base96_gw0p1_s4_b2_er2_tr0_va0.pth  --n_cases 1000  --eval_bs 32 --device cuda --out_time_log /root/placement/flow_GCN/thermalmodel/logs/model_per_case_time.log 

'''