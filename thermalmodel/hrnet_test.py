"""fp32_test.py

FP32 workflow entrypoint (HRNet version):
  1) Select checkpoint by VAL mean_rmse from an existing eval log (--eval_log)
     or by enumerating checkpoints in --fp32_ckpt_dir and evaluating on VAL.
  2) Evaluate the selected checkpoint on TEST split and write metrics log.
  3) Optionally export top-k best and top-k worst test cases to --out_fig_dir.

Run under the chipdiffusion conda env (torch required).
"""

import argparse
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# Allow running `python fp32_test.py ...` from within thermalmodel/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.eval_hrnet_ckpt import eval_hrnet_ckpt


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fp32_ckpt",
        type=str,
        default="",
        help="If non-empty, use this checkpoint directly; otherwise auto-select from --fp32_ckpt_dir by val.",
    )
    ap.add_argument(
        "--fp32_ckpt_dir",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/checkpoints/fp32_b96_lr2e-4_gw0p1_s0_ep250",
        help="Directory containing FP32 checkpoints saved every 5 epochs.",
    )
    ap.add_argument(
        "--eval_log",
        type=str,
        default="",
        help="If set, parse this eval log and select the best VAL checkpoint from it (preferred).",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument(
        "--thermal_map_root",
        type=str,
        default="/root/placement/flow_GCN/Dataset/dataset/output/thermal/thermal_map",
    )
    ap.add_argument(
        "--hotspot_cfg_rel",
        type=str,
        default="Dataset/dataset/output/thermal/hotspot_config",
    )

    ap.add_argument(
        "--out_metrics_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/fp32_test.log",
    )
    ap.add_argument(
        "--out_fig_dir",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/fig",
    )

    ap.add_argument(
        "--time_n_cases",
        type=int,
        default=0,
        help="If >0, run timing over the first N cases (sorted by i,j) and update --out_time_log.",
    )
    ap.add_argument(
        "--out_time_log",
        type=str,
        default="/root/placement/flow_GCN/thermalmodel/test_result/fp32_per_case_time.log",
    )
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument(
        "--select_metric",
        type=str,
        default="mean_rmse",
        choices=["mean_rmse"],
    )
    ap.add_argument("--k_fig", type=int, default=50, help="Export top-k best and bottom-k worst on test split")
    ap.add_argument("--append", action="store_true", help="append metrics log instead of overwrite")
    return ap


def _list_checkpoints(ckpt_dir: str) -> List[str]:
    if not os.path.isdir(ckpt_dir):
        return []
    # HRNet checkpoints look like: hrnet_..._ep0175_... .pth
    pat = re.compile(r".*_ep(\d{4})_.*\.pth$")
    ckpts: List[str] = []
    for fn in os.listdir(ckpt_dir):
        if pat.match(fn):
            ckpts.append(os.path.join(ckpt_dir, fn))
    ckpts.sort()
    return ckpts


def _parse_eval_log_best_ckpt(eval_log_path: str) -> Tuple[float, str]:
    """Parse an eval log and return (best_val_mean_rmse, ckpt_path)."""
    s = open(eval_log_path, "r", encoding="utf-8", errors="replace").read()

    # Preferred explicit selection line
    m = re.search(r"selected best by rmse=([0-9.]+):\s*(\S+\.pth)", s)
    if m:
        return float(m.group(1)), m.group(2)

    # Fallback: scan all val mean_rmse lines
    best_rmse = float("inf")
    best_ckpt = ""
    for mm in re.finditer(r"val mean_rmse=([0-9.]+)\s+ckpt=(\S+\.pth)", s):
        r = float(mm.group(1))
        p = mm.group(2)
        if r < best_rmse:
            best_rmse = r
            best_ckpt = p

    if not best_ckpt:
        raise RuntimeError(f"Could not find best checkpoint in eval log: {eval_log_path}")

    return float(best_rmse), best_ckpt


def _metrics_to_dict(*, metrics, meta: Dict[str, Any]) -> Dict[str, Any]:
    best0 = meta.get("best")[0] if isinstance(meta.get("best"), list) and meta.get("best") else {}
    worst0 = meta.get("worst")[0] if isinstance(meta.get("worst"), list) and meta.get("worst") else {}

    return {
        "units": metrics.units,
        "n_cases": int(metrics.n_cases),
        "mean_rmse": float(metrics.mean_rmse),
        "min_rmse": float(metrics.min_rmse),
        "max_rmse": float(metrics.max_rmse),
        "mean_mae": float(metrics.mean_mae),
        "max_mae": float(metrics.max_mae),
        "mean_mse": float(metrics.mean_mse),
        "mean_mape_pct": float(metrics.mean_mape_pct),
        "mean_abs_rel": float(getattr(metrics, "mean_abs_rel", 0.0)),
        "mean_abs_rel_pct": float(getattr(metrics, "mean_abs_rel_pct", 0.0)),
        "mean_rel": float(getattr(metrics, "mean_rel", 0.0)),
        "mean_rel_pct": float(getattr(metrics, "mean_rel_pct", 0.0)),
        "mean_peak_ae": float(getattr(metrics, "mean_peak_ae", 0.0)),
        "mean_peak_abs_rel": float(getattr(metrics, "mean_peak_abs_rel", 0.0)),
        "mean_peak_abs_rel_pct": float(getattr(metrics, "mean_peak_abs_rel_pct", 0.0)),
        "mean_peak_rel": float(getattr(metrics, "mean_peak_rel", 0.0)),
        "mean_peak_rel_pct": float(getattr(metrics, "mean_peak_rel_pct", 0.0)),
        "mean_grad": float(metrics.mean_grad),
        "max_ae": float(metrics.max_ae),
        "best_i": best0.get("i", ""),
        "best_j": best0.get("j", ""),
        "best_rmse": best0.get("rmse", ""),
        "worst_i": worst0.get("i", ""),
        "worst_j": worst0.get("j", ""),
        "worst_rmse": worst0.get("rmse", ""),
    }


@torch.no_grad()
def _eval_hrnet_ckpt_on_split(
    ckpt_path: str,
    *,
    split: str,
    eval_bs: int,
    seed: int,
    device: str,
    out_fig_dir: str,
    topk: int,
    limit_cases: int = 0,
) -> Dict[str, Any]:
    metrics, meta = eval_hrnet_ckpt(
        ckpt_path,
        split=split,
        batch_size=eval_bs,
        seed=seed,
        prefer_device=device,
        out_fig_dir=out_fig_dir,
        topk=topk,
        limit_cases=int(limit_cases),
    )
    return _metrics_to_dict(metrics=metrics, meta=meta)


@torch.no_grad()
def _time_fp32_ckpt(
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
    # Keep the old signature to avoid breaking downstream scripts.
    # For HRNet, timing is not implemented in this entrypoint.
    raise NotImplementedError("Timing is not implemented for HRNet in fp32_test.py")


def _replace_fp32_block(*, log_path: str, new_block: str) -> None:
    with open(log_path, "r", encoding="utf-8") as f:
        s = f.read()

    start_tok = "[fp32]"
    i0 = s.find(start_tok)
    if i0 < 0:
        raise RuntimeError(f"Did not find {start_tok} in {log_path}")

    next_tok = "\n[hotspot_timing]"
    i1 = s.find(next_tok, i0)
    if i1 < 0:
        raise RuntimeError(f"Did not find [hotspot_timing] after {start_tok} in {log_path}")

    out = s[:i0] + new_block + s[i1:]

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)


def _write_metrics_log(
    *,
    out_path: str,
    mode: str,
    args: argparse.Namespace,
    selected_ckpt: str,
    val_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    lines: List[str] = []
    lines.append("# hrnet fp32_test")
    lines.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("env=conda:chipdiffusion")
    lines.append(f"project_root={_PROJECT_ROOT}")
    lines.append("")

    lines.append("[args]")
    lines.append(f"fp32_ckpt={args.fp32_ckpt}")
    lines.append(f"fp32_ckpt_dir={args.fp32_ckpt_dir}")
    lines.append(f"eval_log={args.eval_log}")
    lines.append(f"seed={args.seed}")
    lines.append(f"eval_bs={args.eval_bs}")
    lines.append(f"device={args.device}")
    lines.append(f"k_fig={args.k_fig}")
    lines.append(f"out_fig_dir={args.out_fig_dir}")
    lines.append("")

    lines.append("[selected_ckpt]")
    lines.append(selected_ckpt)
    lines.append("")

    def _emit_metrics(tag: str, m: Dict[str, Any]) -> None:
        lines.append(f"[{tag}]")
        # stable order
        keys = [
            "units",
            "n_cases",
            "mean_rmse",
            "min_rmse",
            "max_rmse",
            "mean_mae",
            "max_mae",
            "mean_mse",
            "mean_mape_pct",
            "mean_abs_rel",
            "mean_abs_rel_pct",
            "mean_rel",
            "mean_rel_pct",
            "mean_peak_ae",
            "mean_peak_abs_rel",
            "mean_peak_abs_rel_pct",
            "mean_peak_rel",
            "mean_peak_rel_pct",
            "mean_grad",
            "max_ae",
            "best_i",
            "best_j",
            "best_rmse",
            "worst_i",
            "worst_j",
            "worst_rmse",
        ]
        for k in keys:
            if k in m:
                lines.append(f"{k}={m.get(k)}")
        lines.append("")

    _emit_metrics("val_metrics", val_metrics)
    _emit_metrics("test_metrics", test_metrics)

    with open(out_path, mode, encoding="utf-8") as f:
        if mode == "a":
            f.write("\n")
        f.write("\n".join(lines))


def main() -> None:
    args = build_argparser().parse_args()

    # 1) pick checkpoint
    if args.fp32_ckpt:
        selected = args.fp32_ckpt
    elif args.eval_log:
        _best_val, selected = _parse_eval_log_best_ckpt(args.eval_log)
    else:
        ckpts = _list_checkpoints(args.fp32_ckpt_dir)
        if not ckpts:
            raise RuntimeError(f"No checkpoints found in {args.fp32_ckpt_dir}")

        # NOTE: enumerating + val-eval can be very slow; prefer --eval_log
        best_path = ""
        best_val = None
        for p in ckpts:
            m = _eval_hrnet_ckpt_on_split(
                p,
                split="val",
                eval_bs=args.eval_bs,
                seed=args.seed,
                device=args.device,
                out_fig_dir="",
                topk=0,
            )
            score = float(m.get(args.select_metric, 1e30))
            if best_val is None or score < best_val:
                best_val = score
                best_path = p

        selected = best_path

    # 2) val/test metrics
    val_metrics = _eval_hrnet_ckpt_on_split(
        selected,
        split="val",
        eval_bs=args.eval_bs,
        seed=args.seed,
        device=args.device,
        out_fig_dir="",
        topk=0,
    )

    test_metrics = _eval_hrnet_ckpt_on_split(
        selected,
        split="test",
        eval_bs=args.eval_bs,
        seed=args.seed,
        device=args.device,
        out_fig_dir=(args.out_fig_dir if int(args.k_fig) > 0 else ""),
        topk=int(args.k_fig),
    )

    mode = "a" if args.append else "w"
    _write_metrics_log(
        out_path=args.out_metrics_log,
        mode=mode,
        args=args,
        selected_ckpt=selected,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )
    print(f"WROTE {args.out_metrics_log}")

    # 3) timing (not supported)
    if int(args.time_n_cases) > 0:
        raise NotImplementedError("--time_n_cases is not supported for HRNet in fp32_test.py")


if __name__ == "__main__":
    main()

#python fp32_test.py --fp32_ckpt /root/placement/flow_GCN/thermalmodel/checkpoints/fp32/fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_20260507_171818/hrnet_fp32_hrnet_b96_lr2e-4_s4_bps2_er2_gw0.1_aw0.1_mcw0.1_topkw0.0_topkk0_peakw0.0_ep200_seed0_tr0_va0_ep0175_seed0_bs32_lr2e-04_base96_gw0p1_s4_b2_er2_tr0_va0.pth --device cuda --eval_bs 32 --k_fig 50 --out_metrics_log /root/placement/flow_GCN/thermalmodel/logs/test_eval_best_from_val.log --out_fig_dir /root/placement/flow_GCN/thermalmodel/test_result/fig