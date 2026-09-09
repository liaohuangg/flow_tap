"""eval_hrnet_ckpt.py — GNN+HRNet checkpoint 评估 (完整指标 + 最坏/最好图 + 日志)

针对当前 GNN+HRNet 模型 (gnnhrnet.py) 重写。功能:
  1. 加载 checkpoint, 在 val / test 划分上逐 case 推理 (温度反归一化到 °C)。
  2. 计算完整指标: 整场 RMSE/MAE/MSE、相对误差、峰值误差、热点 RMSE、空间梯度差。
  3. 按逐 case RMSE 排序, 取最坏 topk + 最好 topk, 用 draw_thermal_fig 绘制 pred/gt。
  4. 结果写入 --out_log (默认 logs/test.log)。

每个指标的含义见 README.md「八、评估指标说明」。

用法:
  python eval_hrnet_ckpt.py \
      --ckpt checkpoints/gnnhrnet_pwin/best.pth \
      --split test \
      --out_log logs/test.log \
      --out_fig_dir figs/test \
      --topk 20

注意: 模型结构参数 (--base 96 等) 必须与训练时 auto_train.sh 一致。
"""
import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import dataLoader as dl
import gnnhrnet as g
from draw_thermal_fig import plot_thermal_grid_overlay

EPS = 1e-6


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--out_log", type=str, default="logs/test.log")
    ap.add_argument("--out_fig_dir", type=str, default="figs/test")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_cases", type=int, default=0,
                    help=">0 时只评估前 N 个 case (调试用)")
    ap.add_argument("--hotspot_thr", type=float, default=0.05)
    # 模型结构参数 (与 auto_train.sh 一致)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--base", type=int, default=96)
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--blocks_per_stage", type=int, default=2)
    ap.add_argument("--expand_ratio", type=int, default=2)
    return ap


def _per_case_metrics(pred_c, gt_c, pred_max_c, gt_peak_c):
    """在单个 batch 上计算逐 case 指标 (输入均为 °C, [B,1,H,W] / [B,1])。

    返回: dict of 1-D ndarray (长度 B)。
    """
    diff = pred_c - gt_c
    sq = diff ** 2
    adiff = np.abs(diff)
    rmse = np.sqrt(sq.mean(axis=(1, 2, 3)))
    mae = adiff.mean(axis=(1, 2, 3))
    mse = sq.mean(axis=(1, 2, 3))
    max_ae = adiff.max(axis=(1, 2, 3))
    mape_pct = (adiff / (np.abs(gt_c) + EPS)).mean(axis=(1, 2, 3)) * 100.0
    gt_mean = gt_c.mean(axis=(1, 2, 3))
    abs_rel = mae / (gt_mean + EPS)
    rel = diff.mean(axis=(1, 2, 3)) / (gt_mean + EPS)

    pred_max = pred_max_c[:, 0]
    gt_peak = gt_peak_c[:, 0]
    peak_ae = np.abs(pred_max - gt_peak)
    peak_bias = pred_max - gt_peak
    peak_abs_rel = peak_ae / (gt_peak + EPS)

    return {
        "rmse": rmse, "mae": mae, "mse": mse, "max_ae": max_ae,
        "mape_pct": mape_pct, "abs_rel": abs_rel, "rel": rel,
        "pred_max": pred_max, "gt_peak": gt_peak,
        "peak_ae": peak_ae, "peak_bias": peak_bias, "peak_abs_rel": peak_abs_rel,
    }


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    temp_span = g.TEMP_MAX - g.TEMP_MIN

    # --- 数据划分 (与训练一致: 按布局 i 8:1:1, seed=0) ---
    all_cases = dl.list_cases(os.path.join(g.DATA_ROOT, "power_map"))
    _tr, val_cases, test_cases = dl.split_cases_by_i(all_cases, seed=args.seed)
    cases = test_cases if args.split == "test" else val_cases
    if args.limit_cases and args.limit_cases > 0:
        cases = cases[: args.limit_cases]

    # --- 模型 ---
    model = g.GNNHRNetModel(
        node_dim=8, hidden=args.hidden, heads=args.heads, num_layers=args.num_layers,
        grid=args.grid, base=args.base, stages=args.stages,
        blocks_per_stage=args.blocks_per_stage, expand_ratio=args.expand_ratio,
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    ds = g.GNNThermalDataset(cases, grid=args.grid)
    loader = DataLoader(ds, batch_size=args.eval_bs, shuffle=False,
                        num_workers=args.num_workers, collate_fn=g.collate)

    # --- 逐 case 推理 ---
    hm_sq, hmax_ae, hmax_se, n_pix, n_case = 0.0, 0.0, 0.0, 0, 0
    hot_sq, hot_n = 0.0, 0
    grad_sum = 0.0
    per_case = []  # 每个 case 一个 dict (含指标 + pred/gt 数组)

    with torch.no_grad():
        for batch in loader:
            b = g.to_device(batch, device)
            pred01 = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
            diff01 = pred01 - b["temp"]

            # 整场聚合 (与 gnnhrnet.evaluate 一致, 用 pred 自身最大值)
            hm_sq += float((diff01 ** 2).sum())
            n_pix += b["temp"].numel()
            pred_max = pred01.amax(dim=(2, 3))
            hmax_ae += float((pred_max - b["peak"]).abs().sum())
            hmax_se += float((pred_max - b["peak"]).sum())
            n_case += b["peak"].numel()

            # 热点区域 RMSE (功率密度 > 阈值)
            if args.hotspot_thr > 0.0:
                mask = (b["field"][:, 0:1] > args.hotspot_thr).float()
                hot_sq += float((diff01 ** 2 * mask).sum())
                hot_n += int(mask.sum())

            # 空间梯度差 (归一化空间, 最后乘 temp_span 得 °C 量纲)
            grad_sum += float(g._spatial_gradient_loss(pred01, b["temp"])) * pred01.size(0)

            # 反归一化到 °C
            pred_c = (pred01.cpu() * temp_span + g.TEMP_MIN).numpy()  # [B,1,64,64]
            gt_c = (b["temp"].cpu() * temp_span + g.TEMP_MIN).numpy()
            pred_max_c = (pred_max.cpu() * temp_span + g.TEMP_MIN).numpy()  # [B,1]
            gt_peak_c = (b["peak"].cpu() * temp_span + g.TEMP_MIN).numpy()  # [B,1]
            i_arr = b["i"].cpu().numpy()
            j_arr = b["j"].cpu().numpy()

            m = _per_case_metrics(pred_c, gt_c, pred_max_c, gt_peak_c)
            B = pred_c.shape[0]
            for k in range(B):
                per_case.append({
                    "i": int(i_arr[k]), "j": int(j_arr[k]),
                    "rmse": float(m["rmse"][k]), "mae": float(m["mae"][k]),
                    "mse": float(m["mse"][k]), "max_ae": float(m["max_ae"][k]),
                    "mape_pct": float(m["mape_pct"][k]), "abs_rel": float(m["abs_rel"][k]),
                    "rel": float(m["rel"][k]), "pred_max": float(m["pred_max"][k]),
                    "gt_peak": float(m["gt_peak"][k]), "peak_ae": float(m["peak_ae"][k]),
                    "peak_bias": float(m["peak_bias"][k]), "peak_abs_rel": float(m["peak_abs_rel"][k]),
                    "pred": pred_c[k, 0], "gt": gt_c[k, 0],
                })

    # --- 聚合指标 ---
    n = max(len(per_case), 1)
    agg = {
        "hm_rmse": (hm_sq / max(n_pix, 1)) ** 0.5 * temp_span,
        "peak_mae": hmax_ae / max(n_case, 1) * temp_span,
        "peak_bias": hmax_se / max(n_case, 1) * temp_span,
        "hotspot_rmse": (hot_sq / max(hot_n, 1)) ** 0.5 * temp_span if args.hotspot_thr > 0.0 else 0.0,
        "mean_grad": grad_sum / n_case * temp_span,
        "mean_rmse": float(np.mean([d["rmse"] for d in per_case])),
        "min_rmse": float(np.min([d["rmse"] for d in per_case])),
        "max_rmse": float(np.max([d["rmse"] for d in per_case])),
        "mean_mae": float(np.mean([d["mae"] for d in per_case])),
        "max_mae": float(np.max([d["mae"] for d in per_case])),
        "mean_mse": float(np.mean([d["mse"] for d in per_case])),
        "max_ae": float(np.max([d["max_ae"] for d in per_case])),
        "mean_mape_pct": float(np.mean([d["mape_pct"] for d in per_case])),
        "mean_abs_rel": float(np.mean([d["abs_rel"] for d in per_case])),
        "mean_rel": float(np.mean([d["rel"] for d in per_case])),
        "mean_peak_ae": float(np.mean([d["peak_ae"] for d in per_case])),
        "mean_peak_abs_rel": float(np.mean([d["peak_abs_rel"] for d in per_case])),
    }

    # --- 排序取最坏/最好 ---
    per_case.sort(key=lambda d: d["rmse"])
    best = per_case[: args.topk]
    worst = list(reversed(per_case[-args.topk:]))

    # --- 写日志 ---
    os.makedirs(os.path.dirname(args.out_log) or ".", exist_ok=True)
    L = []
    L.append("# GNN+HRNet per-case eval")
    L.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"ckpt={args.ckpt}")
    L.append(f"ckpt_epoch={ck.get('epoch')}  best_rmse={ck.get('best_rmse')}")
    L.append(f"split={args.split}  n_cases={len(per_case)}  topk={args.topk}")
    L.append("")
    L.append("[aggregate]  (含义见 README「八、评估指标说明」)")
    L.append(f"hm_rmse={agg['hm_rmse']:.4f}C  (整场池化 RMSE, 与训练日志 val_hm_rmse 同口径)")
    L.append(f"mean_rmse={agg['mean_rmse']:.4f}C  min_rmse={agg['min_rmse']:.4f}C  max_rmse={agg['max_rmse']:.4f}C  (逐 case RMSE 统计)")
    L.append(f"mean_mae={agg['mean_mae']:.4f}C  max_mae={agg['max_mae']:.4f}C  mean_mse={agg['mean_mse']:.4f}C^2")
    L.append(f"max_ae={agg['max_ae']:.4f}C  (全像素最大单点绝对误差)")
    L.append(f"mean_mape_pct={agg['mean_mape_pct']:.4f}%  mean_abs_rel={agg['mean_abs_rel']:.6f}  mean_rel={agg['mean_rel']:+.6f}")
    L.append(f"mean_peak_ae={agg['mean_peak_ae']:.4f}C  peak_bias={agg['peak_bias']:+.4f}C  mean_peak_abs_rel={agg['mean_peak_abs_rel']:.6f}")
    L.append(f"hotspot_rmse={agg['hotspot_rmse']:.4f}C  mean_grad={agg['mean_grad']:.4f}C/grid")
    L.append("")

    def _fmt(d):
        return (f"i={d['i']} j={d['j']} rmse={d['rmse']:.4f}C mae={d['mae']:.4f}C "
                f"mape={d['mape_pct']:.3f}% peak_ae={d['peak_ae']:.3f}C "
                f"(pred_max={d['pred_max']:.2f}C gt_peak={d['gt_peak']:.2f}C)")

    L.append(f"[worst_{len(worst)}_cases_by_rmse]")
    for rank, d in enumerate(worst, 1):
        L.append(f"#{rank:02d}  " + _fmt(d))
    L.append("")
    L.append(f"[best_{len(best)}_cases_by_rmse]")
    for rank, d in enumerate(best, 1):
        L.append(f"#{rank:02d}  " + _fmt(d))
    L.append("")

    with open(args.out_log, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # --- 绘图 (最坏/最好 topk) ---
    os.makedirs(args.out_fig_dir, exist_ok=True)
    cfg = g.CFG_ROOT
    side_cache = {}
    n_figs = 0
    for tag, entries in (("worst", worst), ("best", best)):
        for rank, d in enumerate(entries, 1):
            i, j = d["i"], d["j"]
            flp = os.path.join(cfg, f"system_{i}_config", "system.flp")
            if i not in side_cache:
                l4 = os.path.join(cfg, f"system_{i}_config", f"system_{i}L4_ChipLayer.flp")
                side_cache[i] = dl.interposer_side_m(l4) * 1000.0
            side_mm = side_cache[i]
            vmin = float(min(d["pred"].min(), d["gt"].min()))
            vmax = float(max(d["pred"].max(), d["gt"].max()))
            stem = f"{tag}_r{rank:02d}_i{i}_j{j}"
            plot_thermal_grid_overlay(
                flp, d["pred"],
                os.path.join(args.out_fig_dir, f"{stem}_pred_rmse{d['rmse']:.4f}.png"),
                title=f"Pred {tag} r={rank:02d} i={i} j={j} RMSE={d['rmse']:.4f}C",
                vmin=vmin, vmax=vmax, side_mm=side_mm,
            )
            plot_thermal_grid_overlay(
                flp, d["gt"],
                os.path.join(args.out_fig_dir, f"{stem}_gt.png"),
                title=f"GT {tag} r={rank:02d} i={i} j={j} peak={d['gt_peak']:.2f}C",
                vmin=vmin, vmax=vmax, side_mm=side_mm,
            )
            n_figs += 2

    # 一行汇总 (脚本用)
    print(
        f"[eval {args.split}] n={len(per_case)} hm_rmse={agg['hm_rmse']:.4f}C "
        f"mean_rmse={agg['mean_rmse']:.4f}C min_rmse={agg['min_rmse']:.4f}C max_rmse={agg['max_rmse']:.4f}C "
        f"mean_mae={agg['mean_mae']:.4f}C max_ae={agg['max_ae']:.4f}C "
        f"mean_mape={agg['mean_mape_pct']:.3f}% peak_ae={agg['mean_peak_ae']:.4f}C "
        f"peak_bias={agg['peak_bias']:+.4f}C hotspot_rmse={agg['hotspot_rmse']:.4f}C"
    )
    print(f"[log] wrote {args.out_log}")
    print(f"[fig] wrote {n_figs} figures -> {args.out_fig_dir}")


if __name__ == "__main__":
    main()
