#!/usr/bin/env python3
"""测量采样时两个引导（热 / 线长）的**实际位移能力**，用于调参。

背景
----
`eval_thermal_guided.py` 里热引导和线长引导走的是两套完全不同的机制：

  线长: models.py::reverse_guidance_opt_force
        SGD(lr=grad_descent_rate) x grad_descent_steps 步, 无 clip
        step = grad_descent_rate * wirelength.guidance_weight * d(log pred_total)/dx
  热:   eval_thermal_guided.py::_thermal_guided_step
        SGD(lr=thermal.guidance_lr) x thermal.guidance_steps 步, grad_clip 裁剪
        step = thermal.guidance_lr * clip(w * dscore/dx, +-grad_clip)

两者的标称尺度差 1000 倍, 再叠加梯度本身的量级差, 实测下来热引导每步只能移动
~3e-8 (归一化坐标), 而数据集坐标网格是 1e-3 mm —— 也就是热引导实际上等于没开。

这个脚本把该测量固化下来, 在真正采样前几秒钟就能看出"这组参数能不能推动布局"。

用法
----
    python gen_dataset/probe_guidance.py --placement-dir <seed_N/placement> \
        --thermal-weight 2.0 --thermal-lr 0.1 --thermal-clip 0.02 \
        --thermal-start 0.5 --thermal-full 0.8

必须用 chipdiffusion 环境:
    /root/anaconda3/envs/chipdiffusion/bin/python
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DIFFUSION = PROJECT / "LayoutGenModel" / "diffusion"
for _p in (str(PROJECT), str(DIFFUSION), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# proxy_eval 会就地补 numpy<2 的别名 (wandb 0.13 需要), 必须在其它 import 之前。
import proxy_eval as PE  # noqa: E402

import torch  # noqa: E402

from train_graph_thermal import (  # noqa: E402
    _thermal_forward,
    _thermal_output_to_grid_and_avg,
)
from wirelength_surrogate import WirelengthSurrogate  # noqa: E402
from eval_thermal_guided import _thermal_cond  # noqa: E402

# 与 config_eval_fm.yaml / run_cases_hubump_5seeds_opt_wsl.sh 对齐的默认值
WL_LR_DEFAULT = 0.1  # model.grad_descent_rate
WL_WEIGHT_DEFAULT = 0.02  # wirelength.guidance_weight
WL_STEPS_DEFAULT = 8  # model.grad_descent_steps
WL_START_DEFAULT = 0.5
WL_FULL_DEFAULT = 0.8

TH_BETA_DEFAULT = 20.0  # thermal.smooth_max_beta
TH_MEAN_WEIGHT_DEFAULT = 0.05  # thermal.mean_weight (config 默认, 脚本未覆盖)


def _guided_step_count(num_steps: int, start: float) -> int:
    """progress = (num_steps - step + 1)/num_steps 从 1.0 降到 1/num_steps。

    schedule 在 progress > start 时施力, 所以施力步数 = (1 - start) * num_steps。
    """
    return max(0, int(round((1.0 - start) * num_steps)))


def _build_cond(record, device):
    """几何走 ProxyEvaluator._to_cond, 图 (edge_index/edge_weight) 从 connections 补。

    注意 edge_weight 必须和 edge_index 同 device —— 否则 _topology 里
    `cond.edge_weight[edge_ids]` 会报 "indices should be either on cpu or on
    the same device as the indexed tensor"。
    """
    x_hat, cond = PE.ProxyEvaluator._to_cond(record)

    name_to_idx = {c["name"]: i for i, c in enumerate(record["chiplets"])}
    edge_index, edge_weight = [], []
    for e in record.get("connections", []):
        a = name_to_idx.get(e["node1"])
        b = name_to_idx.get(e["node2"])
        if a is None or b is None or a == b:
            continue
        wc = float(e.get("wireCount", 0.0) or 0.0)
        edge_index += [[a, b], [b, a]]
        edge_weight += [wc, wc]

    return x_hat, cond, edge_index, edge_weight


def _grad_p90(fn, x_hat, device):
    """对 x_hat 求标量目标 fn(x) 的梯度, 返回 |grad| 的 p90。"""
    x = x_hat.unsqueeze(0).to(device).clone().requires_grad_(True)
    fn(x).sum().backward()
    return x.grad.detach().abs().flatten().quantile(0.9).item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--placement-dir", required=True,
                    help="含 *_placement.json 的目录 (如 seed_5/placement)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-diffusion-steps", type=int, default=100,
                    help="与 model.max_diffusion_steps 一致")

    ap.add_argument("--thermal-weight", type=float, default=0.002)
    ap.add_argument("--thermal-lr", type=float, default=0.001)
    ap.add_argument("--thermal-steps", type=int, default=1)
    ap.add_argument("--thermal-clip", type=float, default=0.02)
    ap.add_argument("--thermal-start", type=float, default=0.75)
    ap.add_argument("--thermal-full", type=float, default=0.90)
    ap.add_argument("--thermal-mean-weight", type=float, default=TH_MEAN_WEIGHT_DEFAULT)
    ap.add_argument("--thermal-smooth-max-beta", type=float, default=TH_BETA_DEFAULT)

    ap.add_argument("--wl-weight", type=float, default=WL_WEIGHT_DEFAULT)
    ap.add_argument("--wl-lr", type=float, default=WL_LR_DEFAULT)
    ap.add_argument("--wl-steps", type=int, default=WL_STEPS_DEFAULT)
    ap.add_argument("--wl-start", type=float, default=WL_START_DEFAULT)

    ap.add_argument("--target-mm-per-run", type=float, default=3.0,
                    help="反推表用的目标总位移 (mm), 默认 3mm")
    args = ap.parse_args()

    pdir = Path(args.placement_dir)
    files = sorted(pdir.glob("*_placement.json"))
    if not files:
        raise SystemExit(f"没有找到 *_placement.json: {pdir}")

    dev = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                       else "cpu")
    ev = PE.ProxyEvaluator(device=args.device)
    wl_surrogate = WirelengthSurrogate(
        dict(enabled=True,
             ckpt=str(PROJECT / "wirelengthmodel/checkpoint/best_wlmodel_total_60k.pt"),
             normalizer=str(PROJECT / "wirelengthmodel/checkpoint/normalizer_compat_69_76.json"),
             objective="log_total"),
        dev,
    )

    th_steps_n = _guided_step_count(args.num_diffusion_steps, args.thermal_start)
    wl_steps_n = _guided_step_count(args.num_diffusion_steps, args.wl_start)

    print(f"布局目录 : {pdir}  ({len(files)} cases)")
    print(f"热引导   : w={args.thermal_weight}  lr={args.thermal_lr}  "
          f"steps/guided={args.thermal_steps}  clip={args.thermal_clip}  "
          f"window>{args.thermal_start} -> {th_steps_n} 个施力步")
    print(f"线长引导 : w={args.wl_weight}  lr={args.wl_lr}  "
          f"steps/guided={args.wl_steps}  window>{args.wl_start} -> {wl_steps_n} 个施力步")
    print()

    hdr = (f"{'case':24s} {'n':>3s} {'tmax_C':>8s} {'side_mm':>8s} | "
           f"{'g_wl p90':>9s} {'d_wl/step':>10s} {'wl_total':>9s} | "
           f"{'g_th p90':>9s} {'d_th/step':>10s} {'th_total':>9s} | {'clip':>5s} {'wl/th':>9s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        n = len(rec["chiplets"])
        x_hat, cond, ei, ew = _build_cond(rec, dev)
        if not ei:
            print(f"{f.stem:24s} {n:3d}  (无 connections, 跳过)")
            continue

        cond.edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous().to(dev)
        cond.edge_weight = torch.tensor(ew, dtype=torch.float32).to(dev)
        cond.tap_hubump = cond.tap_hubump.to(dev)

        side_mm = float(cond.chip_size[2])

        def _thermal_score(x, cond=cond):
            out = _thermal_forward(ev.th_model, x, cond, grid_size=64,
                                   rect_sharpness=80.0, stats=ev.th_stats,
                                   differentiable=True)
            temp, avg = _thermal_output_to_grid_and_avg(out)
            flat = temp.flatten(1)
            score = torch.logsumexp(flat * args.thermal_smooth_max_beta, dim=1) \
                / args.thermal_smooth_max_beta
            mean_t = avg.view(-1) if avg is not None else flat.mean(dim=1)
            return score + args.thermal_mean_weight * mean_t

        g_th = _grad_p90(_thermal_score, x_hat, dev)

        with torch.no_grad():
            out = _thermal_forward(ev.th_model, x_hat.unsqueeze(0).to(dev), cond,
                                   grid_size=64, rect_sharpness=80.0,
                                   stats=ev.th_stats, differentiable=True)
            grid, _avg = _thermal_output_to_grid_and_avg(out)
            tmax_c = float(ev._th["denorm"](grid, ev.th_stats).max().item() - 273.15)

        g_wl = _grad_p90(lambda x, c=cond: wl_surrogate.potential(x, _thermal_cond(c)),
                         x_hat, dev)

        # 每步位移 (归一化坐标)
        d_wl_step = args.wl_lr * args.wl_weight * g_wl
        d_th_step = args.thermal_lr * min(args.thermal_weight * g_th, args.thermal_clip)

        wl_total_norm = d_wl_step * args.wl_steps * wl_steps_n
        th_total_norm = d_th_step * args.thermal_steps * th_steps_n

        rows.append(dict(stem=f.stem, n=n, side_mm=side_mm, g_wl=g_wl, g_th=g_th,
                         d_wl_step=d_wl_step, d_th_step=d_th_step,
                         wl_total_norm=wl_total_norm, th_total_norm=th_total_norm,
                         clip_binds=args.thermal_weight * g_th > args.thermal_clip))

        print(f"{f.stem:24s} {n:3d} {tmax_c:8.1f} {side_mm:8.1f} | "
              f"{g_wl:9.4f} {d_wl_step:10.3e} {wl_total_norm * side_mm / 2:8.2f}mm | "
              f"{g_th:9.5f} {d_th_step:10.3e} {th_total_norm * side_mm / 2:8.2f}mm | "
              f"{'YES' if rows[-1]['clip_binds'] else 'no':>5s} "
              f"{wl_total_norm / max(th_total_norm, 1e-30):9.1f}")

    if not rows:
        raise SystemExit("没有可评估的 case")

    import statistics as st

    for label, key in (("线长", "wl_total_norm"), ("热", "th_total_norm")):
        vals = [r[key] * r["side_mm"] / 2 for r in rows]
        print(f"\n{label}引导总位移 (mm): 中位 {st.median(vals):8.3f}  "
              f"最小 {min(vals):8.3f}  最大 {max(vals):8.3f}")

    ratio = st.median(r["wl_total_norm"] * r["side_mm"] / 2 for r in rows) / \
        max(st.median(r["th_total_norm"] * r["side_mm"] / 2 for r in rows), 1e-30)
    print(f"线长 / 热 位移能力比 = {ratio:.1f}x")

    binds = sum(r["clip_binds"] for r in rows)
    print(f"热 grad_clip 触发: {binds}/{len(rows)} cases "
          f"({'clip 生效, weight 非线性' if binds else 'clip 未触发, weight 近似线性'})")

    # ---- 反推: 要让热引导达到目标位移需要什么参数 ----
    print(f"\n要达到热引导总位移 {args.target_mm_per_run} mm 需要 (中位 case):")
    med = sorted(rows, key=lambda r: r["n"])[len(rows) // 2]
    g_th, side = med["g_th"], med["side_mm"]
    need_total = args.target_mm_per_run * 2.0 / side  # 归一化
    need_step = need_total / max(args.thermal_steps * th_steps_n, 1)
    print(f"  中位 case = {med['stem']} (n={med['n']}, side={side:.1f}mm, g_th p90={g_th:.5f})")
    print(f"  需要每步位移 {need_step:.3e} (归一化), 共 {args.thermal_steps * th_steps_n} 步")
    print(f"  {'lr':>8s} {'weight':>9s} {'normalize':>10s} {'d/step':>11s} {'总位移':>9s}")
    for lr, w, nz in ((0.1, 2.0, "off"), (0.1, 10.0, "off"), (0.2, 5.0, "off"),
                      (0.02, 1.0, "on"), (0.1, 1.0, "on")):
        if nz == "on":
            d = lr * w  # gradient_normalize 把梯度归一为 RMS=1, 再乘回 weight
        else:
            d = lr * min(w * g_th, args.thermal_clip)
        print(f"  {lr:8.3f} {w:9.2f} {nz:>10s} {d:11.3e} "
              f"{d * args.thermal_steps * th_steps_n * side / 2:8.2f}mm")
    print("\n注: normalize=on 需同时设 thermal.gradient_normalize=true, "
          "此时 grad_clip 不再起作用。")


if __name__ == "__main__":
    main()
