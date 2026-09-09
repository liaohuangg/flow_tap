"""hrnet_hotspot_time.py

对比 GNN+HRNet 热模型的「推理速度」与 HotSpot 热仿真器 (真值生成器, 见
gen_dataset/gen_thermal_dataset.py) 的速度。

测量:
  1) 模型逐 case 前向时间 (GPU 核, 不含 DataLoader + H2D) 与端到端时间 (含 DataLoader+H2D)。
  2) HotSpot 仿真器逐 case 墙钟时间 (subprocess, 与模型相同的 test 划分前 N 个 case)。
  3) 加速比 = hotspot 逐 case 时间 / 模型逐 case 前向时间。

结果追加到日志 (默认 logs/time.log), 块格式可多次追加。

用法 (chipdiffusion env, cwd=thermalmodel):
  python hrnet_hotspot_time.py --n_cases 200 --hotspot_cases 20 --eval_bs 64

注意: 模型结构参数 (--base 96 等) 必须与训练时 auto_train.sh 一致。
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.data import DataLoader

import dataLoader as dl
import gnnhrnet as g

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HOTSPOT_BIN = os.path.join(_PROJECT_ROOT, "hotspot", "hotspot")


def _test_cases(seed: int):
    all_cases = dl.list_cases(os.path.join(g.DATA_ROOT, "power_map"))
    _tr, _val, test = dl.split_cases_by_i(all_cases, seed=seed)
    return test


# --------------------------------------------------------------------------- #
# 1) 模型推理计时
# --------------------------------------------------------------------------- #
@torch.no_grad()
def time_model(ckpt_path, *, n_cases, eval_bs, seed, device, num_workers, pin_memory,
               hidden, heads, num_layers, grid, base, stages, blocks_per_stage, expand_ratio):
    dev = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    ck = torch.load(ckpt_path, map_location="cpu")
    model = g.GNNHRNetModel(
        node_dim=8, hidden=hidden, heads=heads, num_layers=num_layers,
        grid=grid, base=base, stages=stages, blocks_per_stage=blocks_per_stage,
        expand_ratio=expand_ratio,
    ).to(dev)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    cases = _test_cases(seed)[:n_cases]
    ds = g.GNNThermalDataset(cases, grid=grid)
    loader = DataLoader(ds, batch_size=eval_bs, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory,
                        collate_fn=g.collate)

    if dev.type == "cuda":
        torch.cuda.synchronize()

    # warmup (触发 CUDA 内核编译 / cuDNN autotune, 不计时), 复用同一迭代器
    it = iter(loader)
    warmup_batches = 3
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            b = g.to_device(batch, dev)
            model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
    if dev.type == "cuda":
        torch.cuda.synchronize()

    t_wall0 = time.perf_counter()
    t_dl_s = 0.0
    t_fwd_s = 0.0
    n_seen = 0

    for batch in it:
        dl0 = time.perf_counter()
        b = g.to_device(batch, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dl1 = time.perf_counter()
        t_dl_s += (dl1 - dl0)

        if dev.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
            end.record()
            torch.cuda.synchronize()
            t_fwd_s += start.elapsed_time(end) / 1e3
        else:
            f0 = time.perf_counter()
            _ = model(b["x"], b["edge_index"], b["batch"], b["edge_attr"], b["field"])
            t_fwd_s += time.perf_counter() - f0

        n_seen += int(b["field"].size(0))

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t_total_s = time.perf_counter() - t_wall0

    return {
        "ckpt": ckpt_path,
        "device": str(dev),
        "dtype": str(next(model.parameters()).dtype),
        "n_cases": int(n_seen),
        "eval_bs": int(eval_bs),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "total_wall_s": float(t_total_s),
        "forward_only_s": float(t_fwd_s),
        "dataloader_h2d_s": float(t_dl_s),
        "per_case_forward_ms": float(t_fwd_s * 1e3 / max(n_seen, 1)),
        "per_case_total_ms": float(t_total_s * 1e3 / max(n_seen, 1)),
        "per_case_dataloader_h2d_ms": float(t_dl_s * 1e3 / max(n_seen, 1)),
        "cases_per_s": float(n_seen / max(t_total_s, 1e-9)),
    }


# --------------------------------------------------------------------------- #
# 2) HotSpot 仿真计时 (真值生成器)
# --------------------------------------------------------------------------- #
def _run_hotspot_timed(i, j, tmp_dir):
    cfg_dir = os.path.join(g.CFG_ROOT, f"system_{i}_config")
    config_file = os.path.join(cfg_dir, "new_hotspot.config")
    flp_file = os.path.join(cfg_dir, f"system_{i}L4_ChipLayer.flp")
    ptrace_file = os.path.join(cfg_dir, f"system_{i}_{j}.ptrace")
    layers_lcf = os.path.join(cfg_dir, f"system_{i}layers.lcf")
    steady_file = os.path.join(tmp_dir, f"system_{i}_{j}.steady")
    grid_steady_file = os.path.join(tmp_dir, f"system_{i}_{j}.grid.steady")

    cmd = [
        _HOTSPOT_BIN, "-c", config_file, "-f", flp_file, "-p", ptrace_file,
        "-steady_file", steady_file, "-grid_steady_file", grid_steady_file,
        "-model_type", "grid", "-detailed_3D", "on", "-grid_layer_file", layers_lcf,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(f"hotspot rc={proc.returncode} for case i={i} j={j}")
    return dt


def time_hotspot(cases, n_hotspot_cases):
    import statistics
    tmp = tempfile.mkdtemp(prefix="hrnet_hotspot_time_")
    try:
        times = []
        for (i, j) in cases[:n_hotspot_cases]:
            times.append(_run_hotspot_timed(int(i), int(j), tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ms = [t * 1e3 for t in times]
    return {
        "n_hotspot_cases": len(ms),
        "per_case_ms_mean": float(statistics.mean(ms)),
        "per_case_ms_median": float(statistics.median(ms)),
        "per_case_ms_min": float(min(ms)),
        "per_case_ms_max": float(max(ms)),
    }


# --------------------------------------------------------------------------- #
# 3) 写日志 + 汇总
# --------------------------------------------------------------------------- #
def _append_block(log_path, block):
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + block)


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/gnnhrnet_pwin/best.pth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_bs", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--n_cases", type=int, default=200, help="模型计时用 case 数")
    ap.add_argument("--hotspot_cases", type=int, default=20, help="HotSpot 计时用 case 数 (慢, 少取)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--out_time_log", type=str, default="logs/time.log")
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


def main():
    args = build_argparser().parse_args()

    m = time_model(
        args.ckpt, n_cases=args.n_cases, eval_bs=args.eval_bs, seed=args.seed,
        device=args.device, num_workers=args.num_workers, pin_memory=args.pin_memory,
        hidden=args.hidden, heads=args.heads, num_layers=args.num_layers,
        grid=args.grid, base=args.base, stages=args.stages,
        blocks_per_stage=args.blocks_per_stage, expand_ratio=args.expand_ratio,
    )

    test_cases = _test_cases(args.seed)
    h = time_hotspot(test_cases, args.hotspot_cases)

    speedup = h["per_case_ms_mean"] / max(m["per_case_forward_ms"], 1e-9)

    block = (
        "[gnnhrnet_vs_hotspot_time]\n"
        f"date={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "env=conda:chipdiffusion\n"
        f"ckpt={m['ckpt']}\n"
        f"device={m['device']}  dtype={m['dtype']}\n"
        f"model: n_cases={m['n_cases']} eval_bs={m['eval_bs']} "
        f"num_workers={m['num_workers']} pin_memory={m['pin_memory']}\n"
        f"model_per_case_forward_ms={m['per_case_forward_ms']:.4f}\n"
        f"model_per_case_total_ms={m['per_case_total_ms']:.4f}\n"
        f"model_per_case_dataloader_h2d_ms={m['per_case_dataloader_h2d_ms']:.4f}\n"
        f"model_cases_per_s={m['cases_per_s']:.2f}\n"
        f"hotspot: n_cases={h['n_hotspot_cases']}\n"
        f"hotspot_per_case_ms_mean={h['per_case_ms_mean']:.2f}\n"
        f"hotspot_per_case_ms_median={h['per_case_ms_median']:.2f}\n"
        f"hotspot_per_case_ms_min={h['per_case_ms_min']:.2f}\n"
        f"hotspot_per_case_ms_max={h['per_case_ms_max']:.2f}\n"
        f"speedup_model_vs_hotspot={speedup:.1f}x\n"
    )
    _append_block(args.out_time_log, block)

    print(f"[model ] {m['per_case_forward_ms']:.4f} ms/case (forward)  "
          f"{m['per_case_total_ms']:.4f} ms/case (end2end)  "
          f"{m['cases_per_s']:.1f} cases/s")
    print(f"[hotspot] {h['per_case_ms_mean']:.2f} ms/case (mean over {h['n_hotspot_cases']})")
    print(f"[speedup] model is {speedup:.0f}x faster than HotSpot")
    print(f"[log] wrote {args.out_time_log}")


if __name__ == "__main__":
    main()
