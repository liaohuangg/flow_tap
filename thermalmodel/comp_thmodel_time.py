#!/usr/bin/env python3
"""comp_thmodel_time.py — 对比 HotSpot 热仿真 vs GNN+HRNet 热模型 的逐 case 时间。

从 placement_dataset_tw 读取 system_{start} 往后数 n_cases 个布局 (默认 30w 往后 100 个,
即 system_300001..300100), 对每个 case 做两件事并计时:

  1) HotSpot (真值生成器): 复刻 gen_dataset/gen_thermal_dataset.py 的 6 层 TAP-2.5D 热模型
     (grid=64, -detailed_3D on), 单线程 subprocess 运行, 计算芯片层温度(°C)。
     计时 = 单次 HotSpot 墙钟时间 (不含几何文件写盘, 那部分 ~ms 可忽略)。
  2) GNN+HRNet 模型 (checkpoints/gnnhrnet_pwin/best.pth): 用与训练完全一致的输入
     (graph + 3 通道场栅格, 复用 gnnhrnet.load_case 的构图逻辑)。
     计时两种口径: 单 case 前向 (batch=1, GPU 核时间) + 批量吞吐 (bs=训练批大小 32)。

输出:
  - 每个 case 的单次时间 (hotspot_s / model_forward_ms / model_build_ms) -> {out_root}/times.csv
  - 汇总 (单次时间 mean/median/min/max + 批量吞吐 + 总时间 + 加速比) -> {out_root}/summary.txt
  - 热模型几何/温度文件 -> {out_root}/config, {out_root}/thermal_map, ... (与 gen_thermal_dataset 同构)

用法 (chipdiffusion env, cwd=thermalmodel):
  python comp_thmodel_time.py --n_cases 100                     # 完整跑 100 case (约 20min 单线程)
  python comp_thmodel_time.py --n_cases 100 --no_hotspot        # 只计时模型 (快速)
  python comp_thmodel_time.py --n_cases 100 --no_model          # 只计时 HotSpot

注意: 模型结构参数 (--base 96 等) 必须与训练时 auto_train.sh 一致 (已按 gnnhrnet_pwin 写死)。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
# 预处理后: body(芯片本体)坐标 + 每个 chiplet 的 hubump 字段 (见 preprocess_bump_region.py)。
PLACE_DATASET = DATASET / "placement_dataset" / "placement_dataset_tw"

RESULT_EVAL = PROJECT / "resultEval"
sys.path.insert(0, str(RESULT_EVAL))
import run_hotspot as rh  # noqa: E402

THERMALMODEL = Path(__file__).resolve().parent
sys.path.insert(0, str(THERMALMODEL))
import dataLoader as dl  # noqa: E402
import gnnhrnet as g  # noqa: E402

# 使用项目顶层 hotspot/ 的 hotspot 二进制 (与 gen_thermal_dataset 一致)。
rh.HOTSPOT_BIN = PROJECT / "hotspot" / "hotspot"

GRID = 64  # 温度 / HotSpot 网格 (与 thermal_dataset_64、模型 grid 一致)
KELVIN = 273.15
CHUNK = 5000  # 每个 chiplet_dataset_{k}.json 含 5000 systems

# 模型结构参数 (与 auto_train.sh / checkpoints/gnnhrnet_pwin/best.pth 一致)。
MODEL_ARGS = dict(hidden=128, heads=4, num_layers=3, grid=GRID, base=96,
                  stages=4, blocks_per_stage=2, expand_ratio=2)


# --------------------------------------------------------------------------- #
# 数据读取 / 几何
# --------------------------------------------------------------------------- #
def load_record(i: int) -> dict:
    """从 placement_dataset/chiplet_dataset_*.json 读取 system_{i}。"""
    k = (i - 1) // CHUNK + 1
    fp = PLACE_DATASET / f"chiplet_dataset_{k}.json"
    data = json.loads(fp.read_text(encoding="utf-8"))
    return data[f"system_{i}"]


def build_chiplets(record: dict, j: int = 0) -> list[rh.Chiplet]:
    """构造 Chiplet 列表。j=0: 原始功耗 (placement 记录里存好的)。"""
    powers = [float(c.get("power", 0.0)) for c in record["chiplets"]]
    chiplets = []
    for idx, c in enumerate(record["chiplets"]):
        chiplets.append(rh.Chiplet(
            json_name=str(c.get("name", f"C{idx}")),
            x_mm=float(c["x-position"]), y_mm=float(c["y-position"]),
            w_mm=float(c["width"]), h_mm=float(c["height"]),
            rotation=float(c.get("rotation", 0.0)), power_w=powers[idx]))
    return chiplets


def layout_geometry(chiplets: list[rh.Chiplet], hubumps: list[float]):
    """复刻 gen_thermal_dataset 的 bbox / interposer 方形尺寸 / 居中位移。"""
    min_left, min_bottom, max_right, max_top = rh._layout_bbox_mm(chiplets, hubumps)
    span_w = max_right - min_left
    span_h = max_top - min_bottom
    intp_size_mm = max(span_w, span_h) + rh.GRANULARITY_MM
    slack_x = intp_size_mm - span_w
    slack_y = intp_size_mm - span_h
    shift_x = (rh.GRANULARITY_MM / 2.0) - min_left + (slack_x - rh.GRANULARITY_MM) / 2.0
    shift_y = (rh.GRANULARITY_MM / 2.0) - min_bottom + (slack_y - rh.GRANULARITY_MM) / 2.0
    return intp_size_mm, shift_x, shift_y


def write_geom(case_dir: Path, case: str, chiplets: list[rh.Chiplet],
               hubumps: list[float], shift_x: float, shift_y: float, intp_size_mm: float):
    """写 6 层热模型几何文件 (L0~L5 flp / layers.lcf / new_hotspot.config / system.flp),
    与 gen_thermal_dataset._run_one 的步骤 2 一致。已存在则复用 (幂等)。"""
    l4_filled = case_dir / f"{case}L4_ChipLayer.flp"
    layers_lcf = case_dir / f"{case}layers.lcf"
    derived_cfg = case_dir / "new_hotspot.config"
    system_flp = case_dir / "system.flp"
    if l4_filled.exists() and layers_lcf.exists() and derived_cfg.exists() and system_flp.exists():
        return l4_filled, layers_lcf, derived_cfg

    rh._write_simple_layer(case_dir / f"{case}L0_Substrate.flp",
                           "Floorplan for Substrate Layer", "Substrate", intp_size_mm / 1000.0)
    rh._write_simple_layer(case_dir / f"{case}L1_C4Layer.flp", "Floorplan for C4 Layer ", "C4Layer",
                           intp_size_mm / 1000.0, rh.MATERIALS["mat_c4"])
    rh._write_simple_layer(case_dir / f"{case}L2_Interposer.flp", "Floorplan for Silicon Interposer Layer",
                           "Interposer", intp_size_mm / 1000.0, rh.MATERIALS["mat_tsv"])
    l3_filled, l4_filled, _ = rh._write_l3_l4_sim(case_dir, case, chiplets, hubumps, shift_x, shift_y, intp_size_mm)
    rh._write_simple_layer(case_dir / f"{case}L5_TIM.flp", "Floorplan for TIM Layer ", "TIM",
                           intp_size_mm / 1000.0)

    with layers_lcf.open("w", encoding="utf-8") as lcf:
        lcf.write("# File Format:\n#<Layer Number>\n#<Lateral heat flow Y/N?>\n#<Power Dissipation Y/N?>\n"
                  "#<Specific heat capacity in J/(m^3K)>\n#<Resistivity in (m-K)/W>\n#<Thickness in m>\n#<floorplan file>\n")
        lcf.write("\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n" + str(case_dir / f"{case}L0_Substrate.flp") + "\n")
        lcf.write("\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n" + str(case_dir / f"{case}L1_C4Layer.flp") + "\n")
        lcf.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n" + str(case_dir / f"{case}L2_Interposer.flp") + "\n")
        lcf.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n" + str(l3_filled) + "\n")
        lcf.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n" + str(l4_filled) + "\n")
        lcf.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n" + str(case_dir / f"{case}L5_TIM.flp") + "\n")

    rh._derive_hotspot_config(rh.HOTSPOT_TEMPLATE_CONFIG, derived_cfg, intp_size_mm, grid=GRID)

    # system.flp (仅 chiplet body, 米单位), 供 load_case 解析 body rects。
    chiplet_rects_mm = [(c.x_mm + shift_x, c.y_mm + shift_y, c.w_mm, c.h_mm, c.power_w) for c in chiplets]
    lines = ["# chiplet layout for occupancy mask (meters)", "# <name>\t<width>\t<height>\t<x>\t<y>"]
    for k, (x, y, w, h, _p) in enumerate(chiplet_rects_mm):
        lines.append(f"Chiplet_{k}\t{w / 1000.0:.6f}\t{h / 1000.0:.6f}\t{x / 1000.0:.6f}\t{y / 1000.0:.6f}")
    system_flp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return l4_filled, layers_lcf, derived_cfg


# --------------------------------------------------------------------------- #
# HotSpot 运行 + 温度提取
# --------------------------------------------------------------------------- #
def run_hotspot(derived_cfg: Path, l4_filled: Path, ptrace: Path, steady: Path,
                grid_steady: Path, layers_lcf: Path) -> float:
    """单线程运行 HotSpot, 返回墙钟秒数。"""
    t0 = time.perf_counter()
    rc, _stdout, stderr = rh._run_hotspot(rh.HOTSPOT_BIN, derived_cfg, l4_filled, ptrace,
                                          steady, grid_steady, layers_lcf, "grid")
    dt = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"hotspot rc={rc}\nSTDERR:\n{stderr[:800]}")
    return dt


def read_grid_steady_c(path: Path) -> np.ndarray:
    """读 grid_steady (扁平单层 '<idx>\t<temp_K>'), 返回 (GRID,GRID) 的摄氏度,
    row 0 = 底部 (垂直翻转与 power_map 对齐, 同 gen_thermal_dataset)。"""
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                vals.append(float(parts[1]))
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size != GRID * GRID:
        raise RuntimeError(f"grid_steady 值个数 {arr.size} != {GRID * GRID}")
    arr = arr.reshape(GRID, GRID)
    return np.flipud(arr) - KELVIN


# --------------------------------------------------------------------------- #
# 模型输入构建 (与 gnnhrnet.load_case 一致, 但跳过温度真值读取)
# --------------------------------------------------------------------------- #
def build_model_input(case_dir: Path, i: int, j: int):
    """从已生成的 config 构建单 case 模型输入: (x, edge_index, edge_attr, batch, field)。"""
    cfg = str(case_dir)
    flp = os.path.join(cfg, "system.flp")
    l4 = os.path.join(cfg, f"system_{i}L4_ChipLayer.flp")
    ptrace = os.path.join(cfg, f"system_{i}_{j}.ptrace")

    rects = [(x, y, w, h, name) for (x, y, w, h, name) in dl.parse_flp_rects(flp)
             if name.startswith("Chiplet")]
    pw = g.read_ptrace_powers(ptrace)
    side_mm = dl.interposer_side_m(l4) * 1000.0

    rects_mm = [(x * 1000.0, y * 1000.0, w * 1000.0, h * 1000.0) for (x, y, w, h, _) in rects]
    powers_w = [pw[name] for (_, _, _, _, name) in rects]
    hubump_w, ubump_rects_mm = g.parse_l4_thermal(l4)
    hubump_mm = [hubump_w.get(name, 0.0) * 1000.0 for (_, _, _, _, name) in rects]
    graph = g.build_graph_from_rects(rects_mm, powers_w, side_mm, grid=GRID, hubump_mm=hubump_mm)

    areas_mm2 = [w * h for (_, _, w, h) in rects_mm]
    p_density = [p / a if a > 0 else 0.0 for p, a in zip(powers_w, areas_mm2)]
    p_grid = g.rasterize_rects_mm(rects_mm, side_mm, GRID,
                                  values=[d / g.POWER_GRID_SCALE for d in p_density])
    mask_grid = g.rasterize_rects_mm(rects_mm, side_mm, GRID)
    hubump_grid = g.rasterize_rects_mm(ubump_rects_mm, side_mm, GRID)
    field_raster = np.stack([p_grid, mask_grid, hubump_grid], axis=0).astype(np.float32)

    field_t = torch.from_numpy(field_raster).unsqueeze(0)  # [1,3,64,64]
    n = graph.x.size(0)
    batch = torch.zeros(n, dtype=torch.long)
    return graph.x, graph.edge_index, graph.edge_attr, batch, field_t


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = g.GNNHRNetModel(node_dim=8, **MODEL_ARGS).to(device)
    ck = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    return model


def collate_inputs(items):
    """把多个 build_model_input 输出拼成一个 batch (graph 索引重新编号)。"""
    xs, eis, eas, bs, fs = [], [], [], [], []
    node_off = 0
    for gidx, (x, ei, ea, _b, f) in enumerate(items):
        k = x.size(0)
        xs.append(x)
        eis.append(ei + node_off)
        eas.append(ea)
        bs.append(torch.full((k,), gidx, dtype=torch.long))
        fs.append(f)
        node_off += k
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.cat(eas), torch.cat(bs), torch.cat(fs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=300001,
                    help="起始 system 编号 (30w 往后, 默认 300001)")
    ap.add_argument("--n_cases", type=int, default=100, help="case 数量 (默认 100)")
    ap.add_argument("--j", type=int, default=0, help="功耗配置 j (默认 0=原始功耗)")
    ap.add_argument("--grid", type=int, default=GRID, help="HotSpot 温度网格 (默认 64)")
    ap.add_argument("--ckpt", type=str, default="checkpoints/gnnhrnet_pwin/best.pth")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--out_root", type=str, default="time_eval_out",
                    help="输出根目录 (config + 温度 + times.csv + summary.txt)")
    ap.add_argument("--run_hotspot", action="store_true", default=True,
                    help="运行 HotSpot 计时 (默认开)")
    ap.add_argument("--no_hotspot", action="store_true", help="跳过 HotSpot 计时")
    ap.add_argument("--run_model", action="store_true", default=True,
                    help="运行模型推理计时 (默认开)")
    ap.add_argument("--no_model", action="store_true", help="跳过模型计时")
    ap.add_argument("--save_temp", action="store_true", help="额外落盘 thermal_map/maxtemp/avgtemp CSV")
    ap.add_argument("--warmup", type=int, default=5, help="模型前向 warmup 次数 (不计时)")
    ap.add_argument("--eval_bs", type=int, default=32, help="模型批量吞吐的 batch size (默认 32, 与训练一致)")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    global GRID
    GRID = int(args.grid)

    do_hotspot = args.run_hotspot and not args.no_hotspot
    do_model = args.run_model and not args.no_model

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = THERMALMODEL / out_root
    cfg_dir = out_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    model = None
    if do_model:
        model = load_model(args.ckpt, device)

    cases = list(range(args.start, args.start + args.n_cases))
    print(f"[comp_time] start={args.start} n={len(cases)} (system_{cases[0]}..system_{cases[-1]}) "
          f"grid={GRID} j={args.j} device={device} run_hotspot={do_hotspot} run_model={do_model}", flush=True)

    # 汇总统计
    hs_times: list[float] = []
    fwd_times: list[float] = []
    build_times: list[float] = []
    rows: list[dict] = []
    model_inputs: list = []
    n_chiplets = None

    t_all0 = time.perf_counter()
    for idx, i in enumerate(cases, 1):
        record = load_record(i)
        case = f"system_{i}"
        case_dir = cfg_dir / f"system_{i}_config"
        case_dir.mkdir(parents=True, exist_ok=True)

        chiplets = build_chiplets(record, j=args.j)
        hubumps = [float(c.get("hubump", 0.0)) for c in record["chiplets"]]
        intp_size_mm, shift_x, shift_y = layout_geometry(chiplets, hubumps)
        l4_filled, layers_lcf, derived_cfg = write_geom(case_dir, case, chiplets, hubumps,
                                                        shift_x, shift_y, intp_size_mm)

        row = {"i": i, "n_chiplets": len(chiplets), "side_mm": round(intp_size_mm, 3)}
        n_chiplets = len(chiplets)

        # ptrace (功耗轨迹) —— HotSpot 与模型输入都需要, 总是写
        ptrace = case_dir / f"{case}_{args.j}.ptrace"
        powers_by_name = {f"Chiplet_{k}": ch.power_w for k, ch in enumerate(chiplets)}
        rh._write_ptrace_from_flp(l4_filled, ptrace, powers_by_name)

        # --- HotSpot ---
        if do_hotspot:
            steady = case_dir / f"{case}_{args.j}.steady"
            grid_steady = case_dir / f"{case}_{args.j}.grid.steady"
            dt = run_hotspot(derived_cfg, l4_filled, ptrace, steady, grid_steady, layers_lcf)
            hs_times.append(dt)
            row["hotspot_s"] = round(dt, 4)
            temp_c = read_grid_steady_c(grid_steady)
            row["temp_max_C"] = round(float(temp_c.max()), 3)
            row["temp_avg_C"] = round(float(temp_c.mean()), 3)
            if args.save_temp:
                tm = out_root / "thermal_map"; tm.mkdir(parents=True, exist_ok=True)
                (out_root / "max_temp").mkdir(parents=True, exist_ok=True)
                (out_root / "avg_temp").mkdir(parents=True, exist_ok=True)
                with (tm / f"system_temp_{i}_{args.j}.csv").open("w") as f:
                    for k, v in enumerate(temp_c.reshape(-1), start=1):
                        f.write(f"{k},{v:.6f}\n")
                (out_root / "max_temp" / f"system_maxtemp_{i}_{args.j}.csv").write_text(f"{temp_c.max():.6f}\n")
                (out_root / "avg_temp" / f"system_avgtemp_{i}_{args.j}.csv").write_text(f"{temp_c.mean():.6f}\n")

        # --- 模型前向 ---
        if do_model:
            tb0 = time.perf_counter()
            x, ei, ea, batch, field = build_model_input(case_dir, i, args.j)
            tb1 = time.perf_counter()
            build_times.append(tb1 - tb0)
            row["model_build_ms"] = round((tb1 - tb0) * 1e3, 3)
            model_inputs.append((x, ei, ea, batch, field))  # CPU 输入, 供批量吞吐

            x, ei, ea, batch, field = (t.to(device) for t in (x, ei, ea, batch, field))
            if device.type == "cuda":
                torch.cuda.synchronize()
                if idx == 1:
                    # warmup: 触发 CUDA 内核编译 / cuDNN autotune, 不计时
                    with torch.no_grad():
                        for _ in range(args.warmup):
                            model(x, ei, batch, ea, field)
                    torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                with torch.no_grad():
                    model(x, ei, batch, ea, field)
                end.record()
                torch.cuda.synchronize()
                fwd_s = start.elapsed_time(end) / 1e3
            else:
                f0 = time.perf_counter()
                with torch.no_grad():
                    model(x, ei, batch, ea, field)
                fwd_s = time.perf_counter() - f0
            fwd_times.append(fwd_s)
            row["model_fwd_ms"] = round(fwd_s * 1e3, 4)

        rows.append(row)

        hs = f"hotspot={row['hotspot_s']:.3f}s" if "hotspot_s" in row else "hotspot=skip"
        fw = f"fwd={row['model_fwd_ms']:.3f}ms build={row['model_build_ms']:.2f}ms" if "model_fwd_ms" in row else "model=skip"
        print(f"[{idx}/{len(cases)}] {case} n_chip={row['n_chiplets']} side={row['side_mm']}mm  {hs}  {fw}", flush=True)

    t_all = time.perf_counter() - t_all0

    # --- 批量推理吞吐 (bs=训练批大小 32) ---
    batched_per_case_ms = batched_total_s = None
    if do_model and model_inputs:
        bs = args.eval_bs
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(0, len(model_inputs), bs):
            x, ei, ea, b, f = collate_inputs(model_inputs[s:s + bs])
            x, ei, ea, b, f = (t.to(device) for t in (x, ei, ea, b, f))
            with torch.no_grad():
                model(x, ei, b, ea, f)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batched_total_s = time.perf_counter() - t0
        batched_per_case_ms = batched_total_s / len(model_inputs) * 1e3

    # --- 写 per-case times.csv ---
    csv_path = out_root / "times.csv"
    cols = ["i", "n_chiplets", "side_mm", "hotspot_s", "temp_max_C", "temp_avg_C",
            "model_build_ms", "model_fwd_ms"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")

    # --- 汇总 ---
    def _stat(vals_ms):
        if not vals_ms:
            return None
        return dict(mean=statistics.mean(vals_ms), median=statistics.median(vals_ms),
                    min=min(vals_ms), max=max(vals_ms), total=sum(vals_ms), n=len(vals_ms))

    hs_ms = [t * 1e3 for t in hs_times]
    hs_s = _stat(hs_ms)
    fwd_s = _stat([t * 1e3 for t in fwd_times])  # ms
    build_s = _stat([t * 1e3 for t in build_times])

    speedup = (hs_s["mean"] / fwd_s["mean"]) if (hs_s and fwd_s) else None

    L = []
    L.append("# comp_thmodel_time summary")
    L.append(f"date={time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"cases: system_{args.start}..system_{args.start + args.n_cases - 1}  n={args.n_cases}  j={args.j}  grid={GRID}")
    L.append(f"device={device}  ckpt={args.ckpt}  n_chiplets={n_chiplets}")
    L.append("")
    if hs_s:
        L.append(f"[HotSpot 单线程, 单 case 墙钟时间] n={hs_s['n']}")
        L.append(f"  mean={hs_s['mean']:.3f} ms/case   median={hs_s['median']:.3f} ms")
        L.append(f"  min={hs_s['min']:.3f} ms   max={hs_s['max']:.3f} ms")
        L.append(f"  total={hs_s['total'] / 1e3 / 3600:.3f} h  ({hs_s['total'] / 1e3:.1f} s)")
    if fwd_s:
        L.append(f"[模型前向 (batch=1, GPU 核时间)] n={fwd_s['n']}")
        L.append(f"  mean={fwd_s['mean']:.4f} ms/case   median={fwd_s['median']:.4f} ms")
        L.append(f"  min={fwd_s['min']:.4f} ms   max={fwd_s['max']:.4f} ms")
        L.append(f"  total={fwd_s['total']:.3f} ms  ({fwd_s['total'] / 1e3:.3f} s)")
    if build_s:
        L.append(f"[模型输入构建 (CPU, 单 case)] mean={build_s['mean']:.3f} ms  total={build_s['total'] / 1e3:.3f} s")
    if batched_per_case_ms is not None:
        L.append(f"[模型批量推理 (bs={args.eval_bs}, 训练批大小)] per_case={batched_per_case_ms:.4f} ms  "
                 f"total={batched_total_s:.3f} s  ({len(model_inputs) / batched_total_s:.0f} cases/s)")
    if speedup:
        L.append(f"[加速比] HotSpot 单线程 / 模型单次前向(batch=1) = {speedup:.0f}x")
    if hs_s and batched_per_case_ms:
        L.append(f"[加速比] HotSpot 单线程 / 模型批量(bs={args.eval_bs}) = {hs_s['mean'] / batched_per_case_ms:.0f}x")
    L.append(f"[总墙钟] 本脚本 {t_all:.1f} s ({t_all / 3600:.3f} h)")
    L.append(f"[files] times.csv={csv_path}")

    summary = "\n".join(L)
    (out_root / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    print("\n" + summary)


if __name__ == "__main__":
    main()
