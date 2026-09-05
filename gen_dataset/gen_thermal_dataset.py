#!/usr/bin/env python3
"""
为 placement_dataset/placement_dataset_tw (body 坐标 + 每个 chiplet 的 hubump 字段) 生成 128×128 热仿真数据集,
每个布局一份 (原功耗, 不再生成重随机功耗变体 j=1)。hubump 直接从 body 记录读取, 不重新计算。

热模型参照 resultEval/run_hotspot.py: 6 层 TAP-2.5D 堆叠, -detailed_3D on, grid 128×128。
HotSpot 二进制用 resultEval/util/hotspot (扁平单层 128×128 grid_steady = 芯片层 Layer 4 热图)。

输出目录 (Dataset/dataset/thermal_dataset/):
  config/system_{i}_config/  热模型文件(L0~L5 flp, layers.lcf, new_hotspot.config,
                              system_{i}_{j}.ptrace/.steady/.grid.steady, system.flp 布局mask)
  power_map/    system_power_{i}_{j}.csv     128×128 功耗网格 (idx,value, 1-based, W)
  total_power/  system_totalpower_{i}_{j}.csv 标量(总功耗, W)
  max_temp/     system_maxtemp_{i}_{j}.csv    标量(芯片层最高温, ℃)
  thermal_map/  system_temp_{i}_{j}.csv      128×128 芯片层温度网格 (idx,value, ℃)
  avg_temp/     system_avgtemp_{i}_{j}.csv    标量(芯片层平均温, ℃)

用法:
  python gen_thermal_dataset.py --start 300001 --end 300010 --workers 4     # 小样本
  python gen_thermal_dataset.py --start 300001 --end 320000 --workers 28    # 2w 布局, 每份单功耗
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
# 预处理后: body(芯片本体)坐标 + 每个 chiplet 的 hubump 字段 (见 preprocess_bump_region.py)。
# 热模型拿 body + hubump 向外扩, 还原原始无重叠 footprint。
PLACE_DATASET = DATASET / "placement_dataset" / "placement_dataset_tw"
THERMAL = DATASET / "thermal_dataset"

RESULT_EVAL = PROJECT / "resultEval"
sys.path.insert(0, str(RESULT_EVAL))
from util.fill_space import fill_space  # noqa: F401
import run_hotspot as rh  # noqa: E402

# 使用项目顶层 hotspot/ 的 hotspot 二进制 (源自 TAP-2.5D/util/hotspot,
# 输出扁平单层 128×128 grid_steady = 芯片层 Layer 4 热图)。
HOTSPOT_DIR = PROJECT / "hotspot"
rh.HOTSPOT_BIN = HOTSPOT_DIR / "hotspot"

GRID = 128
KELVIN = 273.15
CHUNK = 5000  # 每个 chiplet_dataset_{k}.json 含 5000 systems

CONFIG_DIR = THERMAL / "config"
POWER_DIR = THERMAL / "power_map"
TOTAL_DIR = THERMAL / "total_power"
MAXT_DIR = THERMAL / "max_temp"
TEMP_DIR = THERMAL / "thermal_map"
AVGT_DIR = THERMAL / "avg_temp"

for _d in (CONFIG_DIR, POWER_DIR, TOTAL_DIR, MAXT_DIR, TEMP_DIR, AVGT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 数据读取 / 扩展
# --------------------------------------------------------------------------- #
def load_range(start_sys: int, end_sys: int) -> dict:
    """从 placement_dataset/chiplet_dataset_*.json 读取 system_{start_sys}..system_{end_sys}。"""
    records: dict[int, dict] = {}
    k0 = (start_sys - 1) // CHUNK + 1
    k1 = (end_sys - 1) // CHUNK + 1
    for k in range(k0, k1 + 1):
        fp = PLACE_DATASET / f"chiplet_dataset_{k}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid, rec in data.items():
            m = re.fullmatch(r"system_(\d+)", sid)
            if not m:
                continue
            i = int(m.group(1))
            if start_sys <= i <= end_sys:
                records[i] = rec
    return records


def record_powers(record: dict) -> list[float]:
    """返回该布局每个 chiplet 的原始功耗 (不再生成重随机功耗变体)。"""
    return [float(c.get("power", 0.0)) for c in record["chiplets"]]


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _read_grid_steady(path: Path, grid: int = GRID) -> np.ndarray:
    """读取 grid_steady (扁平单层, 每行 '<idx>\t<temp_K>', 每 128 行一个空行)。
    返回 shape (grid, grid) 的 float64 (Kelvin), row 0 = 底部(y=min)。

    HotSpot grid_steady 按 row 0 = 顶部 输出; 这里垂直翻转, 使其与
    power_map / dataLoader.flp_to_mask 的 bottom-origin 约定对齐。
    """
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
    if arr.size != grid * grid:
        raise RuntimeError(f"grid_steady 值个数 {arr.size} != {grid * grid}")
    arr = arr.reshape(grid, grid)  # 行优先: 原始 [y, x], row 0 = 顶部
    return np.flipud(arr)          # 翻转后 row 0 = 底部


def _write_index_value_csv(path: Path, arr: np.ndarray) -> None:
    """写 'idx,value' (idx 1-based, 行优先 y-major)。与 dataLoader.read_index_value_csv 兼容。"""
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    with open(path, "w", encoding="utf-8") as f:
        for idx, v in enumerate(a, start=1):
            f.write(f"{idx},{v:.6f}\n")


def _write_scalar_csv(path: Path, v: float) -> None:
    path.write_text(f"{float(v):.6f}\n", encoding="utf-8")


def _power_map_128(rects: list[tuple[float, float, float, float, float]], intp_size_mm: float, grid: int = GRID) -> np.ndarray:
    """把 chiplet 功耗按面积比例分摊到 grid×grid (覆盖 [0, intp_size_mm]² 方形)。
    rects: (x_mm, y_mm, w_mm, h_mm, power_w), 已平移到方形内。"""
    cell = intp_size_mm / grid
    acc = np.zeros((grid, grid), dtype=np.float64)
    for (x, y, w, h, p) in rects:
        if p == 0.0 or w <= 0 or h <= 0:
            continue
        ix0 = int(np.floor(x / cell))
        iy0 = int(np.floor(y / cell))
        ix1 = int(np.ceil((x + w) / cell))
        iy1 = int(np.ceil((y + h) / cell))
        ix0 = max(0, min(grid, ix0))
        iy0 = max(0, min(grid, iy0))
        ix1 = max(0, min(grid, ix1))
        iy1 = max(0, min(grid, iy1))
        area = w * h
        for iy in range(iy0, iy1):
            y0 = iy * cell
            y1 = y0 + cell
            for ix in range(ix0, ix1):
                x0 = ix * cell
                x1 = x0 + cell
                ox0 = max(x, x0)
                oy0 = max(y, y0)
                ox1 = min(x + w, x1)
                oy1 = min(y + h, y1)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                inter = (ox1 - ox0) * (oy1 - oy0)
                acc[iy, ix] += p * (inter / area)
    return acc


def _write_system_flp(path: Path, rects: list[tuple[float, float, float, float, float]]) -> None:
    """写 system.flp (仅 chiplet 矩形, 米单位), 供 dataLoader.flp_to_mask 生成布局 mask。"""
    lines = ["# chiplet layout for occupancy mask (meters)", "# <name>\t<width>\t<height>\t<x>\t<y>"]
    for k, (x, y, w, h, _p) in enumerate(rects):
        lines.append(f"Chiplet_{k}\t{w / 1000.0:.6f}\t{h / 1000.0:.6f}\t{x / 1000.0:.6f}\t{y / 1000.0:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# 单个 (i, j) 的完整流程
# --------------------------------------------------------------------------- #
def _run_one(i: int, j: int, record: dict, case_dir: Path, case: str) -> str:
    t_csv = TEMP_DIR / f"system_temp_{i}_{j}.csv"
    p_csv = POWER_DIR / f"system_power_{i}_{j}.csv"
    if t_csv.exists() and p_csv.exists():
        return "skip"

    # 1) 构造 Chiplet 列表 (原功耗)
    # 记录里已是 body 坐标(预处理后), 并带每个 chiplet 的 hubump 字段。
    powers = record_powers(record)
    chiplets = []
    for idx, c in enumerate(record["chiplets"]):
        name = str(c.get("name", f"C{idx}"))
        x = float(c["x-position"])
        y = float(c["y-position"])
        w = float(c["width"])
        h = float(c["height"])
        chiplets.append(rh.Chiplet(json_name=name, x_mm=x, y_mm=y, w_mm=w, h_mm=h,
                                   rotation=float(c.get("rotation", 0.0)), power_w=powers[idx]))
    connections = record.get("connections", [])

    # 2) 建 6 层热模型 (复刻 resultEval/run_hotspot.py main)
    # 用预处理存下的 hubump 把 body 还原为 footprint(body + bump ring),
    # 与原始无重叠布局一致。hubump 只取决于互连线数量 wireCount, 与 EMIB 无关。
    hubumps = [float(c.get("hubump", 0.0)) for c in record["chiplets"]]
    min_left, min_bottom, max_right, max_top = rh._layout_bbox_mm(chiplets, hubumps)
    span_w = max_right - min_left
    span_h = max_top - min_bottom
    intp_size_mm = max(span_w, span_h) + rh.GRANULARITY_MM
    slack_x = intp_size_mm - span_w
    slack_y = intp_size_mm - span_h
    shift_x = (rh.GRANULARITY_MM / 2.0) - min_left + (slack_x - rh.GRANULARITY_MM) / 2.0
    shift_y = (rh.GRANULARITY_MM / 2.0) - min_bottom + (slack_y - rh.GRANULARITY_MM) / 2.0

    rh._write_simple_layer(case_dir / f"{case}L0_Substrate.flp",
                           "Floorplan for Substrate Layer with size " + str(intp_size_mm / 1000.0) + "x" + str(intp_size_mm / 1000.0) + " m",
                           "Substrate", intp_size_mm / 1000.0)
    rh._write_simple_layer(case_dir / f"{case}L1_C4Layer.flp", "Floorplan for C4 Layer ", "C4Layer",
                           intp_size_mm / 1000.0, rh.MATERIALS["mat_c4"])
    rh._write_simple_layer(case_dir / f"{case}L2_Interposer.flp", "Floorplan for Silicon Interposer Layer", "Interposer",
                           intp_size_mm / 1000.0, rh.MATERIALS["mat_tsv"])
    l3_filled, l4_filled, _ = rh._write_l3_l4_sim(case_dir, case, chiplets, hubumps, shift_x, shift_y, intp_size_mm)
    rh._write_simple_layer(case_dir / f"{case}L5_TIM.flp", "Floorplan for TIM Layer ", "TIM", intp_size_mm / 1000.0)

    layers_lcf = case_dir / f"{case}layers.lcf"
    with layers_lcf.open("w", encoding="utf-8") as lcf:
        lcf.write("# File Format:\n#<Layer Number>\n#<Lateral heat flow Y/N?>\n#<Power Dissipation Y/N?>\n"
                  "#<Specific heat capacity in J/(m^3K)>\n#<Resistivity in (m-K)/W>\n#<Thickness in m>\n#<floorplan file>\n")
        lcf.write("\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n" + str(case_dir / f"{case}L0_Substrate.flp") + "\n")
        lcf.write("\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n" + str(case_dir / f"{case}L1_C4Layer.flp") + "\n")
        lcf.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n" + str(case_dir / f"{case}L2_Interposer.flp") + "\n")
        lcf.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n" + str(l3_filled) + "\n")
        lcf.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n" + str(l4_filled) + "\n")
        lcf.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n" + str(case_dir / f"{case}L5_TIM.flp") + "\n")

    derived_cfg = case_dir / "new_hotspot.config"
    rh._derive_hotspot_config(rh.HOTSPOT_TEMPLATE_CONFIG, derived_cfg, intp_size_mm)

    ptrace = case_dir / f"{case}_{j}.ptrace"
    powers_by_name = {f"Chiplet_{k}": ch.power_w for k, ch in enumerate(chiplets)}
    rh._write_ptrace_from_flp(l4_filled, ptrace, powers_by_name)

    # 3) 跑 HotSpot
    steady = case_dir / f"{case}_{j}.steady"
    grid_steady = case_dir / f"{case}_{j}.grid.steady"
    rc, stdout, stderr = rh._run_hotspot(rh.HOTSPOT_BIN, derived_cfg, l4_filled, ptrace, steady, grid_steady, layers_lcf, "grid")
    if rc != 0:
        raise RuntimeError(f"hotspot rc={rc}\nSTDERR:\n{stderr[:800]}")

    # 4) 提取温度 (芯片层 128×128, 开尔文 -> 摄氏度)
    temp_K = _read_grid_steady(grid_steady)
    temp_C = temp_K - KELVIN
    _write_index_value_csv(TEMP_DIR / f"system_temp_{i}_{j}.csv", temp_C)
    _write_scalar_csv(MAXT_DIR / f"system_maxtemp_{i}_{j}.csv", float(temp_C.max()))
    _write_scalar_csv(AVGT_DIR / f"system_avgtemp_{i}_{j}.csv", float(temp_C.mean()))

    # 5) 功耗图 (128×128, 覆盖 interposer 方形)
    chiplet_rects_mm = [(c.x_mm + shift_x, c.y_mm + shift_y, c.w_mm, c.h_mm, c.power_w) for c in chiplets]
    acc = _power_map_128(chiplet_rects_mm, intp_size_mm)
    _write_index_value_csv(POWER_DIR / f"system_power_{i}_{j}.csv", acc)

    # 6) 总功耗
    _write_scalar_csv(TOTAL_DIR / f"system_totalpower_{i}_{j}.csv", float(sum(c.power_w for c in chiplets)))

    # 7) system.flp (布局 mask, 仅 j=0 写一次)
    if j == 0:
        _write_system_flp(case_dir / "system.flp", chiplet_rects_mm)

    return "ok"


def process_layout(args) -> tuple[int, str]:
    i, record = args
    case = f"system_{i}"
    case_dir = CONFIG_DIR / f"system_{i}_config"
    case_dir.mkdir(parents=True, exist_ok=True)
    try:
        r0 = _run_one(i, 0, record, case_dir, case)
        return i, f"j0={r0}"
    except Exception as e:  # noqa: BLE001
        return i, f"ERROR: {e}\n{traceback.format_exc(limit=2)}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=300001)
    ap.add_argument("--end", type=int, default=340000)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()

    records = load_range(args.start, args.end)
    items = sorted(records.items())
    print(f"[thermal] 读取布局 {args.start}..{args.end}: {len(items)} 个 (单功耗, 共 {len(items)} 份)", flush=True)
    if not items:
        print("[thermal] 无数据, 退出")
        return

    t0 = time.time()
    done = 0
    errs = 0
    with Pool(args.workers) as pool:
        for i, msg in pool.imap_unordered(process_layout, items):
            done += 1
            if msg.startswith("ERROR"):
                errs += 1
                print(f"[thermal] [{done}/{len(items)}] system_{i}: {msg}", flush=True)
            else:
                print(f"[thermal] [{done}/{len(items)}] system_{i}: {msg}", flush=True)

    print(f"[thermal] DONE: {len(items)} 布局({len(items)} 份) 用时 {time.time() - t0:.1f}s, 失败 {errs}", flush=True)


if __name__ == "__main__":
    main()
