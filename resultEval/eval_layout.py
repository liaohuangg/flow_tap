#!/usr/bin/env python3
"""eval_layout.py — 评估 AT / RL 布局的线长、温度、外接框面积。

输入 (两个转换脚本产出的 placement_dataset 格式布局):
  AT_result/format_result/<前缀>_seed<M>.json
  RL_result/format_result/<前缀>_seed<M>.json

三个指标, 全部复用生成数据集时的原始代码, 不另起一套实现:

  1. 线长 (gen_dataset/gen_wirelength_dataset.py)
     TapSystem(record) + solve_cplex_avg(system) —— TAP-2.5D 的 microbump 布线 ILP
     (CPLEX 22.1.0), min Σ d·f:
         total_wirelength = 目标函数值 (mm)
         avg_wirelength   = total_wirelength / wire_count (mm/根)
     hubump 用 TapSystem 默认的 "die" 模式重算 (已校验与布局里存的 hubump 逐条相等)。

  2. 温度 (gen_dataset/gen_thermal_dataset.py 里 _run_one 的热仿真段)
     6 层 TAP-2.5D 堆叠 + HotSpot (-detailed_3D on), 功耗取布局自带的 power 字段,
     hubump 取布局里存的 hubump 把 body 还原成 footprint。读 <case>.grid.steady 得
     芯片层温度场 (℃):
         avg_temp = 温度场均值
         max_temp = 温度场最大值

  3. 外接框面积 —— 所有 chiplet **含 bump 环**的轴对齐外接矩形:
         bbox = rh._layout_bbox_mm(chiplets, hubumps)
         area = (max_right - min_left) * (max_top - min_bottom)   # mm²
     (即外接框含 hubump; HotSpot 里额外加的 1mm granularity 边距不算在内。)

输出:
  RL_result/result.csv   每行 = 一个 (case, seed)
  AT_result/result.csv

用法:
  python eval_layout.py --method AT                 # 只跑 AT
  python eval_layout.py --method both --stage wl    # 只算线长 (快)
  python eval_layout.py --method both               # 线长 + 热仿真
  python eval_layout.py --method AT --only Case6_seed1

增量: 结果缓存在 <method>_result/eval_cache.json, 重跑只补缺失的项。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
RESULT_EVAL = PROJECT / "resultEval"
GEN_DATASET = PROJECT / "gen_dataset"

# 两个生成脚本都靠 sys.path 找 util/ 与彼此, 先把路径铺好再 import
sys.path.insert(0, str(RESULT_EVAL))
sys.path.insert(0, str(GEN_DATASET))

import numpy as np  # noqa: E402

import run_hotspot as rh  # noqa: E402  (resultEval/run_hotspot.py)
import gen_thermal_dataset as gtd  # noqa: E402  (热仿真的原始实现)
import gen_wirelength_dataset as gwd  # noqa: E402  (线长的原始实现)

# 与 gen_thermal_dataset.py 一致: 用项目顶层 hotspot/ 的二进制
rh.HOTSPOT_BIN = PROJECT / "hotspot" / "hotspot"

METHOD_DIR = {
    "AT": RESULT_EVAL / "AT_result",
    "RL": RESULT_EVAL / "RL_result",
}
LAYOUT_SUBDIR = "format_result"
EVAL_SUBDIR = "eval_out"          # HotSpot 中间文件
CACHE_NAME = "eval_cache.json"
CSV_NAME = "result.csv"

KELVIN = 273.15
GRID_DEFAULT = 64

CSV_COLUMNS = [
    "case", "seed", "n_chiplets", "n_connections",
    "total_wirelength_mm", "avg_wirelength_mm",
    "avg_temp_C", "max_temp_C",
    "bbox_area_mm2", "bbox_width_mm", "bbox_height_mm",
    "wirelength_time_s", "thermal_time_s", "error",
]


# --------------------------------------------------------------------------- #
# 单个布局: 三个指标
# --------------------------------------------------------------------------- #
def _split_stem(stem: str) -> tuple[str, str]:
    """'Case6_seed1' -> ('Case6', '1')。"""
    if "_seed" in stem:
        case, seed = stem.rsplit("_seed", 1)
        return case, seed
    return stem, ""


def _bbox_mm(record: dict) -> tuple[float, float, float, float]:
    """布局的轴对齐外接框 (含 bump 环), 单位 mm。

    与 gen_thermal_dataset._run_one 用的是同一个 rh._layout_bbox_mm:
    每个 chiplet 的 footprint = body 向外扩 hubump。
    """
    chiplets = [
        rh.Chiplet(
            json_name=str(c.get("name", f"C{i}")),
            x_mm=float(c["x-position"]),
            y_mm=float(c["y-position"]),
            w_mm=float(c["width"]),
            h_mm=float(c["height"]),
            rotation=float(c.get("rotation", 0.0)),
            power_w=float(c.get("power", 0.0)),
        )
        for i, c in enumerate(record["chiplets"])
    ]
    hubumps = [float(c.get("hubump", 0.0)) for c in record["chiplets"]]
    return rh._layout_bbox_mm(chiplets, hubumps)


def eval_wirelength(record: dict) -> dict:
    """线长: 复用 gen_wirelength_dataset 的 TapSystem + solve_cplex_avg。"""
    t0 = time.time()
    system = gwd.TapSystem(record)          # hubump_mode="die", 与生成数据集时一致
    avg_wl, total_wl, _d_net, _side = gwd.solve_cplex_avg(system)
    dt = time.time() - t0
    if total_wl is None:
        # solve_cplex_avg 求解失败的哨兵返回 (avg=100.0, total=None)
        raise RuntimeError("CPLEX 未返回解 (solve_cplex_avg 失败)")
    wire_count = sum(
        system.connection_matrix[i][j]
        for i in range(system.chiplet_count) for j in range(system.chiplet_count) if i != j
    )
    return {
        "total_wirelength_mm": round(float(total_wl), 6),
        "avg_wirelength_mm": round(float(avg_wl), 6),
        "wire_count": int(round(wire_count)),
        "wirelength_time_s": round(dt, 3),
    }


def eval_thermal(record: dict, case_dir: Path, grid: int) -> dict:
    """温度: 复刻 gen_thermal_dataset._run_one 的热仿真段 (j=0, 原始功耗)。

    case_dir 下生成 <case>L*.flp / layers.lcf / new_hotspot.config / <case>.ptrace
    以及 HotSpot 输出 <case>.grid.steady。已存在 .grid.steady 时直接复用。
    """
    t0 = time.time()
    case_dir.mkdir(parents=True, exist_ok=True)
    case = case_dir.name

    chiplets = [
        rh.Chiplet(
            json_name=str(c.get("name", f"C{i}")),
            x_mm=float(c["x-position"]),
            y_mm=float(c["y-position"]),
            w_mm=float(c["width"]),
            h_mm=float(c["height"]),
            rotation=float(c.get("rotation", 0.0)),
            power_w=float(c.get("power", 0.0)),
        )
        for i, c in enumerate(record["chiplets"])
    ]
    hubumps = [float(c.get("hubump", 0.0)) for c in record["chiplets"]]

    min_left, min_bottom, max_right, max_top = rh._layout_bbox_mm(chiplets, hubumps)
    span_w = max_right - min_left
    span_h = max_top - min_bottom
    intp_size_mm = max(span_w, span_h) + rh.GRANULARITY_MM
    slack_x = intp_size_mm - span_w
    slack_y = intp_size_mm - span_h
    shift_x = (rh.GRANULARITY_MM / 2.0) - min_left + (slack_x - rh.GRANULARITY_MM) / 2.0
    shift_y = (rh.GRANULARITY_MM / 2.0) - min_bottom + (slack_y - rh.GRANULARITY_MM) / 2.0

    l4_filled = case_dir / f"{case}L4_ChipLayer.flp"
    layers_lcf = case_dir / f"{case}layers.lcf"
    derived_cfg = case_dir / "new_hotspot.config"

    # 几何文件与功耗无关, 已存在就复用
    if not (l4_filled.exists() and layers_lcf.exists() and derived_cfg.exists()):
        rh._write_simple_layer(
            case_dir / f"{case}L0_Substrate.flp",
            "Floorplan for Substrate Layer with size " + str(intp_size_mm / 1000.0) + "x" + str(intp_size_mm / 1000.0) + " m",
            "Substrate", intp_size_mm / 1000.0)
        rh._write_simple_layer(case_dir / f"{case}L1_C4Layer.flp", "Floorplan for C4 Layer ", "C4Layer",
                               intp_size_mm / 1000.0, rh.MATERIALS["mat_c4"])
        rh._write_simple_layer(case_dir / f"{case}L2_Interposer.flp", "Floorplan for Silicon Interposer Layer", "Interposer",
                               intp_size_mm / 1000.0, rh.MATERIALS["mat_tsv"])
        _l3, l4_filled, _sim = rh._write_l3_l4_sim(case_dir, case, chiplets, hubumps, shift_x, shift_y, intp_size_mm)
        rh._write_simple_layer(case_dir / f"{case}L5_TIM.flp", "Floorplan for TIM Layer ", "TIM", intp_size_mm / 1000.0)

        with layers_lcf.open("w", encoding="utf-8") as lcf:
            lcf.write("# File Format:\n#<Layer Number>\n#<Lateral heat flow Y/N?>\n#<Power Dissipation Y/N?>\n"
                      "#<Specific heat capacity in J/(m^3K)>\n#<Resistivity in (m-K)/W>\n#<Thickness in m>\n#<floorplan file>\n")
            lcf.write("\n# Layer 0: substrate\n0\nY\nN\n1.06E+06\n3.33\n0.0002\n" + str(case_dir / f"{case}L0_Substrate.flp") + "\n")
            lcf.write("\n# Layer 1: Epoxy SiO2 underfill with C4 copper pillar\n1\nY\nN\n2.32E+06\n0.625\n0.00007\n" + str(case_dir / f"{case}L1_C4Layer.flp") + "\n")
            lcf.write("\n# Layer 2: silicon interposer\n2\nY\nN\n1.75E+06\n0.01\n0.00011\n" + str(case_dir / f"{case}L2_Interposer.flp") + "\n")
            lcf.write("\n# Layer 3: Underfill with ubump\n3\nY\nN\n2.32E+06\n0.625\n1.00E-05\n" + str(_l3) + "\n")
            lcf.write("\n# Layer 4: Chip layer\n4\nY\nY\n1.75E+06\n0.01\n0.00015\n" + str(l4_filled) + "\n")
            lcf.write("\n# Layer 5: TIM\n5\nY\nN\n4.00E+06\n0.25\n2.00E-05\n" + str(case_dir / f"{case}L5_TIM.flp") + "\n")

        rh._derive_hotspot_config(rh.HOTSPOT_TEMPLATE_CONFIG, derived_cfg, intp_size_mm, grid=grid)

    grid_steady = case_dir / f"{case}.grid.steady"
    if not grid_steady.exists():
        ptrace = case_dir / f"{case}.ptrace"
        powers_by_name = {f"Chiplet_{k}": ch.power_w for k, ch in enumerate(chiplets)}
        rh._write_ptrace_from_flp(l4_filled, ptrace, powers_by_name)

        steady = case_dir / f"{case}.steady"
        rc, _stdout, stderr = rh._run_hotspot(rh.HOTSPOT_BIN, derived_cfg, l4_filled, ptrace,
                                              steady, grid_steady, layers_lcf, "grid")
        if rc != 0:
            raise RuntimeError(f"hotspot rc={rc}\nSTDERR:\n{stderr[:600]}")

    temp_C = gtd._read_grid_steady(grid_steady, grid) - KELVIN
    return {
        "avg_temp_C": round(float(temp_C.mean()), 6),
        "max_temp_C": round(float(temp_C.max()), 6),
        "thermal_time_s": round(time.time() - t0, 3),
    }


# --------------------------------------------------------------------------- #
# 单布局 worker
# --------------------------------------------------------------------------- #
_G_LAYOUT_DIR = _G_EVAL_DIR = None
_G_GRID = GRID_DEFAULT
_G_STAGE = "both"


def _init_worker(layout_dir: str, eval_dir: str, grid: int, stage: str) -> None:
    global _G_LAYOUT_DIR, _G_EVAL_DIR, _G_GRID, _G_STAGE
    _G_LAYOUT_DIR, _G_EVAL_DIR, _G_GRID, _G_STAGE = layout_dir, eval_dir, grid, stage


def eval_one(stem: str) -> tuple[str, dict, str | None]:
    """评估单个布局, 返回 (stem, metrics, error)。"""
    layout_path = Path(_G_LAYOUT_DIR) / f"{stem}.json"
    case, seed = _split_stem(stem)
    out = {"case": case, "seed": seed}
    errs: list[str] = []
    try:
        record = json.loads(layout_path.read_text(encoding="utf-8"))
        out["n_chiplets"] = len(record["chiplets"])
        out["n_connections"] = len(record.get("connections", []))

        # 1) 外接框 (含 hubump)
        l, b, r, t = _bbox_mm(record)
        out["bbox_width_mm"] = round(r - l, 6)
        out["bbox_height_mm"] = round(t - b, 6)
        out["bbox_area_mm2"] = round((r - l) * (t - b), 6)

        # 2) 线长
        if _G_STAGE in ("wl", "both"):
            try:
                out.update(eval_wirelength(record))
            except Exception as e:  # noqa: BLE001
                errs.append(f"wirelength: {type(e).__name__}: {e}")

        # 3) 热仿真
        if _G_STAGE in ("thermal", "both"):
            try:
                case_dir = Path(_G_EVAL_DIR) / stem
                out.update(eval_thermal(record, case_dir, _G_GRID))
            except Exception as e:  # noqa: BLE001
                errs.append(f"thermal: {type(e).__name__}: {e}")
    except Exception as e:  # noqa: BLE001
        errs.append(f"{type(e).__name__}: {e}")
        out.setdefault("traceback", traceback.format_exc(limit=3))

    return stem, out, ("; ".join(errs) if errs else None)


# --------------------------------------------------------------------------- #
# cache / csv
# --------------------------------------------------------------------------- #
def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_csv(path: Path, cache: dict) -> None:
    def sort_key(stem: str):
        case, seed = _split_stem(stem)
        try:
            return (case, int(seed))
        except ValueError:
            return (case, 0)

    lines = [",".join(CSV_COLUMNS)]
    for stem in sorted(cache, key=sort_key):
        row = cache[stem]
        cells = []
        for col in CSV_COLUMNS:
            v = row.get(col, "")
            if isinstance(v, float):
                v = f"{v:.6f}"
            cells.append(str(v))
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(v, unit="", nd=4) -> str:
    if v is None or isinstance(v, str):
        return "n/a"
    return f"{v:.{nd}f}{unit}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_method(method: str, args) -> None:
    mdir = METHOD_DIR[method]
    layout_dir = mdir / LAYOUT_SUBDIR
    eval_dir = mdir / EVAL_SUBDIR
    cache_path = mdir / CACHE_NAME
    csv_path = mdir / CSV_NAME

    stems = sorted(p.stem for p in layout_dir.glob("*.json"))
    if not stems:
        print(f"[{method}] {layout_dir} 下没有布局 json, 跳过")
        return

    cache = _load_cache(cache_path)
    if args.only:
        want = set(args.only)
        stems = [s for s in stems if s in want]
        if not stems:
            print(f"[{method}] --only 没匹配到任何布局")
            return

    # 哪些还需要算: 按 stage 判断该 stage 的字段是否已齐全
    def need(stem: str) -> bool:
        row = cache.get(stem)
        if row is None:
            return True
        if row.get("error"):
            return True
        if args.stage in ("wl", "both") and "total_wirelength_mm" not in row:
            return True
        if args.stage in ("thermal", "both") and "max_temp_C" not in row:
            return True
        return False

    todo = [s for s in stems if need(s)]
    print(f"[{method}] 布局 {len(stems)} 个, 本次需算 {len(todo)} 个 (stage={args.stage}, grid={args.grid}, workers={args.workers})",
          flush=True)

    eval_dir.mkdir(parents=True, exist_ok=True)
    if todo:
        t0 = time.time()
        done = 0
        with Pool(args.workers, initializer=_init_worker,
                  initargs=(str(layout_dir), str(eval_dir), args.grid, args.stage)) as pool:
            for stem, row, err in pool.imap_unordered(eval_one, todo):
                done += 1
                if err:
                    row["error"] = err
                    print(f"[{method}] [{done}/{len(todo)}] {stem}: ERROR {err}", flush=True)
                else:
                    row.pop("error", None)
                    print(f"[{method}] [{done}/{len(todo)}] {stem}: "
                          f"wl_total={_fmt(row.get('total_wirelength_mm'))} "
                          f"wl_avg={_fmt(row.get('avg_wirelength_mm'))} "
                          f"Tavg={_fmt(row.get('avg_temp_C'), 'C', 3)} "
                          f"Tmax={_fmt(row.get('max_temp_C'), 'C', 3)} "
                          f"bbox={_fmt(row.get('bbox_area_mm2'))}mm2", flush=True)
                prev = cache.get(stem, {})
                cache[stem] = {**prev, **row}
                if done % 5 == 0:
                    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
                    _write_csv(csv_path, cache)
        print(f"[{method}] 用时 {time.time() - t0:.1f}s", flush=True)

    cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_csv(csv_path, cache)
    n_ok = sum(1 for r in cache.values() if not r.get("error"))
    print(f"[{method}] -> {csv_path}  ({len(cache)} 行, 无错误 {n_ok}, 有错误 {len(cache) - n_ok})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="both", choices=["AT", "RL", "both"])
    ap.add_argument("--stage", default="both", choices=["wl", "thermal", "both"],
                    help="wl=只算线长+外接框 (快), thermal=只算热, both=全算")
    ap.add_argument("--grid", type=int, default=GRID_DEFAULT, help="HotSpot 温度网格 (默认 64, 与 thermal_dataset_64 一致)")
    ap.add_argument("--workers", type=int, default=8, help="进程数 (热仿真单线程, 但线长的 ILP 很吃内存)")
    ap.add_argument("--only", nargs="*", default=None, help="只算这些 stem, 如 Case6_seed1")
    args = ap.parse_args()

    methods = ["AT", "RL"] if args.method == "both" else [args.method]
    for m in methods:
        run_method(m, args)


if __name__ == "__main__":
    main()
