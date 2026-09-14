#!/usr/bin/env python3
"""convert_AT_layout_to_case.py — 把 AT 的结果布局转成 placement_dataset 的 chiplet 格式。

目标格式 (Dataset/dataset/README.md, `placement_dataset/chiplet_dataset_*.json` 里的单条记录):

    {
      "system_id": "Case6_seed1",
      "chiplets": [
        { "name": "CPU2_0", "x-position": 0.045, "y-position": 29.045,
          "width": 25.91, "height": 24.91, "rotation": 0,
          "power": 65.0, "hubump": 0.045 },
        ...
      ],
      "connections": [ { "node1": "A", "node2": "B", "wireCount": 512 }, ... ]
    }

输入:  AT_result/result/<prefix>_bump/seed<N>/layout.json         (统一参数版)
       AT_result/result/<prefix>_bump_op/seed<N>/layout.json      (各 case 原始参数版)
输出:  AT_result/format_result/<prefix>_seed<N>.json
       AT_result/format_result/<prefix>_op_seed<N>.json

四处关键映射 (全部逐 chiplet 与 benchmark 交叉校验, 不一致直接抛错):

  1. 坐标口径 —— AT 的 x/y 是 **footprint 的中心** (见 reproduce.py: 夹取用
     `xmin + w/2`), README 要的是 **本体左下角**:
         fp_ll   = (x - placed_fp_w/2, y - placed_fp_h/2)      # footprint 左下角
         x-position = fp_ll.x + hubump                          # 本体左下角
     单位 AT 是 um, README 是 mm, 差 1000 倍。

  2. 旋转 —— AT 的 width/height 是 **未旋转的基准 footprint** (Case6 的 HBM 一律
     8090x12090, 靠 angle_rad 表达朝向; 90/270 度时实际占位要交换 W/H, 见
     reproduce.py::effective_size)。README 的 width/height 是 **摆放之后的尺寸**,
     rotation 只是 0/1 的朝向标记、不能再拿来交换。所以:
         placed_fp = effective_size(width, height, angle_rad)
         width     = placed_fp_w/1000 - 2*hubump
         rotation  = 0 (angle≈0/180度) 或 1 (angle≈90/270度)

  3. hubump / 连接数 —— AT 的 layout.json 里没有这两样。hubump 按 chiplet **名字**
     从 benchmark/cases_hubump/<prefix>.json 取 (两边名字完全一致); connections 整段
     从前缀对应的 benchmark 取 (前缀一一对应: Case6_bump -> Case6)。

  4. connections 只保留 README 的 3 个字段 (node1/node2/wireCount), benchmark 里
     额外的 EMIBType/EMIB_length/EMIB_max_width/EMIB_bump_width 按 README 约定丢掉。

chiplet 顺序按 benchmark 排, 便于与 RL 的输出直接对比。

用法:
  python convert_AT_layout_to_case.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

AT_ROOT = "/root/placement/flow_tap/resultEval/AT_result"
RESULT_DIR = os.path.join(AT_ROOT, "result")
OUT_DIR = os.path.join(AT_ROOT, "format_result")
BENCH_DIR = "/root/placement/flow_tap/benchmark/cases_hubump"

# 结果目录后缀 -> (benchmark 前缀, 输出 stem 的标记)。长的后缀要排在前面。
#   <case>_bump      AT 统一参数版         -> <case>_seed<N>.json
#   <case>_bump_op   AT 各 case 原始参数版 -> <case>_op_seed<N>.json
# 标记必须留在输出 stem 里, 否则同一 case 的两种版本会写成同一个 <case>_seed<N>.json。
BUMP_VARIANTS = (
    ("_bump_op", "_op"),
    ("_bump", ""),
)
UM_PER_MM = 1000.0
TOL = 1e-6


def split_result_dir(name: str) -> tuple[str, str] | None:
    """结果目录名 -> (benchmark 前缀, 输出 stem 标记)。不是 *_bump* 目录则返回 None。"""
    for suffix, tag in BUMP_VARIANTS:
        if name.endswith(suffix):
            return name[: -len(suffix)], tag
    return None


def placed_footprint_um(w_um: float, h_um: float, angle_rad: float) -> tuple[float, float]:
    """把未旋转的基准 footprint 按 angle_rad 折算成轴对齐的实际占位 (um)。

    与 reproduce.py::effective_size 同一口径: 90/270 度交换 W/H。
    """
    return (h_um, w_um) if abs(math.sin(angle_rad)) > 0.5 else (w_um, h_um)


def rotation_flag(angle_rad: float) -> int:
    """angle_rad -> README 的 rotation (只允许 0/1)。0/180 度记 0, 90/270 度记 1。"""
    return 1 if abs(math.sin(angle_rad)) > 0.5 else 0


def convert_one(layout_path: str, benchmark_path: str, out_path: str) -> dict:
    src = json.load(open(layout_path, encoding="utf-8"))
    bench = json.load(open(benchmark_path, encoding="utf-8"))

    if src.get("unit") != "um":
        raise ValueError(f"{layout_path}: 预期 unit='um', 实际 {src.get('unit')!r}")

    bench_by_name = {c["name"]: c for c in bench["chiplets"]}

    chiplets = []
    for c in src["chiplets"]:
        name = c["name"]
        b = bench_by_name.get(name)
        if b is None:
            raise KeyError(f"{layout_path}: chiplet {name!r} 在 benchmark 中不存在")

        angle = float(c.get("angle_rad", 0.0))
        base_w_um, base_h_um = float(c["width"]), float(c["height"])
        fp_w_um, fp_h_um = placed_footprint_um(base_w_um, base_h_um, angle)
        fp_w, fp_h = fp_w_um / UM_PER_MM, fp_h_um / UM_PER_MM   # mm

        # AT 的基准 footprint 与 benchmark 的 footprint 必须同为一对 (旋转只是交换顺序)
        if abs(sorted([fp_w, fp_h])[0] - sorted([b["footprint_w"], b["footprint_h"]])[0]) > TOL or \
           abs(sorted([fp_w, fp_h])[1] - sorted([b["footprint_w"], b["footprint_h"]])[1]) > TOL:
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 footprint {sorted([fp_w, fp_h])} "
                             f"与 benchmark {sorted([b['footprint_w'], b['footprint_h']])} 不符")

        hubump = b["hubump"]
        width = fp_w - 2.0 * hubump
        height = fp_h - 2.0 * hubump
        if width <= 0 or height <= 0:
            raise ValueError(f"{layout_path}: chiplet {name!r} 去环后尺寸非正 ({width}, {height})")

        # 中心 -> footprint 左下角 -> 本体左下角
        x_pos = c["x"] / UM_PER_MM - fp_w / 2.0 + hubump
        y_pos = c["y"] / UM_PER_MM - fp_h / 2.0 + hubump

        if abs(float(c["power_w"]) - b["power"]) > TOL:
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 power {c['power_w']} != {b['power']}")

        chiplets.append({
            "name": name,
            "x-position": round(x_pos, 6),
            "y-position": round(y_pos, 6),
            "width": round(width, 6),
            "height": round(height, 6),
            "rotation": rotation_flag(angle),
            "power": b["power"],
            "hubump": hubump,
        })

    # chiplet 顺序对齐 benchmark (AT 的书写顺序不同, 便于与 RL 的输出/benchmark 直接对比)
    order = {c["name"]: k for k, c in enumerate(bench["chiplets"])}
    if set(order) != {c["name"] for c in chiplets}:
        raise ValueError(f"{layout_path}: chiplet 名字集合与 benchmark 不一致")
    chiplets.sort(key=lambda c: order[c["name"]])

    stem = os.path.splitext(os.path.basename(out_path))[0]
    out = {
        "system_id": stem,                     # 文件名去掉 .json, 与 README 的 "system_id" 同义
        "chiplets": chiplets,
        "connections": [                       # 只保留 README 的 3 个字段
            {"node1": e["node1"], "node2": e["node2"], "wireCount": e["wireCount"]}
            for e in bench["connections"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_skip = 0
    for result_dir in sorted(glob.glob(os.path.join(RESULT_DIR, "*"))):
        if not os.path.isdir(result_dir):
            continue
        split = split_result_dir(os.path.basename(result_dir))
        if split is None:
            continue
        prefix, tag = split
        benchmark_path = os.path.join(BENCH_DIR, f"{prefix}.json")
        if not os.path.exists(benchmark_path):
            print(f"[skip] {prefix}: 无对应 benchmark {benchmark_path}", file=sys.stderr)
            n_skip += 1
            continue
        seed_dirs = sorted(glob.glob(os.path.join(result_dir, "seed*")))
        if not seed_dirs:
            print(f"[skip] {prefix}: 无 seed* 子目录", file=sys.stderr)
            n_skip += 1
            continue
        for seed_dir in seed_dirs:
            layout_path = os.path.join(seed_dir, "layout.json")
            if not os.path.exists(layout_path):
                print(f"[skip] {layout_path}: 不存在", file=sys.stderr)
                n_skip += 1
                continue
            seed_tag = os.path.basename(seed_dir)
            out_path = os.path.join(OUT_DIR, f"{prefix}{tag}_{seed_tag}.json")
            try:
                out = convert_one(layout_path, benchmark_path, out_path)
            except Exception as e:  # noqa: BLE001
                print(f"[err ] {prefix}/{seed_tag}: {type(e).__name__}: {e}", file=sys.stderr)
                n_skip += 1
                continue
            print(f"[ok] {os.path.basename(layout_path):<12} -> {os.path.basename(out_path):<22} "
                  f"chiplets={len(out['chiplets'])} connections={len(out['connections'])}")
            n_ok += 1
    print(f"\n完成: 生成 {n_ok} 个文件 -> {OUT_DIR}  (跳过/失败 {n_skip})")


if __name__ == "__main__":
    main()
