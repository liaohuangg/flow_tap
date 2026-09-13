#!/usr/bin/env python3
"""convert_FM_layout_to_case.py — 把 FM (ChipletFM flow matching) 的结果布局转成 placement_dataset 的 chiplet 格式。

目标格式 (Dataset/dataset/README.md, `placement_dataset/chiplet_dataset_*.json` 里的单条记录):

    {
      "system_id": "Case6_bestT_seed67",
      "chiplets": [
        { "name": "CPU2_0", "x-position": 0.045, "y-position": 29.045,
          "width": 25.91, "height": 24.91, "rotation": 0,
          "power": 65.0, "hubump": 0.045 },
        ...
      ],
      "connections": [ { "node1": "A", "node2": "B", "wireCount": 512 }, ... ]
    }

输入:  FM_result/fm-result/<prefix>/<prefix>_best{T,WL}_seed<N>.json
输出:  FM_result/format_result/<prefix>_best{T,WL}_seed<N>.json

五处关键映射 (全部逐 chiplet 与 benchmark 交叉校验, 不一致直接抛错):

  1. 坐标口径 —— **与 README 同口径, 坐标原样搬运, 不做任何换算**。
     依据: 这份 json 的消费方 gen_dataset/gen_wirelength_dataset.py::TapSystem
     明确按 "x-position/y-position = 本体左下角, width/height = 本体(die) 尺寸" 解释
     (第 133-137 行: self.width = c["width"]  # body(die) 宽;
      self.x = c["x-position"] + width/2, 注释 "中心 = body 左下角 + body/2")。
     FM 报数用的评分调用就是这个类, 所以两边同口径 —— 不像 AT 那边需要做
     "footprint 中心 -> 本体左下角" 的换算 (见 convert_AT_layout_to_case.py)。

  2. rotation —— FM 全部输出 0 (532/532 条), 原样保留, 只校验 ∈ {0,1}。
     README 第五节第 2 条: 本目录的 width/height 已经是摆放后的尺寸, 不能再按 rotation 交换。

  3. hubump —— FM 的 json **没有这个字段**, 按 chiplet 名字从
     benchmark/cases_hubump/<prefix>.json 取 (两边名字完全一致)。
     `syn6` 不在该目录 (benchmark 只有 12 个主 case), 回退到
     baseline/ATPlace_pub/cases_hubump/syn6.json —— 与 baseline/RL/examples/syn6.json
     逐字段相同, 同为 gen_dataset/wrap_hubump_cases.py 的产物。
     取到的 hubump 必须满足 width + 2*hubump == benchmark 的 footprint_*, 否则抛错。

  4. connections —— FM 的 connections 已经与 benchmark 逐条一致 (已校验),
     但只保留 README 的 3 个字段 (node1/node2/wireCount), benchmark 里额外的
     EMIBType/EMIB_length/EMIB_max_width/EMIB_bump_width 按 README 约定丢掉。
     整段 (含顺序) 取自 benchmark。

  5. 文件名保留 bestT/bestWL 标记 —— FM 每个 case 出两个 pick (README: 两个目标
     互相不一致, 通常选出不同布局)。去掉标记的话 cpu-dram 的 bestT/bestWL 会同名
     (两者 seed 都是 62, 内容也确实逐字节相同), 且没法再从 format_result 里区分
     这个布局是为温度还是为线长选的, 所以保留。
     前缀 -> benchmark 的映射: <prefix>_bestT / <prefix>_bestWL -> <prefix>。

FM 原 json 里的 wirelength / area / aspect_ratio 是它自己算的一遍数 (与 resultEval 的
口径不同), 不搬进目标格式; 这三个数在 fm-result/README.md 的表里有。

chiplet 顺序按 benchmark 排, 便于与 RL / AT 的输出直接对比。

用法:
  python convert_FM_layout_to_case.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

FM_ROOT = "/root/placement/flow_tap/resultEval/FM_result"
SRC_DIR = os.path.join(FM_ROOT, "fm-result")
OUT_DIR = os.path.join(FM_ROOT, "format_result")

BENCH_DIR = "/root/placement/flow_tap/benchmark/cases_hubump"
# benchmark/cases_hubump 只有 12 个主 case, syn6 从这里回退
BENCH_FALLBACK_DIR = "/root/placement/flow_tap/baseline/ATPlace_pub/cases_hubump"

TOL = 1e-6
PICKS = ("bestT", "bestWL")


def _resolve_benchmark(prefix: str) -> tuple[str, bool]:
    """返回 (benchmark 路径, 是否用了回退)。"""
    primary = os.path.join(BENCH_DIR, f"{prefix}.json")
    if os.path.exists(primary):
        return primary, False
    fallback = os.path.join(BENCH_FALLBACK_DIR, f"{prefix}.json")
    if os.path.exists(fallback):
        return fallback, True
    raise FileNotFoundError(f"无对应 benchmark: {primary} (回退 {fallback} 也没有)")


def convert_one(layout_path: str, benchmark_path: str, out_path: str) -> dict:
    src = json.load(open(layout_path, encoding="utf-8"))
    bench = json.load(open(benchmark_path, encoding="utf-8"))
    bench_by_name = {c["name"]: c for c in bench["chiplets"]}

    src_names = {c["name"] for c in src["chiplets"]}
    if src_names != set(bench_by_name):
        raise ValueError(f"{layout_path}: chiplet 名字集合与 {benchmark_path} 不一致 "
                         f"(多 {sorted(src_names - set(bench_by_name))}, "
                         f"少 {sorted(set(bench_by_name) - src_names)})")

    chiplets = []
    for c in src["chiplets"]:
        name = c["name"]
        b = bench_by_name[name]

        # FM 的 json 没有 hubump, 从 benchmark 按名字取
        hubump = float(b["hubump"])
        width, height = float(c["width"]), float(c["height"])
        x_pos, y_pos = float(c["x-position"]), float(c["y-position"])
        rotation = int(c.get("rotation", 0))

        if rotation not in (0, 1):
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 rotation={rotation!r} 不是 0/1")
        if width <= 0 or height <= 0:
            raise ValueError(f"{layout_path}: chiplet {name!r} 尺寸非正 ({width}, {height})")

        # --- 与 benchmark 交叉校验 ---
        for label, got, want_v in (
            ("width", width, b["width"]),
            ("height", height, b["height"]),
            ("power", float(c["power"]), b["power"]),
        ):
            if abs(got - float(want_v)) > TOL:
                raise ValueError(f"{layout_path}: chiplet {name!r} 的 {label} 与 benchmark 不符: "
                                 f"{got} != {want_v}")

        # 加环后的 footprint 必须与 benchmark 声明的 footprint_* 同为一对 (旋转只是交换顺序)
        fp_w, fp_h = width + 2.0 * hubump, height + 2.0 * hubump
        got, want = sorted([fp_w, fp_h]), sorted([float(b["footprint_w"]), float(b["footprint_h"])])
        if abs(got[0] - want[0]) > TOL or abs(got[1] - want[1]) > TOL:
            raise ValueError(f"{layout_path}: chiplet {name!r} 的 footprint {got} "
                             f"与 benchmark {want} 不符")

        chiplets.append({
            "name": name,
            "x-position": round(x_pos, 6),
            "y-position": round(y_pos, 6),
            "width": round(width, 6),
            "height": round(height, 6),
            "rotation": rotation,
            "power": b["power"],
            "hubump": hubump,
        })

    # --- connections: FM 与 benchmark 必须逐条一致, 输出取 benchmark 的 3 字段版本 ---
    def key(e: dict) -> tuple:
        return (e["node1"], e["node2"], int(e["wireCount"]))

    src_conn = {key(e) for e in src.get("connections", [])}
    bench_conn = {key(e) for e in bench.get("connections", [])}
    if src_conn != bench_conn:
        raise ValueError(f"{layout_path}: connections 与 benchmark 不一致 "
                         f"(多 {sorted(src_conn - bench_conn)[:3]}, "
                         f"少 {sorted(bench_conn - src_conn)[:3]})")

    # chiplet 顺序对齐 benchmark (FM 的书写顺序不同, 便于与 RL / AT 的输出直接对比)
    order = {c["name"]: k for k, c in enumerate(bench["chiplets"])}
    chiplets.sort(key=lambda c: order[c["name"]])

    stem = os.path.splitext(os.path.basename(out_path))[0]
    out = {
        "system_id": stem,                     # 文件名去掉 .json, 与 README 的 "system_id" 同义
        "chiplets": chiplets,
        "connections": [                       # 只留 README 的 3 个字段, EMIB* 丢掉
            {"node1": e["node1"], "node2": e["node2"], "wireCount": int(e["wireCount"])}
            for e in bench["connections"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = n_err = 0
    layout_paths = sorted(glob.glob(os.path.join(SRC_DIR, "*", "*.json")))
    for layout_path in layout_paths:
        stem = os.path.splitext(os.path.basename(layout_path))[0]   # 形如 Case6_bestT_seed67
        case_dir = os.path.basename(os.path.dirname(layout_path))   # 形如 Case6

        if case_dir not in stem:
            print(f"[skip] {stem}: 父目录名 {case_dir!r} 与文件名不符", file=sys.stderr)
            n_err += 1
            continue

        # <prefix>_best{T,WL}_seed<N> -> prefix
        if "_best" not in stem or "_seed" not in stem:
            print(f"[skip] {stem}: 不是 <prefix>_best{{T,WL}}_seed<N> 形式", file=sys.stderr)
            n_err += 1
            continue
        prefix, rest = stem.split("_best", 1)     # 'Case6' , 'T_seed67'
        pick, _seed = rest.split("_seed", 1)      # 'T'     , '67'
        if f"best{pick}" not in PICKS:
            print(f"[skip] {stem}: pick={pick!r} 不是 {PICKS}", file=sys.stderr)
            n_err += 1
            continue
        if prefix != case_dir:
            print(f"[skip] {stem}: 前缀 {prefix!r} 与父目录 {case_dir!r} 不符", file=sys.stderr)
            n_err += 1
            continue

        try:
            benchmark_path, used_fb = _resolve_benchmark(prefix)
        except FileNotFoundError as e:
            print(f"[skip] {stem}: {e}", file=sys.stderr)
            n_err += 1
            continue

        out_path = os.path.join(OUT_DIR, f"{stem}.json")
        try:
            out = convert_one(layout_path, benchmark_path, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"[err ] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            n_err += 1
            continue
        src = json.load(open(layout_path, encoding="utf-8"))
        fb = "  [benchmark 回退]" if used_fb else ""
        print(f"[ok] {stem:<26} -> {os.path.basename(out_path):<26} "
              f"chiplets={len(out['chiplets'])} connections={len(out['connections'])} "
              f"FM_wl={src.get('wirelength')}{fb}")
        n_ok += 1

    print(f"\n完成: 生成 {n_ok} 个文件 -> {OUT_DIR}  (跳过/失败 {n_err})")


if __name__ == "__main__":
    main()
