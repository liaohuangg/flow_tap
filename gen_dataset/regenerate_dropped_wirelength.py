#!/usr/bin/env python3
"""补生成被丢弃 system 的 wirelength 数据。

背景: 340001..360000 里 682 个 system 因 bump region 物理不可行 (小 die 塞了过高互连带宽,
45um microbump 环宽超过芯片本体) 在 preprocess_bump_region 阶段被丢弃, 故未进入
placement_dataset/placement_dataset_tw, 也没有 wirelength 数据。

方法: 这些 system 的原始 footprint 数据仍完整保留在 placement_dataset 里。线长求解器
(TapSystem, hubump_mode="die") 本身是"拿 body 尺寸向外扩环"计算 hubump, 对这类 case
把原始 footprint 当作 body 直接求解即可 (已验证 682 个全部可解, 最坏 hubump 仅 1.485mm)。

用法:
  python regenerate_dropped_wirelength.py --workers 8
"""
from __future__ import annotations

import argparse
import re
import time
from multiprocessing import Pool
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent))

from gen_wirelength_dataset import (
    AVG_DIR,
    TOTAL_DIR,
    TIME_DIR,
    TapSystem,
    _write_scalar_csv,
    solve_cplex_avg,
)

DATASET = Path("/root/placement/flow_tap/Dataset/dataset")
SRC_DIR = DATASET / "placement_dataset"          # 原始 footprint 数据(含被丢弃的)
CHUNK = 5000
START, END = 340001, 360000


def missing_systems() -> list[int]:
    present = {int(m.group(1)) for f in AVG_DIR.glob("*.csv")
               if (m := re.search(r"_(\d+)\.csv$", f.name))}
    return sorted(i for i in range(START, END + 1) if i not in present)


def load_record(i: int) -> dict:
    k = (i - 1) // CHUNK + 1
    import json
    data = json.loads((SRC_DIR / f"chiplet_dataset_{k}.json").read_text(encoding="utf-8"))
    return data[f"system_{i}"]


def worker(i: int) -> tuple[int, float, float | None, float, str | None]:
    try:
        rec = load_record(i)
        system = TapSystem(rec)                    # footprint 当 body, 向外扩环算 hubump
        t0 = time.time()
        avg_wl, total_wl = solve_cplex_avg(system)
        dt = time.time() - t0
        _write_scalar_csv(AVG_DIR / f"system_avg_wirelength_{i}.csv", avg_wl)
        if total_wl is not None:
            _write_scalar_csv(TOTAL_DIR / f"system_total_wirelength_{i}.csv", total_wl)
        _write_scalar_csv(TIME_DIR / f"system_wirelength_time_{i}.csv", dt)
        return i, avg_wl, total_wl, dt, None
    except Exception as e:  # noqa: BLE001
        return i, 0.0, None, 0.0, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    missing = missing_systems()
    print(f"[regen] 待补 {len(missing)} 个 system (340001..360000)", flush=True)
    if not missing:
        print("[regen] 无需补生成")
        return

    t0 = time.time()
    ok = 0
    errs = []
    with Pool(args.workers) as pool:
        for i, avg, total, dt, err in pool.imap_unordered(worker, missing):
            if err:
                errs.append((i, err))
                print(f"[regen] system_{i}: ERROR {err}", flush=True)
            else:
                ok += 1
                print(f"[regen] system_{i}: total={total if total is not None else 'NA'}mm "
                      f"avg={avg:.4f}mm t={dt:.2f}s", flush=True)

    print(f"[regen] DONE: {ok}/{len(missing)} 成功, 失败 {len(errs)}, "
          f"墙钟 {time.time() - t0:.1f}s", flush=True)
    if errs:
        for i, err in errs:
            print(f"  system_{i}: {err}")


if __name__ == "__main__":
    main()
