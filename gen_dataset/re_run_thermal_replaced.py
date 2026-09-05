#!/usr/bin/env python3
"""重跑被替换 system 的热仿真 (300001..320000, 30w-32w)。

前置: apply_replacement.py 已完成 (body 数据已换成新布局)。
用法: python3 re_run_thermal_replaced.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
sys.path.insert(0, str(PROJECT / "gen_dataset"))

import gen_thermal_dataset as gd  # noqa: E402

TMP = Path("/tmp/replacement")
BODY = gd.PLACE_DATASET  # placement_dataset_tw
CHUNK = gd.CHUNK


def read_ids(path: Path) -> list[int]:
    return sorted(int(l.split()[0]) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())


def load_records(ids: list[int]) -> dict[int, dict]:
    """按 chunk 分组, 每个 JSON 文件只读一次, 返回 {system_id: record}。"""
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in ids:
        groups[(i - 1) // CHUNK + 1].append(i)
    records: dict[int, dict] = {}
    for k, sub in groups.items():
        data = json.loads((BODY / f"chiplet_dataset_{k}.json").read_text(encoding="utf-8"))
        for i in sub:
            records[i] = data[f"system_{i}"]
    return records


def delete_old_outputs(i: int) -> None:
    """删掉被替换 system 的旧热仿真输出, 强制重跑 (gen_thermal_dataset 有 skip 逻辑)。"""
    for d, pat in [
        (gd.TEMP_DIR, f"system_temp_{i}_0.csv"),
        (gd.POWER_DIR, f"system_power_{i}_0.csv"),
        (gd.MAXT_DIR, f"system_maxtemp_{i}_0.csv"),
        (gd.AVGT_DIR, f"system_avgtemp_{i}_0.csv"),
        (gd.TOTAL_DIR, f"system_totalpower_{i}_0.csv"),
    ]:
        p = d / pat
        if p.exists():
            p.unlink()
    cfg = gd.CONFIG_DIR / f"system_{i}_config"
    if cfg.exists():
        shutil.rmtree(cfg, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    ids = read_ids(TMP / "replaced_thermal.txt")
    print(f"[re-run thermal] 待重跑 {len(ids)} 个 (300001..320000)", flush=True)
    if not ids:
        return

    for i in ids:
        delete_old_outputs(i)

    records = load_records(ids)
    items = [(i, records[i]) for i in ids]
    t0 = time.time()
    done = errs = 0
    with Pool(args.workers) as pool:
        for i, msg in pool.imap_unordered(gd.process_layout, items):
            done += 1
            if msg.startswith("ERROR"):
                errs += 1
                print(f"[thermal] [{done}/{len(ids)}] system_{i}: {msg}", flush=True)
            else:
                print(f"[thermal] [{done}/{len(ids)}] system_{i}: {msg}", flush=True)
    print(f"[re-run thermal] DONE {done} 个, 失败 {errs}, 墙钟 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
