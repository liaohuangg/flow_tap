#!/usr/bin/env python3
"""
Pack chiplet placement + interconnect into per-system records.

For each system i in [start, end]:
  - chiplets:    from output/placement/system_i.json
                 keep ONLY: name, x-position, y-position, width, height, rotation, power
  - connections: from input_test/system_i.json
                 keep ONLY: node1, node2, wireCount

Records are grouped into output files by system index:
    file k  <=>  systems [(k-1)*chunk+1, k*chunk]

Writing is UPSERT: if the target file already exists, new records are merged
in — so this supports incremental appends (default).  To force a full fresh
repackage, delete the target files first.

All paths are resolved relative to this script's directory.
"""

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

# 脚本位于 gen_dataset/ 顶层,数据在 ../Dataset/dataset/ 下
BASE = Path(__file__).resolve().parent.parent / "Dataset" / "dataset"
INPUT_DIR = BASE / "input_test"
PLACE_DIR = BASE / "output" / "placement"
OUT_DIR = BASE / "placement_dataset"

CHIPLET_FIELDS = ["name", "x-position", "y-position", "width", "height", "rotation", "power"]
CONN_FIELDS = ["node1", "node2", "wireCount"]


def build_record(i: int) -> dict:
    """One merged system record: placement coords + input_test connections."""
    ipath = INPUT_DIR / f"system_{i}.json"
    ppath = PLACE_DIR / f"system_{i}.json"
    with open(ipath, "r", encoding="utf-8") as f:
        spec = json.load(f)
    with open(ppath, "r", encoding="utf-8") as f:
        place = json.load(f)

    chiplets = [{k: c[k] for k in CHIPLET_FIELDS} for c in place["chiplets"]]
    connections = [{k: c[k] for k in CONN_FIELDS} for c in spec.get("connections", [])]

    return {"system_id": f"system_{i}", "chiplets": chiplets, "connections": connections}


def write_chunk(args):
    """Upsert systems of one output file; returns (name, before, after, missing)."""
    k, indices = args
    out_path = OUT_DIR / f"chiplet_dataset_{k}.json"
    data = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    before = len(data)
    missing = []
    for i in indices:
        try:
            data[f"system_{i}"] = build_record(i)
        except Exception as e:  # noqa: BLE001
            missing.append((i, str(e)))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return out_path.name, before, len(data), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = args.end - args.start + 1

    groups = {}
    for i in range(args.start, args.end + 1):
        k = (i - 1) // args.chunk + 1
        groups.setdefault(k, []).append(i)

    print(f"[pack] {args.start}..{args.end} = {total} systems, "
          f"touch {len(groups)} file(s), {args.workers} workers -> {OUT_DIR.relative_to(BASE)}")

    t0 = time.time()
    with Pool(args.workers) as pool:
        results = pool.map(write_chunk, sorted(groups.items()))

    n_total = 0
    all_missing = []
    for name, before, after, missing in results:
        n_total += after - before
        all_missing.extend((i, e) for i, e in missing)
        tag = "OK" if not missing else f"{len(missing)} MISSING"
        print(f"[pack] {name}: +{after - before} (was {before}, now {after}) {tag}")

    print(f"[pack] DONE: +{n_total} systems in {time.time() - t0:.1f}s")
    if all_missing:
        print(f"[pack] WARNING: {len(all_missing)} failed:")
        for i, e in all_missing[:20]:
            print(f"  system_{i}: {e}")


if __name__ == "__main__":
    main()
