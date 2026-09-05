#!/usr/bin/env python3
"""重跑被替换 system 的线长仿真 (340001..360000, 34w-36w, avg 变体)。

前置: apply_replacement.py 已完成 (body 数据已换成新布局)。
用法: python3 re_run_wirelength_replaced.py --workers 28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
sys.path.insert(0, str(PROJECT / "gen_dataset"))

import gen_wirelength_dataset as gd  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--manifest", type=str, default=None,
                    help="清单文件路径(默认 /tmp/replacement/replaced_wirelength.txt)")
    args = ap.parse_args()

    manifest = Path(args.manifest) if args.manifest else (TMP / "replaced_wirelength.txt")
    ids = read_ids(manifest)
    print(f"[re-run wirelength] 待重跑 {len(ids)} 个 (清单 {manifest.name})", flush=True)
    if not ids:
        return

    records = load_records(ids)
    items = [(i, records[i], "avg") for i in ids]
    t0 = time.time()
    done = errs = 0
    with Pool(args.workers) as pool:
        for i, res in pool.imap_unordered(gd.process_layout, items):
            done += 1
            err = res.get("error")
            if err:
                errs += 1
                print(f"[wirelength] [{done}/{len(ids)}] system_{i}: ERROR {err}", flush=True)
            else:
                print(f"[wirelength] [{done}/{len(ids)}] system_{i}: total={res['total_wirelength']} "
                      f"avg={res['avg_wirelength']:.4f} t={res['avg_solve_time']:.2f}s", flush=True)
    print(f"[re-run wirelength] DONE {done} 个, 失败 {errs}, 墙钟 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
