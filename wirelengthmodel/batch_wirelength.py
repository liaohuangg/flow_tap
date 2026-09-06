#!/usr/bin/env python3
"""高效批量生成线长标签: 每个 chiplet_dataset_{k}.json 只加载一次, 复用 exact_wirelength.wirelength()。

与原 exact_wirelength.py --start/--end 的区别:
  - 原版对每个 system 都重新 json.load 整个 5000-system 文件 (20000 个要反复读 2 万次, ~80 分钟);
  - 本版每个文件只 load 一次, 逐 system 调用同一个已验证的 wirelength() (LP, 误差 0.000000%)。

输出与 wirelength_dataset 一致:
  total_wirelength/system_total_wirelength_{sid}.csv
  avg_wirelength/system_avg_wirelength_{sid}.csv

用法: python batch_wirelength.py --start 380001 --end 400000 --workers 4
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exact_wirelength import wirelength, PLACE_DIR, OUT_DIR  # noqa: E402

CHUNK = 5000


def process_file(k_start_end):
    k, start, end = k_start_end
    fp = f"{PLACE_DIR}/chiplet_dataset_{k}.json"
    with open(fp, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for sid, rec in data.items():
        i = int(sid.split("_")[1])
        if i < start or i > end:
            continue
        try:
            total, avg = wirelength(rec)
            out.append((i, total, avg, None))
        except Exception as e:  # noqa: BLE001
            out.append((i, None, None, f"{type(e).__name__}: {e}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    total_dir = os.path.join(OUT_DIR, "total_wirelength")
    avg_dir = os.path.join(OUT_DIR, "avg_wirelength")
    os.makedirs(total_dir, exist_ok=True)
    os.makedirs(avg_dir, exist_ok=True)

    ks = list(range((args.start - 1) // CHUNK + 1, (args.end - 1) // CHUNK + 2))
    tasks = [(k, args.start, args.end) for k in ks]

    t0 = time.time()
    n_ok = n_err = 0
    with Pool(args.workers) as pool:
        for results in pool.imap_unordered(process_file, tasks):
            for sid, total, avg, err in results:
                if err:
                    n_err += 1
                    print(f"system_{sid}: ERROR {err}", flush=True)
                    continue
                with open(f"{total_dir}/system_total_wirelength_{sid}.csv", "w") as f:
                    f.write(f"{total:.6f}\n")
                with open(f"{avg_dir}/system_avg_wirelength_{sid}.csv", "w") as f:
                    f.write(f"{avg:.6f}\n")
                n_ok += 1
            print(f"  完成一个文件, 累计 {n_ok} 个 ({time.time() - t0:.1f}s)", flush=True)

    print(f"DONE: {n_ok} 成功, {n_err} 失败, 耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
