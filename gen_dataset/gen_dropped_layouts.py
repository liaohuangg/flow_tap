#!/usr/bin/env python3
"""
为 placement_dataset_tw 中「缺失」的 system(footprint 有、但 body 因 bump region 不可行
被预处理丢弃)重新随机生成可行布局(body 长宽都 > 3mm), 用于补全 body 数据集。

与 gen_replacement_layouts.py 的区别:
  - read_missing_ids(): 找「footprint 有、body 缺失」的 system(而不是 body<3)。
  - BASE_SEED = 700001(与 49,059 替换用的 600001 错开, 避免 seed 冲突)。
  - TMP = /tmp/replacement_dropped。

生成/校验逻辑复用 gen_replacement_layouts 的 gen_system / is_feasible。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DS = PROJECT / "Dataset"
DATASET = DS / "dataset"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_replacement_layouts import gen_system, is_feasible  # noqa: E402

BASE_SEED = 700001
TMP = Path("/tmp/replacement_dropped")
IN_DIR = TMP / "input_test"
PLACE_DIR = TMP / "placement"
FOOT = DATASET / "placement_dataset" / "placement_dataset"
BODY = DATASET / "placement_dataset" / "placement_dataset_tw"


def read_missing_ids() -> list[int]:
    """返回「footprint 有、但 body 缺失」的 system 编号(升序)。"""
    foot: set[int] = set()
    body: set[int] = set()
    for fp in FOOT.glob("chiplet_dataset_*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid in data:
            m = re.fullmatch(r"system_(\d+)", sid)
            if m:
                foot.add(int(m.group(1)))
    for fp in BODY.glob("chiplet_dataset_*.json"):
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid in data:
            m = re.fullmatch(r"system_(\d+)", sid)
            if m:
                body.add(int(m.group(1)))
    return sorted(foot - body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=None, help="默认补全全部缺失; 可指定只补前 count 个")
    ap.add_argument("--workers", type=int, default=20, help="贪心布局并行 chunk 数")
    ap.add_argument("--only-check", action="store_true", help="只生成+校验, 不跑贪心布局")
    args = ap.parse_args()

    targets = read_missing_ids()
    count = len(targets) if args.count is None else min(args.count, len(targets))
    print(f"[dropped] 读到 {len(targets)} 个「footprint 有、body 缺失」的 system, 本次补前 {count} 个", flush=True)

    for _d in (IN_DIR, PLACE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    # 逐 m 生成候选(seed 连续), 直到收集到 count 个可行候选
    seed_to: dict[int, tuple[list[dict], list[dict]]] = {}
    feasible_seeds: list[int] = []
    m = 0
    while len(feasible_seeds) < count:
        seed = BASE_SEED + m
        chiplets, connections = gen_system(seed)
        seed_to[seed] = (chiplets, connections)
        if is_feasible(chiplets, connections):
            feasible_seeds.append(seed)
        m += 1
        if m > count * 3 + 1000:
            raise RuntimeError("候选重试过多, 无法生成足够多可行布局")
        if m % 1000 == 0:
            print(f"[dropped] 已生成候选 {m}, 可行 {len(feasible_seeds)}/{count}", flush=True)

    print(f"[dropped] 候选 {len(seed_to)} 个(含不可行 {len(seed_to) - len(feasible_seeds)} 个), "
          f"可行 {len(feasible_seeds)} 个", flush=True)

    # 写 input_test(全部候选, 连续 seed)保证贪心布局区间连续
    for seed, (chiplets, connections) in seed_to.items():
        (IN_DIR / f"system_{seed}.json").write_text(
            json.dumps({"chiplets": chiplets, "connections": connections}), encoding="utf-8")

    # mapping: target -> 可行 seed
    mapping: dict[str, int] = {}
    for k, target in enumerate(targets[:count]):
        mapping[str(target)] = feasible_seeds[k]

    (TMP / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[dropped] 生成完成: {len(mapping)} 个可行布局 -> {TMP / 'mapping.json'}", flush=True)

    # 记录补全范围内需要重仿真的清单(30w-32w 热 / 34w-36w 线长)
    thermal_ids = sorted(int(k) for k in mapping if 300001 <= int(k) <= 320000)
    wl_ids = sorted(int(k) for k in mapping if 340001 <= int(k) <= 360000)
    (TMP / "dropped_thermal.txt").write_text(
        "".join(f"{t}\t{mapping[str(t)]}\n" for t in thermal_ids), encoding="utf-8")
    (TMP / "dropped_wirelength.txt").write_text(
        "".join(f"{t}\t{mapping[str(t)]}\n" for t in wl_ids), encoding="utf-8")
    print(f"[dropped] 清单: 30w-32w {len(thermal_ids)} 个 -> dropped_thermal.txt; "
          f"34w-36w {len(wl_ids)} 个 -> dropped_wirelength.txt", flush=True)

    if args.only_check:
        return

    # 贪心布局(连续 seed 区间, 分 chunk 并行, 在 dataset/ 目录下运行)
    lo = BASE_SEED
    hi = BASE_SEED + len(seed_to) - 1
    workers = max(1, args.workers)
    print(f"[dropped] 开始贪心布局 seed {lo}..{hi} (共 {len(seed_to)} 个, {workers} 并行)...", flush=True)

    log_dir = TMP / "greedy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    per = (len(seed_to) + workers - 1) // workers
    procs: list[subprocess.Popen] = []
    chunk_lo = lo
    while chunk_lo <= hi:
        chunk_hi = min(chunk_lo + per - 1, hi)
        cmd = [sys.executable, str(PROJECT / "gen_dataset" / "gen_legal_pla_greedy.py"),
               "--input-dir", str(IN_DIR), "--output-dir", str(PLACE_DIR),
               "--start", str(chunk_lo), "--end", str(chunk_hi)]
        logf = open(log_dir / f"greedy_{chunk_lo}_{chunk_hi}.log", "w")
        procs.append(subprocess.Popen(cmd, cwd=str(DATASET), stdout=logf, stderr=subprocess.STDOUT))
        chunk_lo = chunk_hi + 1

    fails = 0
    for p in procs:
        if p.wait() != 0:
            fails += 1
            print("[dropped] 贪心布局失败一个 chunk", file=sys.stderr, flush=True)
    if fails:
        print(f"[dropped] 贪心布局失败 {fails} 个 chunk", file=sys.stderr)
        sys.exit(1)
    print("[dropped] 贪心布局完成", flush=True)


if __name__ == "__main__":
    main()
