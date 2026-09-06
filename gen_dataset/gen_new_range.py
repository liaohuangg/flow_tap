#!/usr/bin/env python3
"""为新 system 区间 [start, end] 生成 body(本体)长宽 > 3mm 的布局数据。

复刻原始数据集规则(芯片 3~20、footprint 3~30mm、长宽比 0.8~1.25、功耗 1~200、
连通生成树 + 25% 额外边、带宽 128/256/512/1024),只保留 body > 3mm 的可行布局
(复用 gen_replacement_layouts.gen_system + is_feasible),再跑贪心布局、打包 footprint + body。

输出(原始单位,不做归一化,与现有数据一致):
  dataset/placement_dataset/placement_dataset/chiplet_dataset_{k}.json    (footprint, 无 hubump)
  dataset/placement_dataset/placement_dataset_tw/chiplet_dataset_{k}.json (body, 含 hubump)

用法:
  python gen_new_range.py --start 380001 --end 400000 --workers 28
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DATASET = PROJECT / "Dataset" / "dataset"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_replacement_layouts import gen_system, is_feasible  # noqa: E402
from preprocess_bump_region import preprocess_record  # noqa: E402

FOOT_DIR = DATASET / "placement_dataset" / "placement_dataset"
BODY_DIR = DATASET / "placement_dataset" / "placement_dataset_tw"
TMP = Path("/tmp/new_range")
IN_DIR = TMP / "input_test"
PLACE_DIR = TMP / "placement"
CHUNK = 5000
CHIPLET_FIELDS = ["name", "x-position", "y-position", "width", "height", "rotation", "power"]
CONN_FIELDS = ["node1", "node2", "wireCount"]
RETRY_OFFSET = 1_000_000  # 重试 seed 偏移, 避免与已有数据 seed(1..380000 / 600001+ / 700001+)冲突


def gen_feasible(target: int) -> tuple[list[dict], list[dict]]:
    """为目标 id 确定性生成一个 body>3mm 的可行 footprint 系统。"""
    for r in range(100):
        seed = target + r * RETRY_OFFSET
        chiplets, connections = gen_system(seed)
        if is_feasible(chiplets, connections):
            return chiplets, connections
    raise RuntimeError(f"system_{target}: 100 次重试仍不可行")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()

    start, end = args.start, args.end
    for d in (IN_DIR, PLACE_DIR, TMP / "greedy_logs"):
        d.mkdir(parents=True, exist_ok=True)

    # 1) 生成可行 footprint, 写 input_test (connections 带 EMIB 字段供贪心读连通性)
    print(f"[gen] 生成 {start}..{end} 可行布局 ...", flush=True)
    for t in range(start, end + 1):
        chiplets, connections = gen_feasible(t)
        (IN_DIR / f"system_{t}.json").write_text(
            json.dumps({"chiplets": chiplets, "connections": connections}), encoding="utf-8")
        if t % 5000 == 0:
            print(f"[gen]   input_test 已写 {t}/{end}", flush=True)
    print(f"[gen] 生成 {end - start + 1} 个可行布局 (input_test 完成)", flush=True)

    # 2) 贪心布局 (并行 chunk, 与 gendataset.sh 一致, cwd=dataset/)
    workers = max(1, args.workers)
    per = (end - start + 1 + workers - 1) // workers
    procs = []
    chunk_lo = start
    while chunk_lo <= end:
        chunk_hi = min(chunk_lo + per - 1, end)
        cmd = [sys.executable, str(PROJECT / "gen_dataset" / "gen_legal_pla_greedy.py"),
               "--input-dir", str(IN_DIR), "--output-dir", str(PLACE_DIR),
               "--start", str(chunk_lo), "--end", str(chunk_hi)]
        logf = open(TMP / "greedy_logs" / f"greedy_{chunk_lo}_{chunk_hi}.log", "w")
        procs.append((chunk_lo, chunk_hi,
                      subprocess.Popen(cmd, cwd=str(DATASET), stdout=logf, stderr=subprocess.STDOUT)))
        chunk_lo = chunk_hi + 1

    fails = 0
    for _lo, _hi, p in procs:
        if p.wait() != 0:
            fails += 1
            print(f"[gen] 贪心布局失败 chunk {_lo}..{_hi}", file=sys.stderr, flush=True)
    if fails:
        print(f"[gen] 贪心布局失败 {fails} 个 chunk, 中止", file=sys.stderr)
        sys.exit(1)
    print("[gen] 贪心布局完成", flush=True)

    # 3) 打包 footprint + body (body 通过 preprocess_record 计算 hubump 并内缩)
    k0, k1 = (start - 1) // CHUNK + 1, (end - 1) // CHUNK + 1
    for k in range(k0, k1 + 1):
        foot_path = FOOT_DIR / f"chiplet_dataset_{k}.json"
        body_path = BODY_DIR / f"chiplet_dataset_{k}.json"
        foot_data = json.loads(foot_path.read_text()) if foot_path.exists() else {}
        body_data = json.loads(body_path.read_text()) if body_path.exists() else {}
        lo_k, hi_k = max(start, (k - 1) * CHUNK + 1), min(end, k * CHUNK)
        for t in range(lo_k, hi_k + 1):
            spec = json.loads((IN_DIR / f"system_{t}.json").read_text())
            place = json.loads((PLACE_DIR / f"system_{t}.json").read_text())
            chiplets = [{f: c[f] for f in CHIPLET_FIELDS} for c in place["chiplets"]]
            connections = [{f: c[f] for f in CONN_FIELDS} for c in spec["connections"]]
            foot_rec = {"system_id": f"system_{t}", "chiplets": chiplets, "connections": connections}
            body_rec = preprocess_record(foot_rec)
            foot_data[f"system_{t}"] = foot_rec
            body_data[f"system_{t}"] = body_rec
        foot_path.write_text(json.dumps(foot_data, ensure_ascii=False), encoding="utf-8")
        body_path.write_text(json.dumps(body_data, ensure_ascii=False), encoding="utf-8")
        print(f"[gen] 打包 {foot_path.name} + body (共 {len(foot_data)} systems)", flush=True)

    print(f"[gen] 全部完成 {start}..{end}", flush=True)


if __name__ == "__main__":
    main()
