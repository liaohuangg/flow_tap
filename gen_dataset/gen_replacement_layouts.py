#!/usr/bin/env python3
"""
为 body 数据(placement_dataset_tw)中所有「有 chiplet body 长/宽 < 3mm」的 system
重新随机生成可行布局(新布局 body 长宽都 > 3mm), 用于原位替换。

生成规则复刻原始数据集(gendataset.sh):
  - 芯片数 3~20, 尺寸 3~30 mm, 长宽比 0.8~1.25 (整数)
  - 功耗 1~200
  - 连接: 连通生成树 + 25% 额外边, 带宽 128/256/512/1024
  - 确定性 seed (seed=600001+m), 连续编号, 便于一次跑通贪心布局

关键点:
  - input_test 的 connections 带 EMIB 字段(EMIBType=interfaceC, 与 input_preprocess.matrix_to_connections
    一致), 这样 gen_legal_pla_greedy.py 的 load_emib_placement_json 才能读到连通性做聚类/力导向布局,
    否则会退化成"无连接打包"。
  - 候选 seed 连续: 逐 m 生成, 收集到 count 个 hubump 可行的候选为止; 不可行候选也写入 input_test,
    保证贪心布局所需 [BASE_SEED, BASE_SEED+m-1] 是连续区间, 一次性跑通。

只产出替换所需的 JSON, 不生成/改写 .cfg 与 config zip:
  input_test/     全部候选的 input json (连续 seed)
  placement/      全部候选的贪心布局 json (含 x/y/rotation)
  mapping.json    {target_id(int) -> seed(int)}

用法:
  python gen_replacement_layouts.py                 # 替换全部 dropped
  python gen_replacement_layouts.py --count 40      # 只替换前 40 个(调试)
  python gen_replacement_layouts.py --only-check    # 只生成+校验, 不跑贪心
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DS = PROJECT / "Dataset"
DATASET = DS / "dataset"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess_bump_region import compute_hubump, _connection_matrix  # noqa: E402

BW = [128, 256, 512, 1024]
BASE_SEED = 600001
MIN_BODY = 3.0          # 新约束: 计算后 body(本体)长宽都要 > 3mm
TMP = Path("/tmp/replacement")
IN_DIR = TMP / "input_test"
PLACE_DIR = TMP / "placement"

# EMIB 默认参数, 与 input_preprocess.matrix_to_connections 一致 (interfaceC)
_LINEAR_IO = 40.0
_MAX_REACH = 100.0
_AREA_IO = 80.0


# --------------------------------------------------------------------------- #
# 生成逻辑(复刻原始数据集规则)
# --------------------------------------------------------------------------- #
def gen_dims(n: int, rng: random.Random) -> list[tuple[int, int]]:
    dims = []
    for _ in range(n):
        w = rng.randint(3, 30)
        lo = max(3, math.ceil(w * 0.8))
        hi = min(30, math.floor(w * 1.25))
        h = rng.randint(lo, hi)
        dims.append((w, h))
    return dims


def gen_weighted_edges(n: int, rng: random.Random) -> list[tuple[int, int, int]]:
    # 连通生成树
    order = list(range(n))
    rng.shuffle(order)
    edges: list[tuple[int, int]] = []
    connected = [order[0]]
    for node in order[1:]:
        parent = rng.choice(connected)
        edges.append((parent, node))
        connected.append(node)
    # 25% 额外边
    tree = set((min(a, b), max(a, b)) for a, b in edges)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in tree:
                continue
            if rng.random() < 0.25:
                edges.append((i, j))
    return [(a, b, rng.choice(BW)) for a, b in edges]


def _add_emib(connections: list[dict]) -> list[dict]:
    """给每条连接补齐 EMIB 字段(interfaceC 默认), 供贪心布局 load_emib_placement_json 使用。"""
    for c in connections:
        wc = float(c["wireCount"])
        emib_length = wc / _LINEAR_IO
        emib_max_width = (_MAX_REACH - 2.0 * (wc / _AREA_IO) / emib_length) if emib_length > 0 else _MAX_REACH
        emib_bump_width = (wc / _AREA_IO) / emib_length if emib_length > 0 else 0.0
        c["EMIBType"] = "interfaceC"
        c["EMIB_length"] = round(emib_length, 4)
        c["EMIB_max_width"] = round(emib_max_width, 4)
        c["EMIB_bump_width"] = round(emib_bump_width, 4)
    return connections


def gen_system(seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    n = rng.randint(3, 20)
    dims = gen_dims(n, rng)
    powers = [float(rng.randint(1, 200)) for _ in range(n)]
    weighted = gen_weighted_edges(n, rng)

    chiplets = [{"name": chr(ord("A") + i), "width": float(dims[i][0]),
                 "height": float(dims[i][1]), "power": powers[i]} for i in range(n)]
    connections = [{"node1": chr(ord("A") + a), "node2": chr(ord("A") + b),
                    "wireCount": w} for a, b, w in weighted]
    _add_emib(connections)
    return chiplets, connections


def is_feasible(chiplets: list[dict], connections: list[dict]) -> bool:
    """校验每个 chiplet 的 body(=footprint-2*hubump) 长宽都 > MIN_BODY, 即可用。"""
    M = _connection_matrix(chiplets, connections)
    n = len(chiplets)
    for i, c in enumerate(chiplets):
        s = sum(M[i][j] + M[j][i] for j in range(n))
        try:
            hu = compute_hubump(float(c["width"]), float(c["height"]), s)
        except ValueError:
            return False
        if float(c["width"]) - 2 * hu <= MIN_BODY or float(c["height"]) - 2 * hu <= MIN_BODY:
            return False
    return True


def read_bad_ids() -> list[int]:
    """扫描 body 数据(placement_dataset_tw), 返回所有「有 chiplet body 长或宽 < 3mm」的 system 编号(升序)。"""
    body_dir = DATASET / "placement_dataset" / "placement_dataset_tw"
    ids = []
    for k in range(1, 77):
        fp = body_dir / f"chiplet_dataset_{k}.json"
        if not fp.exists():
            continue
        data = json.loads(fp.read_text(encoding="utf-8"))
        for sid, rec in data.items():
            m = re.fullmatch(r"system_(\d+)", sid)
            if not m:
                continue
            i = int(m.group(1))
            for c in rec.get("chiplets", []):
                if float(c["width"]) < MIN_BODY or float(c["height"]) < MIN_BODY:
                    ids.append(i)
                    break
    return sorted(ids)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=None, help="默认替换全部 dropped; 可指定只替换前 count 个")
    ap.add_argument("--workers", type=int, default=20, help="贪心布局并行 chunk 数")
    ap.add_argument("--only-check", action="store_true", help="只生成+校验, 不跑贪心布局")
    args = ap.parse_args()

    targets = read_bad_ids()
    count = len(targets) if args.count is None else min(args.count, len(targets))
    print(f"[repl] 读到 {len(targets)} 个 body<3 的 system, 本次替换前 {count} 个", flush=True)

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
            print(f"[repl] 已生成候选 {m}, 可行 {len(feasible_seeds)}/{count}", flush=True)

    print(f"[repl] 候选 {len(seed_to)} 个(含不可行 {len(seed_to) - len(feasible_seeds)} 个), "
          f"可行 {len(feasible_seeds)} 个", flush=True)

    # 写 input_test(全部候选, 连续 seed) 保证贪心布局区间连续
    for seed, (chiplets, connections) in seed_to.items():
        (IN_DIR / f"system_{seed}.json").write_text(
            json.dumps({"chiplets": chiplets, "connections": connections}), encoding="utf-8")

    # 建立 mapping: target -> 可行 seed
    mapping: dict[str, int] = {}
    for k, target in enumerate(targets[:count]):
        mapping[str(target)] = feasible_seeds[k]

    (TMP / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[repl] 生成完成: {len(mapping)} 个可行布局, 映射 -> {TMP / 'mapping.json'}", flush=True)

    # 记录替换范围内的布局清单 (30w-32w 热仿真 / 34w-36w 线长仿真), 供重仿真使用
    thermal_ids = sorted(int(k) for k in mapping if 300001 <= int(k) <= 320000)
    wl_ids = sorted(int(k) for k in mapping if 340001 <= int(k) <= 360000)
    (TMP / "replaced_thermal.txt").write_text(
        "".join(f"{t}\t{mapping[str(t)]}\n" for t in thermal_ids), encoding="utf-8")
    (TMP / "replaced_wirelength.txt").write_text(
        "".join(f"{t}\t{mapping[str(t)]}\n" for t in wl_ids), encoding="utf-8")
    print(f"[repl] 记录清单: 30w-32w 替换 {len(thermal_ids)} 个 -> replaced_thermal.txt; "
          f"34w-36w 替换 {len(wl_ids)} 个 -> replaced_wirelength.txt", flush=True)

    if args.only_check:
        return

    # 贪心布局(连续 seed 区间, 分 chunk 并行, 在 dataset/ 目录下运行, 与 gendataset.sh 一致)
    lo = BASE_SEED
    hi = BASE_SEED + len(seed_to) - 1
    workers = max(1, args.workers)
    print(f"[repl] 开始贪心布局 seed {lo}..{hi} (共 {len(seed_to)} 个, {workers} 并行)...", flush=True)

    log_dir = TMP / "greedy_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    per = (len(seed_to) + workers - 1) // workers
    procs: list[tuple[int, int, subprocess.Popen]] = []
    chunk_lo = lo
    while chunk_lo <= hi:
        chunk_hi = min(chunk_lo + per - 1, hi)
        cmd = [sys.executable, str(PROJECT / "gen_dataset" / "gen_legal_pla_greedy.py"),
               "--input-dir", str(IN_DIR), "--output-dir", str(PLACE_DIR),
               "--start", str(chunk_lo), "--end", str(chunk_hi)]
        logf = open(log_dir / f"greedy_{chunk_lo}_{chunk_hi}.log", "w")
        procs.append((chunk_lo, chunk_hi, subprocess.Popen(cmd, cwd=str(DATASET), stdout=logf, stderr=subprocess.STDOUT)))
        chunk_lo = chunk_hi + 1

    fails = 0
    for _lo, _hi, p in procs:
        if p.wait() != 0:
            fails += 1
            print(f"[repl] 贪心布局失败 chunk {_lo}..{_hi}", file=sys.stderr, flush=True)
    if fails:
        print(f"[repl] 贪心布局失败 {fails} 个 chunk", file=sys.stderr)
        sys.exit(1)
    print("[repl] 贪心布局完成", flush=True)


if __name__ == "__main__":
    main()
