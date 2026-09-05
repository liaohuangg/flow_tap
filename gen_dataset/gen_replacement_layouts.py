#!/usr/bin/env python3
"""
生成 40 个「bump region 可行」的合法布局, 用于原位替换 dropped_systems.txt 里记录的
40 个非法布局(小 die + 过高 wireCount, 导致 hubump 环宽超过本体, body<=0)。

生成规则复刻原始数据集(README STEP 1 / gendataset.sh):
  - 芯片数 3~20
  - 尺寸 3~30 mm, 长宽比 0.8~1.25 (整数)
  - 功耗 1~200
  - 连接: 连通生成树 + 25% 额外边, 带宽 128/256/512/1024
  - 确定性 seed (seed=400001+k)

流程: 生成 cfg -> input_test json(cfg_to_json) -> greedy 布局(gen_legal_pla_greedy.py)
       -> 组装 system 记录(pack 逻辑) -> 校验 hubump 可行。

输出到 /tmp/replacement/:
  cfg/            40 个新 .cfg (system_<seed>.cfg)
  input_test/     40 个 input json
  placement/      40 个 greedy 布局 json
  mapping.json    {target_id(int) -> seed(int)}

用法:
  python gen_replacement_layouts.py --count 40
  python gen_replacement_layouts.py --count 40 --only-check   # 只生成+校验, 不跑 greedy
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

PROJECT = Path("/root/placement/flow_tap")
DS = PROJECT / "Dataset"
DATASET = DS / "dataset"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from input_preprocess import cfg_to_json  # noqa: E402
from preprocess_bump_region import compute_hubump, _connection_matrix  # noqa: E402

BW = [128, 256, 512, 1024]
BASE_SEED = 400001
TMP = Path("/tmp/replacement")
CFG_DIR = TMP / "cfg"
IN_DIR = TMP / "input_test"
PLACE_DIR = TMP / "placement"


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


def edges_to_matrix(n: int, weighted: list[tuple[int, int, int]]) -> list[list[int]]:
    M = [[0] * n for _ in range(n)]
    for a, b, w in weighted:
        M[a][b] = w
        M[b][a] = w
    return M


def format_cfg(n: int, dims: list[tuple[int, int]], powers: list[float],
               M: list[list[int]], weighted: list[tuple[int, int, int]]) -> str:
    rows = [",".join(str(x) for x in M[i]) for i in range(n)]
    us, vs, es = [], [], []
    for a, b, w in sorted(weighted):
        us.append(str(a)); vs.append(str(b)); es.append(str(w))
    lines = [
        "[general]",
        "path = outputs/system/",
        "",
        "[chiplets]",
        f"chiplet_count = {n}",
        "widths = \t" + ",\t".join(str(d[0]) for d in dims),
        "heights = \t" + ",\t".join(str(d[1]) for d in dims),
        "powers = \t" + ",\t".join(str(int(p)) for p in powers),
        "target_reward = 0",
        "connections = " + ";\n\t\t\t".join(rows),
        "u =  " + ", ".join(us),
        "v =  " + ", ".join(vs),
        "e =  " + ", ".join(es),
        "x = " + ", ".join(["0"] * n),
        "y = " + ", ".join(["0"] * n),
        "",
    ]
    return "\n".join(lines)


def gen_system(seed: int) -> tuple[list[dict], list[dict], str]:
    rng = random.Random(seed)
    n = rng.randint(3, 20)
    dims = gen_dims(n, rng)
    powers = [float(rng.randint(1, 200)) for _ in range(n)]
    weighted = gen_weighted_edges(n, rng)
    M = edges_to_matrix(n, weighted)

    chiplets = [{"name": chr(ord("A") + i), "width": float(dims[i][0]),
                 "height": float(dims[i][1]), "power": powers[i]} for i in range(n)]
    connections = [{"node1": chr(ord("A") + a), "node2": chr(ord("A") + b),
                    "wireCount": w} for a, b, w in weighted]
    cfg_text = format_cfg(n, dims, powers, M, weighted)
    return chiplets, connections, cfg_text


def is_feasible(chiplets: list[dict], connections: list[dict]) -> bool:
    """校验每个 chiplet 的 body(=footprint-2*hubump) > 0, 即 bump region 可行。"""
    M = _connection_matrix(chiplets, connections)
    n = len(chiplets)
    for i, c in enumerate(chiplets):
        s = sum(M[i][j] + M[j][i] for j in range(n))
        hu = compute_hubump(float(c["width"]), float(c["height"]), s)
        if float(c["width"]) - 2 * hu <= 0 or float(c["height"]) - 2 * hu <= 0:
            return False
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def read_dropped_ids() -> list[int]:
    p = DATASET / "placement_dataset" / "placement_dataset_tw" / "dropped_systems.txt"
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("system_"):
            ids.append(int(line.split()[0].replace("system_", "")))
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", type=int, default=40)
    ap.add_argument("--only-check", action="store_true", help="只生成+校验, 不跑 greedy 布局")
    args = ap.parse_args()

    targets = read_dropped_ids()
    print(f"[repl] 读到 {len(targets)} 个非法编号: {targets}", flush=True)
    assert len(targets) == args.count, f"非法编号数 {len(targets)} != --count {args.count}"

    for _d in (CFG_DIR, IN_DIR, PLACE_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    mapping: dict[str, int] = {}
    retries = 0
    for k, target in enumerate(targets):
        seed = BASE_SEED + k
        while True:
            chiplets, connections, cfg_text = gen_system(seed)
            if is_feasible(chiplets, connections):
                break
            seed += 1000000
            retries += 1

        # 写 cfg
        (CFG_DIR / f"system_{seed}.cfg").write_text(cfg_text, encoding="utf-8")
        # 写 input_test json (与原始 input_test 相同: 无坐标 chiplets + 含 EMIB 的 connections)
        in_json = {"chiplets": chiplets, "connections": connections}
        (IN_DIR / f"system_{seed}.json").write_text(json.dumps(in_json), encoding="utf-8")

        mapping[str(target)] = seed
        print(f"[repl] [{k+1}/{len(targets)}] target system_{target} <- seed {seed} "
              f"(chiplets={len(chiplets)})", flush=True)

    (TMP / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"[repl] 生成完成: {len(mapping)} 个合法布局, 重试 {retries} 次, 映射 -> {TMP / 'mapping.json'}", flush=True)

    if args.only_check:
        return

    # greedy 布局(在 dataset/ 目录下运行, 与 gendataset.sh 一致)
    print("[repl] 开始 greedy 布局...", flush=True)
    cmd = [sys.executable, str(PROJECT / "gen_dataset" / "gen_legal_pla_greedy.py"),
           "--input-dir", str(IN_DIR), "--output-dir", str(PLACE_DIR),
           "--start", str(BASE_SEED), "--end", str(BASE_SEED + len(targets) - 1)]
    r = subprocess.run(cmd, cwd=str(DATASET))
    if r.returncode != 0:
        print("[repl] greedy 布局失败", file=sys.stderr)
        sys.exit(r.returncode)
    print("[repl] greedy 布局完成", flush=True)


if __name__ == "__main__":
    main()
