#!/usr/bin/env python3
"""生成 cpu-dram 风格的芯粒布局配置文件 (system_{start}.cfg .. system_{end}.cfg)。

格式完全对齐 Dataset/config/system_1.cfg (官方 cpu-dram.cfg 标准):
  [general] path=...; [chiplets] chiplet_count/widths/heights/powers/
  target_reward/connections 对称矩阵/u v e 有向边列表/x y 初始位置。

随机规则:
  - 芯片数 n: 3 ~ 20 随机
  - 尺寸: 宽/高均为 3 ~ 30 的整数, 长宽比 (w/h) 约束在 0.8 ~ 1.25
  - 功耗: 1 ~ 200 随机整数
  - 连接: 带宽只取 {128, 256, 512, 1024}, 对称矩阵, 保证整图连通
      第一步 生成随机生成树 (打乱编号, 每个新节点随机挂到一个已连通节点) -> 保证连通
      第二步 对每条未连边的芯片对 (i<j) 以 25% 概率额外加一条边

用法:
  python gen_cfg.py --start 380001 --end 400000
  python gen_cfg.py --start 380001 --end 400000 --seed 42 --out_dir Dataset/config
"""
from __future__ import annotations

import argparse
import math
import os
import random

BW_VALUES = [128, 256, 512, 1024]
EXTRA_EDGE_PROB = 0.25

N_CHIPS_MIN, N_CHIPS_MAX = 3, 20
DIM_MIN, DIM_MAX = 3, 30
ASPECT_MIN, ASPECT_MAX = 0.8, 1.25
POWER_MIN, POWER_MAX = 1, 200

DEFAULT_OUT_DIR = "/root/placement/flow_tap/Dataset/config"


def gen_size(rng: random.Random):
    """随机宽高, 保证 3<=w<=30, 3<=h<=30, 0.8 <= w/h <= 1.25。"""
    w = rng.randint(DIM_MIN, DIM_MAX)
    lo = max(DIM_MIN, math.ceil(w / ASPECT_MAX))   # w/h <= 1.25 => h >= w/1.25
    hi = min(DIM_MAX, math.floor(w / ASPECT_MIN))  # w/h >= 0.8  => h <= w/0.8
    h = rng.randint(lo, hi)
    return w, h


def gen_connections(n: int, rng: random.Random):
    """生成对称连接矩阵 (n×n, 对角 0), 保证连通, 带宽只取 BW_VALUES。"""
    mat = [[0] * n for _ in range(n)]

    # 第一步: 随机生成树 (打乱编号, 每个新节点随机挂到一个已连通节点)
    order = list(range(n))
    rng.shuffle(order)
    connected = [order[0]]
    for node in order[1:]:
        parent = rng.choice(connected)
        bw = rng.choice(BW_VALUES)
        mat[parent][node] = bw
        mat[node][parent] = bw
        connected.append(node)

    # 第二步: 每条未连边的芯片对 (i<j) 以 25% 概率额外加边
    for i in range(n):
        for j in range(i + 1, n):
            if mat[i][j] == 0 and rng.random() < EXTRA_EDGE_PROB:
                bw = rng.choice(BW_VALUES)
                mat[i][j] = bw
                mat[j][i] = bw

    return mat


def _fmt_row(vals):
    """一行矩阵: 首元素后每个值前加 ',\\t' (与官方 cfg 一致)。"""
    return str(vals[0]) + "".join(",\t" + str(v) for v in vals[1:])


def build_cfg(n: int, widths, heights, powers, mat) -> str:
    """按 system_1.cfg 的精确格式拼接一个 config 文本。"""
    lines = []
    lines.append("[general]")
    lines.append("path = outputs/system/")
    lines.append("")
    lines.append("[chiplets]")
    lines.append(f"chiplet_count = {n}")
    lines.append("widths = \t" + ",\t".join(map(str, widths)))
    lines.append("heights = \t" + ",\t".join(map(str, heights)))
    lines.append("powers = \t" + ",\t".join(map(str, powers)))
    lines.append("target_reward = 0")

    # connections 对称矩阵: 首行带 'connections = ' 前缀, 续行 3 个 tab 缩进,
    # 除最后一行外每行末尾加分号
    lines.append("connections = " + _fmt_row(mat[0]) + ";")
    for i in range(1, n - 1):
        lines.append("\t\t\t" + _fmt_row(mat[i]) + ";")
    lines.append("\t\t\t" + _fmt_row(mat[n - 1]))

    # u/v/e: 枚举所有有向边 (源 i 升序, 目标 j 升序)
    u, v, e = [], [], []
    for i in range(n):
        for j in range(n):
            if mat[i][j] > 0:
                u.append(i)
                v.append(j)
                e.append(mat[i][j])
    lines.append("u =  " + ", ".join(map(str, u)))
    lines.append("v =  " + ", ".join(map(str, v)))
    lines.append("e =  " + ", ".join(map(str, e)))

    zeros = ",".join("0" for _ in range(n))
    lines.append("x = " + zeros)
    lines.append("y = " + zeros)

    return "\n".join(lines) + "\n"


def gen_system(n: int, rng: random.Random):
    """生成一个 system 的 (widths, heights, powers, mat)。"""
    widths, heights, powers = [], [], []
    for _ in range(n):
        w, h = gen_size(rng)
        widths.append(w)
        heights.append(h)
        powers.append(rng.randint(POWER_MIN, POWER_MAX))
    mat = gen_connections(n, rng)
    return widths, heights, powers, mat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=380001)
    ap.add_argument("--end", type=int, default=400000)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=None, help="随机种子 (默认 None=不固定)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    n_total = args.end - args.start + 1
    for idx, sid in enumerate(range(args.start, args.end + 1)):
        n = rng.randint(N_CHIPS_MIN, N_CHIPS_MAX)
        widths, heights, powers, mat = gen_system(n, rng)
        text = build_cfg(n, widths, heights, powers, mat)
        fp = os.path.join(args.out_dir, f"system_{sid}.cfg")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(text)
        if (idx + 1) % 2000 == 0 or idx == 0:
            print(f"[gen_cfg] {idx + 1}/{n_total} system_{sid} (n={n})", flush=True)

    print(f"[gen_cfg] DONE: 生成 {n_total} 个 cfg -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
