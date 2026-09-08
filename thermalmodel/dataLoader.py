"""dataLoader.py — GNN+HRNet 数据加载的底层工具函数。

本文件只保留被 `gnnhrnet.py` 复用的小工具:case 枚举与划分、FLP 解析、
CSV 读取、min-max 归一化。旧的 128×128 `ThermalDataset`(HRNet 专用数据类)
及其配套(`compute_minmax` / `MinMaxStats` / `flp_to_mask` / `vec_to_grid` 等)
已随纯 HRNet 管线一并删除。
"""
import os
import random
import re
from typing import List, Tuple

import numpy as np


def minmax_scale(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = (vmax - vmin)
    if denom == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / denom).astype(np.float32)


def read_scalar_csv(path: str) -> float:
    # file contains a single number (possibly with newline)
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    return float(s)


def read_index_value_csv(path: str) -> np.ndarray:
    # format: idx,value per line. idx may start from 0 or 1.
    # returns values in file order; we ignore idx and assume it's consistent.
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            vals.append(float(parts[1]))
    return np.asarray(vals, dtype=np.float32)


def parse_flp_rects(flp_path: str) -> List[Tuple[float, float, float, float, str]]:
    # Returns [(x, y, w, h, name), ...] only for chiplets (exclude TIM blocks like T0...)
    rects: List[Tuple[float, float, float, float, str]] = []
    with open(flp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue
            name = parts[0]
            if name.startswith("T"):
                continue
            w = float(parts[1])
            h = float(parts[2])
            x = float(parts[3])
            y = float(parts[4])
            rects.append((x, y, w, h, name))
    return rects


def interposer_side_m(flp_path: str) -> float:
    """Interposer 方形边长(米) = FLP 所有块的最大覆盖范围 [0, side]。

    L4_ChipLayer.flp 的 Edge_* 块横跨 [0, side_m], 故所有块 (x+w)/(y+h) 的最大值即
    intp_size_mm / 1000, 与 power/temp 网格范围一致。
    """
    side = 0.0
    try:
        with open(flp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                if len(parts) < 5:
                    continue
                try:
                    w = float(parts[1])
                    h = float(parts[2])
                    x = float(parts[3])
                    y = float(parts[4])
                except ValueError:
                    continue
                side = max(side, x + w, y + h)
    except OSError:
        pass
    return side


def list_cases(powercsv_dir: str) -> List[Tuple[int, int]]:
    # files like system_power_i_j.csv
    cases: List[Tuple[int, int]] = []
    pat = re.compile(r"system_power_(\d+)_(\d+)\.csv$")
    for fn in os.listdir(powercsv_dir):
        m = pat.match(fn)
        if not m:
            continue
        i = int(m.group(1))
        j = int(m.group(2))
        cases.append((i, j))
    cases.sort()
    return cases


def split_cases_by_i(
    cases: List[Tuple[int, int]],
    *,
    seed: int = 0,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Split cases by layout index i.

    All (i, j) pairs for the same i will stay in the same split.
    Deterministic given seed.
    """

    if not cases:
        return [], [], []

    # Unique i values
    iset = sorted({int(i) for i, _j in cases})

    rng = random.Random(int(seed))
    rng.shuffle(iset)

    n_i = len(iset)
    n_train_i = int(n_i * float(train_ratio))
    n_val_i = int(n_i * float(val_ratio))
    n_train_i = max(0, min(n_i, n_train_i))
    n_val_i = max(0, min(n_i - n_train_i, n_val_i))
    # remainder goes to test (so ratios don't need to sum exactly to 1.0)

    train_is = set(iset[:n_train_i])
    val_is = set(iset[n_train_i : n_train_i + n_val_i])
    test_is = set(iset[n_train_i + n_val_i :])

    train_cases = [(i, j) for (i, j) in cases if i in train_is]
    val_cases = [(i, j) for (i, j) in cases if i in val_is]
    test_cases = [(i, j) for (i, j) in cases if i in test_is]

    return train_cases, val_cases, test_cases
