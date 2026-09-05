import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _project_root() -> str:
    # thermalmodel/dataLoader.py -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class MinMaxStats:
    power_min: float
    power_max: float
    total_power_min: float
    total_power_max: float
    temp_min: float
    temp_max: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "power_min": float(self.power_min),
            "power_max": float(self.power_max),
            "total_power_min": float(self.total_power_min),
            "total_power_max": float(self.total_power_max),
            "temp_min": float(self.temp_min),
            "temp_max": float(self.temp_max),
        }


def minmax_scale(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = (vmax - vmin)
    if denom == 0:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - vmin) / denom).astype(np.float32)


def minmax_unscale(x01: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (x01 * (vmax - vmin) + vmin).astype(np.float32)


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


def vec_to_grid(vec: np.ndarray, grid_size: int = 128) -> np.ndarray:
    if vec.size != grid_size * grid_size:
        raise ValueError(f"Expected {grid_size*grid_size} values, got {vec.size}")
    return vec.reshape(grid_size, grid_size)


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


def flp_to_mask(flp_path: str, grid_size: int = 128) -> np.ndarray:
    rects = parse_flp_rects(flp_path)
    if not rects:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    # Use bounding box to map meters -> grid indices
    xs = [x for x, _, w, _, _ in rects] + [x + w for x, _, w, _, _ in rects]
    ys = [y for _, y, _, h, _ in rects] + [y + h for _, y, _, h, _ in rects]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # avoid degenerate
    spanx = max(maxx - minx, 1e-12)
    spany = max(maxy - miny, 1e-12)

    mask = np.zeros((grid_size, grid_size), dtype=np.float32)
    for x, y, w, h, _name in rects:
        x0 = (x - minx) / spanx
        x1 = (x + w - minx) / spanx
        y0 = (y - miny) / spany
        y1 = (y + h - miny) / spany

        # map to indices [0, grid_size)
        ix0 = int(np.floor(x0 * grid_size))
        ix1 = int(np.ceil(x1 * grid_size))
        iy0 = int(np.floor(y0 * grid_size))
        iy1 = int(np.ceil(y1 * grid_size))

        ix0 = max(0, min(grid_size, ix0))
        ix1 = max(0, min(grid_size, ix1))
        iy0 = max(0, min(grid_size, iy0))
        iy1 = max(0, min(grid_size, iy1))

        if ix1 > ix0 and iy1 > iy0:
            mask[iy0:iy1, ix0:ix1] = 1.0

    return mask


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


def compute_minmax(
    data_root: str,
    grid_size: int = 128,
    cases: Optional[List[Tuple[int, int]]] = None,
) -> MinMaxStats:
    power_dir = os.path.join(data_root, "power_map")
    totalp_dir = os.path.join(data_root, "total_power")
    temp_dir = os.path.join(data_root, "thermal_map")

    pmins, pmaxs = [], []
    tpmins, tpmaxs = [], []
    tmins, tmaxs = [], []

    it_cases = cases if cases is not None else list_cases(power_dir)

    for i, j in it_cases:
        p = read_index_value_csv(os.path.join(power_dir, f"system_power_{i}_{j}.csv"))
        tt = read_index_value_csv(os.path.join(temp_dir, f"system_temp_{i}_{j}.csv"))

        pmins.append(float(p.min()))
        pmaxs.append(float(p.max()))
        tmins.append(float(tt.min()))
        tmaxs.append(float(tt.max()))

        # total power is stored as a scalar per case
        tp = read_scalar_csv(os.path.join(totalp_dir, f"system_totalpower_{i}_{j}.csv"))
        tpmins.append(float(tp))
        tpmaxs.append(float(tp))

    return MinMaxStats(
        power_min=float(np.min(pmins)),
        power_max=float(np.max(pmaxs)),
        total_power_min=float(np.min(tpmins)),
        total_power_max=float(np.max(tpmaxs)),
        temp_min=float(np.min(tmins)),
        temp_max=float(np.max(tmaxs)),
    )


class ThermalDataset(Dataset):
    def __init__(
        self,
        thermal_map_rel: str = "Dataset/dataset/thermal_dataset",
        hotspot_cfg_rel: str = "Dataset/dataset/thermal_dataset/config",
        power_grid_size: int = 128,
        temp_grid_size: int = 128,
        stats: Optional[MinMaxStats] = None,
        cases: Optional[List[Tuple[int, int]]] = None,
    ):
        self.power_grid_size = power_grid_size
        self.temp_grid_size = temp_grid_size
        self.data_root = os.path.join(_project_root(), thermal_map_rel)
        self.hotspot_root = os.path.join(_project_root(), hotspot_cfg_rel)

        self.power_dir = os.path.join(self.data_root, "power_map")
        self.totalp_dir = os.path.join(self.data_root, "total_power")
        self.avgtemp_dir = os.path.join(self.data_root, "avg_temp")
        self.temp_dir = os.path.join(self.data_root, "thermal_map")

        self.cases = cases if cases is not None else list_cases(self.power_dir)
        if stats is None:
            # Power grid and temp grid may be different resolutions; stats are scalar min/max
            # over all values, so they are still well-defined.
            stats = compute_minmax(self.data_root, grid_size=self.temp_grid_size, cases=self.cases)
        self.stats = stats

    def __len__(self) -> int:
        return len(self.cases)

    def _layout_mask(self, i: int) -> np.ndarray:
        flp_path = os.path.join(self.hotspot_root, f"system_{i}_config", "system.flp")
        return flp_to_mask(flp_path, grid_size=self.power_grid_size)

    def __getitem__(self, idx: int):
        i, j = self.cases[idx]

        p_vec = read_index_value_csv(os.path.join(self.power_dir, f"system_power_{i}_{j}.csv"))
        t_vec = read_index_value_csv(os.path.join(self.temp_dir, f"system_temp_{i}_{j}.csv"))
        tp = read_scalar_csv(os.path.join(self.totalp_dir, f"system_totalpower_{i}_{j}.csv"))
        avg_t = read_scalar_csv(os.path.join(self.avgtemp_dir, f"system_avgtemp_{i}_{j}.csv"))

        p_grid = vec_to_grid(p_vec, grid_size=self.power_grid_size)
        t_grid = vec_to_grid(t_vec, grid_size=self.temp_grid_size)
        mask = self._layout_mask(i)

        p01 = minmax_scale(p_grid, self.stats.power_min, self.stats.power_max)
        t01 = minmax_scale(t_grid, self.stats.temp_min, self.stats.temp_max)

        # total power is a scalar; normalize it and also scale by (128*128) as requested
        tp_scaled = tp / float(self.power_grid_size * self.power_grid_size)
        tp01 = minmax_scale(
            np.asarray(tp_scaled, dtype=np.float32),
            self.stats.total_power_min / float(self.power_grid_size * self.power_grid_size),
            self.stats.total_power_max / float(self.power_grid_size * self.power_grid_size),
        )

        # avg temp is a scalar in the same units as temp grid; normalize with temp stats
        avg01 = minmax_scale(np.asarray(avg_t, dtype=np.float32), self.stats.temp_min, self.stats.temp_max)

        # tensors
        power = torch.from_numpy(p01).unsqueeze(0)  # (1,Hp,Wp)
        layout = torch.from_numpy(mask).unsqueeze(0)  # (1,Hp,Wp)
        temp = torch.from_numpy(t01).unsqueeze(0)  # (1,Ht,Wt)
        total_power = torch.tensor([[float(tp01)]], dtype=torch.float32).view(1)  # (1,)
        avg_temp = torch.tensor([[float(avg01)]], dtype=torch.float32).view(1)  # (1,)

        return {
            "i": i,
            "j": j,
            "power": power,
            "layout": layout,
            "temp": temp,
            "total_power": total_power,
            "avg_temp": avg_temp,
        }
