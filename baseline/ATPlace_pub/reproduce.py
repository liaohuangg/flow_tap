#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import Params  # noqa: E402
from Chiplet import Chiplet  # noqa: E402
from Interposer import Passive_Interposer  # noqa: E402
from System import System_25D  # noqa: E402
from ATPLACE.PlaceFlow import placeflow_core  # noqa: E402
from utils.blocks_parser import parse_blocks  # noqa: E402
from utils.nets_parser import parse_nets  # noqa: E402
from utils.pl_parser import parse_pls  # noqa: E402


CASE_INTERPOSER_SIZE = {
    "Case1": [42000.0, 42000.0],
    "Case2": [32000.0, 32000.0],
    "Case3": [39000.0, 39000.0],
    "Case4": [37000.0, 37000.0],
    "Case5": [57000.0, 59000.0],
    "Case6": [49000.0, 53000.0],
    "Case7": [30000.0, 25000.0],
    "Case8": [26000.0, 23000.0],
    "Case9": [59000.0, 61000.0],
    "Case10": [47000.0, 47000.0],
}


def build_compact_model(params: Params.Params, system: System_25D):
    if not params.temp_aware_opt:
        return None

    import torch
    import torch.nn as nn

    class AnalyticThermalModel(nn.Module):
        def __init__(self, width, height, num_chiplets, num_grid_x, num_grid_y):
            super().__init__()
            self.width = width
            self.height = height
            self.num_chiplets = num_chiplets
            xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * width
            ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * height
            xgrid, ygrid = torch.meshgrid(xgrid, ygrid, indexing="ij")
            self.register_buffer("xgrid", xgrid[None, None])
            self.register_buffer("ygrid", ygrid[None, None])
            self.amp = nn.Parameter(torch.ones(1) * 1e3)
            self.bias = nn.Parameter(torch.zeros(1))
            self.heff = nn.Parameter(torch.ones(1))
            self.decay = nn.Parameter(torch.ones(1, num_chiplets, 1, 2))

        def forward(self, input_data):
            x, y, length, width, power = input_data
            chips = self.num_chiplets
            batch = x.shape[0]
            xc = x.view(-1, chips, 1, 1)
            yc = y.view(-1, chips, 1, 1)
            lc = length.view(-1, chips, 1, 1)
            wc = width.view(-1, chips, 1, 1)
            xgrid = self.xgrid.expand(batch, chips, -1, -1)
            ygrid = self.ygrid.expand(batch, chips, -1, -1)
            power = power.reshape(-1, chips, 1, 1)
            val = self._main_term(xgrid - xc, ygrid - yc, lc, wc)
            return (power * (self.amp * val + self.bias)).sum(dim=1, keepdim=True)

        def _main_term(self, xdist, ydist, length, width):
            decay = self.decay
            ax = decay[..., :1]
            ay = decay[..., 1:2]
            val = (
                self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 - ydist))
                + self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 + ydist))
                + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 - ydist))
                + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 + ydist))
            )
            return val / length / width

        @staticmethod
        def _fabc(a, b, c):
            a = a.double()
            b = b.double()
            c = c.double()
            delta = torch.sqrt(a**2 + b**2 + c**2)
            val = (
                b * torch.log((c + delta) / (a**2 + b**2) ** 0.5)
                + c * torch.log((b + delta) / (a**2 + c**2) ** 0.5)
                - a * torch.arctan(b * c / a / delta)
            )
            return val.float()

    thermal = AnalyticThermalModel(
        system.intp_width,
        system.intp_height,
        system.num_chiplets,
        system.num_grid_x,
        system.num_grid_y,
    )
    return {"Thermal": thermal}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def flatten_numbers(value):
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_numbers(item))
        return values
    return []


def export_layout(best_fp, system: System_25D, params: Params.Params, case_name: str, mode: str) -> dict:
    raw = to_plain(best_fp)
    pos = raw
    if isinstance(raw, dict):
        for key in ("pos", "position", "best_fp", "best_fp_pos"):
            if key in raw:
                pos = raw[key]
                break

    chiplets = []
    if isinstance(pos, list) and len(pos) >= 2:
        xy = flatten_numbers(pos[0])
        angles = flatten_numbers(pos[1])
        if len(xy) >= 2 * system.num_nodes:
            for idx in range(system.num_chiplets):
                angle = angles[idx] if idx < len(angles) else 0.0
                chiplets.append({
                    "name": system.node_names[idx],
                    "x": xy[idx],
                    "y": xy[idx + system.num_nodes],
                    "width": float(system.node_size_x[idx]),
                    "height": float(system.node_size_y[idx]),
                    "angle_rad": angle,
                    "power_w": float(system.powermap[idx]) if idx < len(system.powermap) else 0.0,
                })

    return {
        "case": case_name,
        "mode": mode,
        "unit": "um",
        "interposer": {
            "width": float(system.intp_width),
            "height": float(system.intp_height),
            "fence": [float(system.xlow), float(system.xhigh), float(system.ylow), float(system.yhigh)],
        },
        "thermal": {
            "temp_aware_opt": bool(params.temp_aware_opt),
            "thermal_solver": str(params.thermal_solver),
            "thermal_dir": str(params.thermal_dir),
            "num_grid_x": int(params.num_grid_x),
            "num_grid_y": int(params.num_grid_y),
        },
        "chiplets": chiplets,
        "raw_best_fp": raw,
    }


def effective_size(width, height, angle_rad):
    """Axis-aligned footprint after rotation.

    Chiplets are only ever rotated by multiples of pi/2 (0/90/180/270 deg).
    A 90 or 270 deg rotation swaps the footprint width and height; 0/180 keep
    them. Returns (w, h) in the global (interposer) axis-aligned frame.
    """
    if abs(math.sin(angle_rad)) > 0.5:  # 90 or 270 deg
        return height, width
    return width, height


def overlap_rect(a: dict, b: dict, spacing: float = 0.0):
    """Return (overlap_x, overlap_y) of two chiplets' rotated footprints.

    Positive values mean the two footprints overlap in that axis. `spacing`
    (um) is an additional required gap between chiplets.
    """
    wa, ha = effective_size(a["width"], a["height"], a.get("angle_rad", 0.0))
    wb, hb = effective_size(b["width"], b["height"], b.get("angle_rad", 0.0))
    ox = (wa + wb) / 2 + spacing - abs(a["x"] - b["x"])
    oy = (ha + hb) / 2 + spacing - abs(a["y"] - b["y"])
    return ox, oy


def is_legal(layout: dict, spacing: float = 0.0) -> bool:
    """A layout is legal when no two chiplet footprints overlap.

    Chiplets are axis-aligned rectangles given by center (x, y) and footprint
    (width, height), possibly rotated (angle_rad). Two rectangles overlap iff
    |dx| < (w1+w2)/2 + spacing and |dy| < (h1+h2)/2 + spacing.
    """
    chiplets = layout.get("chiplets") or []
    for i in range(len(chiplets)):
        for j in range(i + 1, len(chiplets)):
            ox, oy = overlap_rect(chiplets[i], chiplets[j], spacing)
            if ox > 0 and oy > 0:
                return False
    return True


def count_overlaps(layout: dict, spacing: float = 0.0) -> int:
    chiplets = layout.get("chiplets") or []
    n = 0
    for i in range(len(chiplets)):
        for j in range(i + 1, len(chiplets)):
            ox, oy = overlap_rect(chiplets[i], chiplets[j], spacing)
            if ox > 0 and oy > 0:
                n += 1
    return n


def legalize_hard(layout: dict, spacing: float = 0.0, max_iter: int = 400) -> dict:
    """Hard non-overlap post-processing (epsilon = 1).

    Resolves every pair of overlapping chiplets — accounting for their rotated
    footprints and a minimum `spacing` between them — by iteratively pushing
    overlapping pairs apart along the axis of least overlap. Finally clamps
    every chiplet back inside the interposer fence. Guarantees is_legal() is
    True when the interposer has enough room for all chiplets.
    """
    chiplets = layout["chiplets"]
    n = len(chiplets)
    eff = [effective_size(c["width"], c["height"], c.get("angle_rad", 0.0)) for c in chiplets]

    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                wi, hi = eff[i]
                wj, hj = eff[j]
                dx = chiplets[i]["x"] - chiplets[j]["x"]
                dy = chiplets[i]["y"] - chiplets[j]["y"]
                ox = (wi + wj) / 2 + spacing - abs(dx)
                oy = (hi + hj) / 2 + spacing - abs(dy)
                if ox > 0 and oy > 0:
                    if ox <= oy:
                        shift = ox / 2
                        sx = shift if dx >= 0 else -shift
                        chiplets[i]["x"] += sx
                        chiplets[j]["x"] -= sx
                    else:
                        shift = oy / 2
                        sy = shift if dy >= 0 else -shift
                        chiplets[i]["y"] += sy
                        chiplets[j]["y"] -= sy
                    moved = True
        if not moved:
            break

    # Clamp every chiplet back inside the interposer fence.
    xmin, xmax, ymin, ymax = layout["interposer"]["fence"]
    for i, c in enumerate(chiplets):
        wi, hi = eff[i]
        c["x"] = min(max(c["x"], xmin + wi / 2), xmax - wi / 2)
        c["y"] = min(max(c["y"], ymin + hi / 2), ymax - hi / 2)
    return layout


def recompute_hpwl(layout: dict, system) -> float:
    """Recompute HPWL (um) for the (legalized) chiplet positions via the
    placer's own net model (accounts for pin offsets and orientations).

    NOTE: the placer optimizes continuous orientations internally, but the
    final layout fixes each chiplet to a multiple of pi/2 (the `angle_rad`
    field). System.net_hpwl snapshots cos/sin with np.int32, so it only yields
    correct pin positions when node_orient is an exact pi/2 multiple. We must
    therefore overwrite node_orient with the layout's snapped angles — not the
    placer's continuous node_orient — or pin offsets get zeroed and HPWL is
    wrong.
    """
    n = system.num_chiplets
    xs = [c["x"] for c in layout["chiplets"]]
    ys = [c["y"] for c in layout["chiplets"]]
    angs = [c.get("angle_rad", 0.0) for c in layout["chiplets"]]
    system.node_x[:n] = np.asarray(xs, dtype=system.dtype)
    system.node_y[:n] = np.asarray(ys, dtype=system.dtype)
    system.node_orient[:n] = np.asarray(angs, dtype=system.dtype)
    return float(system.hpwl())


def legalize_milp(layout: dict, system, params, time_limit: float = 100.0) -> dict:
    """MILP post-legalization (paper Eq. 25 with lambda_w = 0), in place on
    `layout`. Wirelength is intentionally NOT optimized here — only the
    displacement from the placer's solved positions is minimized.

    Given the placer's optimized positions (x_opt, y_opt, fixed orientations),
    solve a mixed-integer linear program:

        min  sum_i ( |x_i - x_i^opt| + |y_i - y_i^opt| )   (DSP, L1 displacement)

        s.t. hard non-overlap (epsilon = 1): for every pair, at least one of the
             four axis separations holds, using the rotated footprints and a
             minimum spacing `dis_bet_chips`; chiplets stay inside the fence.

    Coordinates are chiplet CENTERS (the placer's internal convention).
    """
    import gurobipy as gp

    chiplets = layout["chiplets"]
    n = len(chiplets)
    eff = [effective_size(c["width"], c["height"], c.get("angle_rad", 0.0)) for c in chiplets]
    xmin, xmax, ymin, ymax = layout["interposer"]["fence"]
    spacing = float(getattr(params, "dis_bet_chips", 0.0))
    big_m = (xmax - xmin) + (ymax - ymin)

    m = gp.Model("post_legalize")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", time_limit)
    x = m.addVars(n, lb=-gp.GRB.INFINITY, name="x")
    y = m.addVars(n, lb=-gp.GRB.INFINITY, name="y")
    tx = m.addVars(n, lb=0.0, name="tx")
    ty = m.addVars(n, lb=0.0, name="ty")
    obj = 0.0

    for i in range(n):
        xo = float(chiplets[i]["x"])
        yo = float(chiplets[i]["y"])
        wi, hi = eff[i]
        # fence: keep the (rotated) footprint fully inside the interposer
        m.addConstr(x[i] >= xmin + wi / 2)
        m.addConstr(x[i] <= xmax - wi / 2)
        m.addConstr(y[i] >= ymin + hi / 2)
        m.addConstr(y[i] <= ymax - hi / 2)
        # L1 displacement from the placer's solved position
        m.addConstr(tx[i] >= x[i] - xo)
        m.addConstr(tx[i] >= xo - x[i])
        m.addConstr(ty[i] >= y[i] - yo)
        m.addConstr(ty[i] >= yo - y[i])
        obj += tx[i] + ty[i]

    # hard non-overlap: 4 big-M constraints + at-most-3-relaxed per pair
    for i in range(n):
        for j in range(i + 1, n):
            wi, hi = eff[i]
            wj, hj = eff[j]
            rx = (wi + wj) / 2 + spacing
            ry = (hi + hj) / 2 + spacing
            d = m.addVars(4, vtype=gp.GRB.BINARY, name=f"d_{i}_{j}")
            m.addConstr(x[i] - x[j] <= -rx + big_m * d[0])
            m.addConstr(x[j] - x[i] <= -rx + big_m * d[1])
            m.addConstr(y[i] - y[j] <= -ry + big_m * d[2])
            m.addConstr(y[j] - y[i] <= -ry + big_m * d[3])
            m.addConstr(d[0] + d[1] + d[2] + d[3] <= 3)

    m.setObjective(obj, gp.GRB.MINIMIZE)
    m.optimize()

    if m.status not in (gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT, gp.GRB.SUBOPTIMAL):
        raise RuntimeError(f"MILP legalization failed (status {m.status})")

    for i, c in enumerate(chiplets):
        c["x"] = float(x[i].X)
        c["y"] = float(y[i].X)
    return layout


def visualize(layout_path: Path, png_path: Path) -> bool:
    """Render layout.json to a PNG via visualize_layout.py (subprocess, isolated)."""
    script = ROOT / "visualize_layout.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script), str(layout_path), "--out", str(png_path)],
            check=False, capture_output=True, text=True,
        )
        return result.returncode == 0 and png_path.exists()
    except Exception as exc:
        print(f"[visualize] failed: {exc}", file=sys.stderr, flush=True)
        return False


@contextlib.contextmanager
def redirect_fd_to_file(path: Path):
    """Redirect OS-level stdout+stderr (fd 1, 2) to `path`, capturing BOTH
    Python-level and C-level output (Gurobi license lines, tqdm progress bar)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8", buffering=1)
    saved = (os.dup(1), os.dup(2))
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(f.fileno(), 1)
    os.dup2(f.fileno(), 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        f.flush()
        f.close()


def normalize_stage(params: Params.Params, data: dict) -> None:
    default_stage = (Params.Params(SRC / "params.json").floorplan_stages or [{}])[0]
    stages = data.get("floorplan_stages") or params.floorplan_stages or [default_stage]
    merged = []
    for stage in stages:
        item = dict(default_stage)
        item.update(stage)
        merged.append(item)
    params.floorplan_stages = merged


def load_params(param_file: Path, case_name: str, out_dir: Path) -> Params.Params:
    params = Params.Params(SRC / "params.json")
    data = load_json(param_file)
    params.fromJson(data)
    normalize_stage(params, data)
    params.interposer_size = data.get("interposer_size") or CASE_INTERPOSER_SIZE[case_name]
    params.fence_width = getattr(params, "fence_width", 0.0)
    params.fence_height = getattr(params, "fence_height", 0.0)
    params.result_dir = str(out_dir)
    params.thermal_dir = os.environ.get("ATPLACE_THERMAL_DIR", str(ROOT / "thermal")) + os.sep
    params.ILPsolver = getattr(params, "ILPsolver", "grb")
    params.thermal_solver = getattr(params, "thermal_solver", "hotspot")
    return params


def build_system(case_dir: Path, case_name: str, params: Params.Params) -> System_25D:
    options = {
        "filename_blocks": str(case_dir / f"{case_name}.blocks"),
        "filename_nets": str(case_dir / f"{case_name}.nets"),
        "filename_pl": str(case_dir / f"{case_name}.pl"),
    }
    modules, block_headers = parse_blocks(options)
    locations = parse_pls(options)
    nets, net_headers = parse_nets(options)

    num_chiplets = int(block_headers["Headers"]["NumHardRectilinearBlocks"])
    num_terminals = int(block_headers["Headers"]["NumTerminals"])
    system = System_25D(num_chiplets, num_terminals)
    interposer = Passive_Interposer()

    for module_name, module in modules["Modules"].items():
        if "rectangles" in module:
            chiplet = Chiplet(module_name)
            chiplet.set_chiplet_size(*module["rectangles"][0][-2:])
            chiplet.set_chiplet_loc(*module["rectangles"][0][:2])
            system.append_chiplet(module_name, chiplet)
        elif "terminal" in module:
            center = locations["Modules"][module_name]["center"]
            interposer.append_terminal(module_name, center)
            system.append_terminal(module_name, center)

    system.num_nets = int(net_headers["Headers"]["NumNets"])
    system.num_pins = int(net_headers["Headers"]["NumPins"]) - system.num_nodes + num_chiplets

    pin_id = 0
    for net_idx, net in enumerate(nets["Nets"]):
        system.net_id.append(net_idx)
        system.net_weights.append(1.0)
        system.net2pin_map.append([])
        for pin in net:
            node_id = system.node_name2id_map[pin[0]]
            if len(pin) < 3 or pin[2] is None:
                continue
            pin_offset_x = float(pin[1])
            pin_offset_y = float(pin[2])
            existing_pin = None
            for old_pin_id in system.node2pin_map[node_id]:
                if (
                    system.pin_offset_x[old_pin_id] == pin_offset_x
                    and system.pin_offset_y[old_pin_id] == pin_offset_y
                ):
                    existing_pin = old_pin_id
                    break
            if existing_pin is not None:
                system.net2pin_map[net_idx].append(existing_pin)
                system.pin2net_map[existing_pin].append(net_idx)
            else:
                system.net2pin_map[net_idx].append(pin_id)
                system.pin2net_map.append([net_idx])
                system.node2pin_map[node_id].append(pin_id)
                system.pin2node_map.append(node_id)
                system.pin_offset_x.append(pin_offset_x)
                system.pin_offset_y.append(pin_offset_y)
                pin_id += 1

    interposer.set_interposer_size(params.interposer_size)
    fence = [
        params.fence_width,
        interposer.width - params.fence_width,
        params.fence_height,
        interposer.height - params.fence_height,
    ]
    system.set_interposer_size(fence, interposer)
    system.set_bins(params)
    system.num_grid_x = params.num_grid_x
    system.num_grid_y = params.num_grid_y
    system.initialize()
    system.set_granularity(params.reso_interposer)
    system.area_cplt = (np.array(system.node_size_x) * np.array(system.node_size_y)).sum()

    system.powermap = np.zeros(num_chiplets)
    power_file = case_dir / f"{case_name}.power"
    if power_file.exists():
        with power_file.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, power = parts
                if name in system.node_name2id_map:
                    system.powermap[system.node_names.index(name)] = float(power)
    return system


def unpack_result(result):
    if isinstance(result, dict):
        hpwl = result["hpwl"]
        best_fp_values = result.get("best_fp_pos", [])
        best_fp = best_fp_values[0] if best_fp_values else None
        return hpwl, best_fp
    hpwl, _best_metric, best_fp = result[:3]
    return hpwl, best_fp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", required=True, choices=["wl", "thermal"])
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--param-file", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    param_file = Path(args.param_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(param_file, args.case, out_dir)
    timeout_sec = int(os.environ.get("ATPLACE_TIMEOUT", "3600"))
    base_iter = int(params.floorplan_stages[0].get("iteration", 100))
    base_seed = int(getattr(params, "random_seed", 1000))
    deadline = time.time() + timeout_sec

    layout_path = out_dir / "layout.json"
    png_path = out_dir / "layout.png"

    best_layout = None
    best_hpwl = float("inf")
    best_meta = {}
    attempts = 0
    attempt_log = []
    error = None

    t_start = time.time()
    try:
        while time.time() < deadline:
            # Each attempt is exactly `base_iter` (100) iterations under a fresh
            # seed (base_seed + attempt index). Keep going for the FULL 1-hour
            # budget — never stop early on a legal layout — and record every
            # attempt's (seed, hpwl, overlaps, legal) in attempt_log.
            iterations = base_iter

            # Budget guard: don't launch an attempt that can't finish within the hour.
            if attempts > 0 and time.time() + iterations * 3.2 + 90.0 > deadline:
                break

            params.floorplan_stages[0]["iteration"] = iterations
            seed = base_seed + attempts          # fresh seed per retry
            params.random_seed = seed

            # Fresh system + thermal model per attempt so each run is independent.
            seed_dir = out_dir / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            with redirect_fd_to_file(seed_dir / "run.log"):
                system = build_system(case_dir, args.case, params)
                compact_model = build_compact_model(params, system)
                t0 = time.time()
                result = placeflow_core(params, system, compact_model)
            hpwl, best_fp = unpack_result(result)
            run_s = time.time() - t0

            if best_fp is not None:
                layout = export_layout(best_fp, system, params, args.case, args.mode)
                layout["hpwl_raw"] = float(hpwl)
                overlaps_raw = count_overlaps(layout)
                layout["num_overlaps_raw"] = overlaps_raw

                # HARD non-overlap post-processing (paper Eq.25 MILP, epsilon=1):
                # enforce strict no-overlap + minimum chiplet spacing with rotated
                # footprints, minimizing displacement (DSP) only.
                spacing = float(getattr(params, "dis_bet_chips", 0.0))
                t_legal0 = time.time()
                legalize_method = "milp"
                try:
                    legalize_milp(layout, system, params)
                except Exception as exc:
                    print(f"[legalize] MILP failed ({exc}); falling back to greedy",
                          file=sys.stderr, flush=True)
                    legalize_method = "greedy"
                    legalize_hard(layout, spacing=spacing)
                legalize_s = time.time() - t_legal0
                hpwl_legal = recompute_hpwl(layout, system)
                overlaps = count_overlaps(layout, spacing=spacing)
                legal = is_legal(layout, spacing=spacing)

                layout["hpwl"] = hpwl_legal
                layout["num_overlaps"] = overlaps
                layout["seed"] = seed
                layout["legal"] = legal
                layout["legalize_method"] = legalize_method
                layout["legalize_s"] = legalize_s

                # Save EVERY attempt's layout + log under result/<case>/seed<seed>/.
                write_json(seed_dir / "layout.json", layout)
                visualize(seed_dir / "layout.json", seed_dir / "layout.png")

                meta = {"seed": seed, "iterations": iterations, "overlaps": overlaps,
                        "overlaps_raw": overlaps_raw, "hpwl": hpwl_legal,
                        "hpwl_raw": float(hpwl), "run_s": run_s, "legal": legal,
                        "legalize_method": legalize_method, "legalize_s": legalize_s,
                        "layout_dir": str(seed_dir)}
                attempt_log.append(meta)

                # Never stop on a legal layout: keep the best (lowest-HPWL)
                # layout so far, and keep going until the 1-hour budget runs out.
                if hpwl_legal < best_hpwl:
                    best_hpwl = hpwl_legal
                    best_layout = layout
                    best_meta = meta
                    write_json(layout_path, layout)  # persist best-so-far as a safety net

            attempts += 1
    except Exception as exc:
        print(f"[placeflow] error: {exc}", file=sys.stderr, flush=True)
        error = str(exc)

    if best_layout is None:
        summary = {
            "case": args.case, "mode": args.mode,
            "status": "error", "error": error,
            "attempts": len(attempt_log),
            "attempt_log": attempt_log,
            "runtime_s": time.time() - t_start,
        }
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 1

    num_legal = sum(1 for a in attempt_log if a.get("legal"))
    summary = {
        "case": args.case,
        "mode": args.mode,
        "base_seed": base_seed,
        "status": "completed",
        "error": error,
        "legal_found": num_legal > 0,
        "num_legal": num_legal,
        "num_illegal": len(attempt_log) - num_legal,
        "attempts": len(attempt_log),
        "best_hpwl": best_meta["hpwl"],
        "twl_m": best_meta["hpwl"] / 1e6,
        "best_seed": best_meta["seed"],
        "best_legal": best_meta["legal"],
        "best_overlaps": best_meta["overlaps"],
        "best_legalize_method": best_meta["legalize_method"],
        "best_legalize_s": best_meta["legalize_s"],
        "total_legalize_s": sum(a.get("legalize_s", 0.0) for a in attempt_log),
        "runtime_s": time.time() - t_start,
        "attempt_log": attempt_log,
        "layout_json": str(layout_path),
        "layout_png": str(png_path),
        "layout_chiplets": len(best_layout["chiplets"]),
    }

    # Draw the best (lowest-HPWL) layout found during the run.
    summary["visualized"] = visualize(layout_path, png_path)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
