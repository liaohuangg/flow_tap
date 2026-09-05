import sys
from pathlib import Path

_DIFFUSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DIFFUSION_DIR.parent
_FLOW_GCN_ROOT = _REPO_ROOT.parent
for _path in (_FLOW_GCN_ROOT, _REPO_ROOT, _DIFFUSION_DIR):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import utils
import torch
import hydra
import models
from omegaconf import OmegaConf, open_dict
import legalization
import analysis_utils
import common
import os
import time
import wandb
import pickle
import policies
import guidance
import json
import numpy as np
from PIL import Image
from models import _cfg_bool, _cfg_float, _smoothstep
from evaluation.wirelength_bbox import export_wirelength_bbox_results
from train_graph_thermal import (
    _build_thermal_model_from_ckpt,
    _load_thermal_checkpoint,
    _thermal_output_to_grid_and_avg,
    _thermal_rasterize,
    _denorm_temp_k,
)

CORE_METRIC_KEYS = (
    "idx",
    "thermal_max_c",
    "thermal_mean_c",
    "thermal_avg_head_c",
    "tap_intp_size",
    "tap_avg_wirelength",
    "hpwl_ratio",
    "hpwl_rescaled",
    "legality_2",
    "expanded_legality_2",
    "bbox_area_ratio",
    "model_time",
    "generation_time",
    "eval_time",
)


def _core_metrics(metrics):
    return {key: metrics[key] for key in CORE_METRIC_KEYS if key in metrics}


def _as_float_list(value):
    if isinstance(value, torch.Tensor):
        return [float(v) for v in value.detach().cpu().view(-1).tolist()]
    return [float(v) for v in np.asarray(value).reshape(-1).tolist()]


def _chip_size_tensor(cond, *, device=None, dtype=None):
    if "chip_size" not in cond:
        return torch.ones((2,), device=device, dtype=dtype or torch.float32), torch.zeros(
            (2,), device=device, dtype=dtype or torch.float32
        )
    chip_size = torch.as_tensor(cond.chip_size, device=device, dtype=dtype or torch.float32).view(-1)
    if chip_size.numel() == 4:
        return (chip_size[2:] - chip_size[:2]).clamp_min(1e-12), chip_size[:2]
    return chip_size[:2].clamp_min(1e-12), torch.zeros((2,), device=chip_size.device, dtype=chip_size.dtype)


def _source_chiplet_sizes(source):
    if not source or "chiplets" not in source:
        return None
    return torch.tensor(
        [[float(ch.get("width", 0.0)), float(ch.get("height", 0.0))] for ch in source["chiplets"]],
        dtype=torch.float32,
    )


def _connection_matrix_from_source(source):
    if not source or "chiplets" not in source:
        return None
    names = [str(ch.get("name", f"C{i}")) for i, ch in enumerate(source["chiplets"])]
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    matrix = [[0 for _ in names] for _ in names]
    for conn in source.get("connections", []):
        n1 = str(conn.get("node1", ""))
        n2 = str(conn.get("node2", ""))
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue
        i = name_to_idx[n1]
        j = name_to_idx[n2]
        if i == j:
            continue
        wire_count = int(round(float(conn.get("wireCount", conn.get("edge_weight", 0)) or 0)))
        matrix[i][j] += wire_count
        matrix[j][i] += wire_count
    return matrix


def _cond_benchmark_name(cond):
    if "benchmark_name" not in cond:
        return None
    name = cond.benchmark_name
    if isinstance(name, (list, tuple)):
        name = name[0] if name else None
    return None if name is None else str(name)


def _tap_intp_size_for_cond(cond):
    name = _cond_benchmark_name(cond)
    if not name:
        return None
    import configparser

    flow_root = Path(__file__).resolve().parents[2]
    candidates = []
    if os.environ.get("TAP25D_ROOT"):
        candidates.append(Path(os.environ["TAP25D_ROOT"]) / "configs" / "atplace_cases" / f"{name}.cfg")
    candidates.extend([
        flow_root / "TAP-2.5D" / "configs" / "atplace_cases" / f"{name}.cfg",
        flow_root.parent / "TAP-2.5D" / "configs" / "atplace_cases" / f"{name}.cfg",
    ])
    for path in candidates:
        if not path.exists():
            continue
        parser = configparser.ConfigParser()
        parser.read(path)
        try:
            return float(parser.get("interposer", "intp_size"))
        except Exception:
            return None
    return None


def _tap25d_root():
    if os.environ.get("TAP25D_ROOT"):
        return Path(os.environ["TAP25D_ROOT"])
    flow_root = Path(__file__).resolve().parents[2]
    candidates = [
        flow_root / "TAP-2.5D",
        flow_root.parent / "TAP-2.5D",
    ]
    for path in candidates:
        if (path / "routing.py").exists():
            return path
    return candidates[0]


def _add_cplex_python_path():
    if os.environ.get("CPLEX_PYTHON_DIR"):
        cplex_python_dir = Path(os.environ["CPLEX_PYTHON_DIR"])
    elif os.environ.get("CPLEX_STUDIO_DIR"):
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        cplex_python_dir = Path(os.environ["CPLEX_STUDIO_DIR"]) / "cplex" / "python" / py_ver / "x86-64_linux"
    else:
        return
    if cplex_python_dir.exists() and str(cplex_python_dir) not in sys.path:
        sys.path.insert(0, str(cplex_python_dir))


def _retarget_to_tap_canvas(x, cond):
    intp_size = _tap_intp_size_for_cond(cond)
    if intp_size is None or intp_size <= 0.0:
        return x, cond, None

    source = utils._load_benchmark_input_json(cond)
    source_sizes = _source_chiplet_sizes(source)
    if source_sizes is None or source_sizes.shape[0] != cond.x.shape[0]:
        return x, cond, None

    device = cond.x.device
    dtype = cond.x.dtype
    old_size, old_offset = _chip_size_tensor(cond, device=device, dtype=dtype)
    new_size = torch.tensor([intp_size, intp_size], device=device, dtype=dtype)
    new_offset = torch.zeros((2,), device=device, dtype=dtype)

    x_retargeted = x.clone()
    coords = x_retargeted[..., :2]
    phys = ((coords + 1.0) / 2.0) * old_size.view(*((1,) * (coords.dim() - 1)), 2) + old_offset.view(
        *((1,) * (coords.dim() - 1)), 2
    )
    x_retargeted[..., :2] = 2.0 * ((phys - new_offset.view(*((1,) * (coords.dim() - 1)), 2)) / new_size.view(*((1,) * (coords.dim() - 1)), 2)) - 1.0

    retargeted = cond.clone()
    retargeted.chip_size = torch.tensor([0.0, 0.0, intp_size, intp_size], device=device, dtype=dtype)
    retargeted.x = 2.0 * source_sizes.to(device=device, dtype=dtype) / new_size.view(1, 2)

    if "edge_attr" in retargeted and retargeted.edge_attr is not None and retargeted.edge_attr.shape[-1] >= 4:
        edge_attr = retargeted.edge_attr.clone()
        scale = (old_size / new_size).view(1, 2)
        edge_attr[:, :2] = edge_attr[:, :2] * scale
        edge_attr[:, 2:4] = edge_attr[:, 2:4] * scale
        retargeted.edge_attr = edge_attr

    return x_retargeted, retargeted, intp_size


def _compute_tap_hubump(widths, heights, connection_matrix, link_type="nppl"):
    hubumps = []
    for i in range(len(widths)):
        s = 0
        for j in range(len(widths)):
            s += connection_matrix[i][j] + connection_matrix[j][i]
        if link_type == "ppl":
            s *= 2
        h = 1
        w_stretch = 0.045 * h
        while ((widths[i] + heights[i]) * 2 * w_stretch + 4 * w_stretch * w_stretch) / 0.045 / 0.045 < s:
            h += 1
            w_stretch = 0.045 * h
            if h > 1000:
                raise RuntimeError("microbump is too high to be a feasible case")
        hubumps.append(w_stretch)
    return hubumps


def _prepare_tap_expanded_cond(cond):
    source = utils._load_benchmark_input_json(cond)
    source_sizes = _source_chiplet_sizes(source)
    connection_matrix = _connection_matrix_from_source(source)
    if source_sizes is None or connection_matrix is None or source_sizes.shape[0] != cond.x.shape[0]:
        return cond, None

    chip_size, _chip_offset = _chip_size_tensor(cond, device=cond.x.device, dtype=cond.x.dtype)
    widths = source_sizes[:, 0].tolist()
    heights = source_sizes[:, 1].tolist()
    hubumps = _compute_tap_hubump(widths, heights, connection_matrix)
    hubump_tensor = torch.tensor(hubumps, device=cond.x.device, dtype=cond.x.dtype).view(-1, 1)
    size_delta = 4.0 * hubump_tensor / chip_size.view(1, 2)

    expanded = cond.clone()
    original_x = cond.x.detach().clone()
    expanded.x = cond.x + size_delta
    expanded.tap_original_x = original_x
    expanded.tap_size_delta = size_delta
    expanded.tap_hubump = hubump_tensor.view(-1)
    expanded.tap_connection_matrix = connection_matrix
    expanded.tap_source_chiplet_sizes = source_sizes.to(device=cond.x.device, dtype=cond.x.dtype)
    once_expanded = cond.clone()
    once_expanded.x = expanded.x
    expanded.tap_once_expanded_x = once_expanded.x.detach().clone()
    return expanded, {
        "source": source,
        "original_x": original_x,
        "hubump": hubump_tensor.view(-1),
        "connection_matrix": connection_matrix,
    }


def _bare_cond_from_expanded(cond):
    if "tap_original_x" not in cond:
        return cond
    bare = cond.clone()
    bare.x = cond.tap_original_x
    return bare


def _original_case_cond_for_output(cond):
    bare = _bare_cond_from_expanded(cond)
    source = utils._load_benchmark_input_json(bare)
    source_sizes = _source_chiplet_sizes(source)
    if source_sizes is None or source_sizes.shape[0] != bare.x.shape[0]:
        return bare

    chip_size, _chip_offset = _chip_size_tensor(bare, device=bare.x.device, dtype=bare.x.dtype)
    output = bare.clone()
    original_x = 2.0 * source_sizes.to(device=bare.x.device, dtype=bare.x.dtype) / chip_size.view(1, 2)
    size_delta = bare.x - original_x
    output.x = original_x
    if "edge_attr" in output and output.edge_attr is not None and output.edge_attr.shape[-1] >= 4:
        edge_attr = output.edge_attr.clone()
        edge_attr[:, :2] = edge_attr[:, :2] + size_delta[output.edge_index[0], :] / 2.0
        edge_attr[:, 2:4] = edge_attr[:, 2:4] + size_delta[output.edge_index[1], :] / 2.0
        output.edge_attr = edge_attr
    return output


def _once_expanded_cond_for_legality(cond):
    if "tap_once_expanded_x" not in cond:
        return cond
    once_expanded = cond.clone()
    once_expanded.x = cond.tap_once_expanded_x
    return once_expanded


def _thermal_cond(cond):
    return _bare_cond_from_expanded(cond)


def _normalized_centers_to_physical(x_sample, cond):
    chip_size, chip_offset = _chip_size_tensor(cond, device=x_sample.device, dtype=x_sample.dtype)
    return ((x_sample[:, :2] + 1.0) / 2.0) * chip_size.view(1, 2) + chip_offset.view(1, 2)


def _solve_tap_avg_wirelength(x_sample, cond):
    if not all(key in cond for key in ("tap_source_chiplet_sizes", "tap_hubump", "tap_connection_matrix")):
        return None

    tap_dir = _tap25d_root()
    routing_path = tap_dir / "routing.py"
    if not routing_path.exists():
        return None

    import importlib.util

    _add_cplex_python_path()

    spec = importlib.util.spec_from_file_location("tap25d_routing_eval", routing_path)
    routing = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(routing)
    except Exception as exc:
        print(f"WARNING: failed to load TAP-2.5D routing.py: {exc}")
        return None

    centers = _normalized_centers_to_physical(x_sample.detach(), cond).detach().cpu().numpy()
    sizes = cond.tap_source_chiplet_sizes.detach().cpu().numpy()

    class TapSystem:
        pass

    system = TapSystem()
    system.chiplet_count = int(sizes.shape[0])
    system.width = [float(v) for v in sizes[:, 0]]
    system.height = [float(v) for v in sizes[:, 1]]
    system.hubump = _as_float_list(cond.tap_hubump)
    system.x = [float(v) for v in centers[:, 0]]
    system.y = [float(v) for v in centers[:, 1]]
    system.connection_matrix = cond.tap_connection_matrix
    system.intp_type = "passive"
    system.link_type = "nppl"

    cwd = os.getcwd()
    try:
        os.chdir(tap_dir)
        return float(routing.solve_Cplex(system))
    except Exception as exc:
        print(f"WARNING: TAP-2.5D wirelength evaluation failed: {exc}")
        return None
    finally:
        os.chdir(cwd)


def _solve_tap_avg_wirelength_from_placement_json(placement_json_path):
    source_json_path = _source_json_for_placement(placement_json_path)
    if source_json_path is None:
        return None
    with source_json_path.open("r", encoding="utf-8") as f:
        source = json.load(f)

    if "chiplets" not in source:
        return None
    source_chiplets = source["chiplets"]
    source_sizes = _source_chiplet_sizes(source)
    connection_matrix = _connection_matrix_from_source(source)
    if source_sizes is None or connection_matrix is None:
        return None

    with open(placement_json_path, "r", encoding="utf-8") as f:
        placement = json.load(f)
    placed_by_name = {str(ch.get("name", f"C{i}")): ch for i, ch in enumerate(placement.get("chiplets", []))}

    widths = [float(ch.get("width", 0.0)) for ch in source_chiplets]
    heights = [float(ch.get("height", 0.0)) for ch in source_chiplets]
    hubumps = _compute_tap_hubump(widths, heights, connection_matrix)
    centers_x = []
    centers_y = []
    for i, src in enumerate(source_chiplets):
        name = str(src.get("name", f"C{i}"))
        placed = placed_by_name.get(name)
        if placed is None:
            return None
        centers_x.append(float(placed.get("x-position", placed.get("x", 0.0))) + widths[i] / 2.0)
        centers_y.append(float(placed.get("y-position", placed.get("y", 0.0))) + heights[i] / 2.0)

    tap_dir = _tap25d_root()
    routing_path = tap_dir / "routing.py"
    if not routing_path.exists():
        return None

    import importlib.util

    _add_cplex_python_path()

    spec = importlib.util.spec_from_file_location("tap25d_routing_eval_from_json", routing_path)
    routing = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(routing)
    except Exception as exc:
        print(f"WARNING: failed to load TAP-2.5D routing.py: {exc}")
        return None

    class TapSystem:
        pass

    system = TapSystem()
    system.chiplet_count = len(widths)
    system.width = widths
    system.height = heights
    system.hubump = hubumps
    system.x = centers_x
    system.y = centers_y
    system.connection_matrix = connection_matrix
    system.intp_type = "passive"
    system.link_type = "nppl"

    cwd = os.getcwd()
    try:
        os.chdir(tap_dir)
        return float(routing.solve_Cplex(system))
    except Exception as exc:
        print(f"WARNING: TAP-2.5D wirelength evaluation from placement json failed: {exc}")
        return None
    finally:
        os.chdir(cwd)


def cost(output_metrics):
    """
    Returns dict with cost function(s) for hyperparam sweep
    """
    legality_target = 0.995
    legality_temp = 0.001
    hpwl = torch.tensor(output_metrics["hpwl_rescaled"]).mean()

    legality = torch.tensor(output_metrics["legality_2"]).mean()
    legality_cost_factor = 1 + 10 * torch.nn.functional.relu((legality_target - legality)/legality_temp)

    full_cost = (legality_cost_factor * hpwl).item()
    costs = {
        "cost": full_cost,
    }
    return costs


class ThermalEvaluator:
    def __init__(self, thermal_cfg, device):
        self.device = device
        self.ckpt_path = thermal_cfg.get("ckpt", "none")
        if self.ckpt_path in (None, "", "none"):
            raise ValueError("eval_thermal.py requires +thermal.ckpt=<thermal guidance checkpoint>")
        self.grid_size = int(thermal_cfg.get("grid_size", 128))
        self.rect_sharpness = float(thermal_cfg.get("rect_sharpness", 80.0))
        ckpt = _load_thermal_checkpoint(self.ckpt_path)
        self.stats = ckpt.get("stats") if isinstance(ckpt.get("stats"), dict) else None
        self.model = _build_thermal_model_from_ckpt(ckpt, device)

    @torch.no_grad()
    def predict_temperature_c(self, x_sample, cond):
        cond = _thermal_cond(cond)
        x_batch = x_sample.unsqueeze(0).to(self.device)
        power_grid, layout_grid, total_power = _thermal_rasterize(
            x_batch,
            cond,
            grid_size=self.grid_size,
            rect_sharpness=self.rect_sharpness,
            stats=self.stats,
        )
        temp, avg_temp = _thermal_output_to_grid_and_avg(self.model(power_grid, layout_grid, total_power))
        has_temp_stats = self.stats is not None and "temp_min" in self.stats and "temp_max" in self.stats
        if has_temp_stats:
            temp = _denorm_temp_k(temp, self.stats) - 273.15
            if avg_temp is not None:
                avg_temp = _denorm_temp_k(avg_temp, self.stats) - 273.15
        avg_temp_c = None if avg_temp is None else avg_temp.detach().cpu().view(-1)[0].item()
        return temp.detach().cpu()[0, 0], avg_temp_c

    @torch.no_grad()
    def __call__(self, x_sample, cond):
        cond = _thermal_cond(cond)
        x_batch = x_sample.unsqueeze(0).to(self.device)
        power_grid, layout_grid, total_power = _thermal_rasterize(
            x_batch,
            cond,
            grid_size=self.grid_size,
            rect_sharpness=self.rect_sharpness,
            stats=self.stats,
        )
        temp, avg_temp = _thermal_output_to_grid_and_avg(self.model(power_grid, layout_grid, total_power))
        has_temp_stats = self.stats is not None and "temp_min" in self.stats and "temp_max" in self.stats
        if has_temp_stats:
            temp = _denorm_temp_k(temp, self.stats)
            if avg_temp is not None:
                avg_temp = _denorm_temp_k(avg_temp, self.stats)
        metrics = {
            "thermal_max_k": temp.max().detach().cpu().item(),
            "thermal_mean_k": temp.mean().detach().cpu().item(),
        }
        if avg_temp is not None:
            metrics["thermal_avg_head_k"] = avg_temp.mean().detach().cpu().item()
        if has_temp_stats:
            metrics.update(
                {
                    "thermal_max_c": (temp.max() - 273.15).detach().cpu().item(),
                    "thermal_mean_c": (temp.mean() - 273.15).detach().cpu().item(),
                }
            )
            if avg_temp is not None:
                metrics["thermal_avg_head_c"] = (avg_temp.mean() - 273.15).detach().cpu().item()
        return metrics


def _write_flp_from_placement_json(placement_json_path, flp_path):
    chiplets, _source_json_path = _chiplet_rects_from_placement_json(placement_json_path)
    os.makedirs(os.path.dirname(flp_path), exist_ok=True)
    with open(flp_path, "w", encoding="utf-8") as f:
        for name, width, height, x, y in chiplets:
            width_m = width / 1000.0
            height_m = height / 1000.0
            x_m = x / 1000.0
            y_m = y / 1000.0
            f.write(f"{name} {width_m:.12g} {height_m:.12g} {x_m:.12g} {y_m:.12g}\n")


def _source_json_for_placement(placement_json_path):
    placement_json_path = Path(placement_json_path)
    stem = placement_json_path.stem
    if stem.endswith("_placement"):
        stem = stem[: -len("_placement")]
    parts = stem.split("_", 1)
    case_name = parts[1] if len(parts) == 2 and parts[0].isdigit() else stem

    repo_root = Path(__file__).resolve().parents[1]
    flow_root = repo_root.parent
    candidates = []
    if os.environ.get("ATPLACE_BENCHMARK_ROOT"):
        candidates.append(Path(os.environ["ATPLACE_BENCHMARK_ROOT"]) / f"{case_name}.json")
    if os.environ.get("MTAP_ROOT"):
        candidates.append(Path(os.environ["MTAP_ROOT"]) / "benchmark" / "test_input" / f"{case_name}.json")
    candidates.extend([
        flow_root / "benchmark" / "ATPlace_json" / f"{case_name}.json",
        flow_root / "MTAP" / "benchmark" / "test_input" / f"{case_name}.json",
    ])
    for path in candidates:
        if path.exists():
            return path
    return None


def _chiplet_rects_from_placement_json(placement_json_path):
    with open(placement_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_json_path = _source_json_for_placement(placement_json_path)
    source_chiplets = None
    if source_json_path is not None:
        with source_json_path.open("r", encoding="utf-8") as f:
            source_data = json.load(f)
        source_chiplets = source_data.get("chiplets", [])

    rects = []
    for idx, ch in enumerate(data.get("chiplets", [])):
        name = str(ch.get("name", f"C{idx}"))
        source_ch = source_chiplets[idx] if source_chiplets is not None and idx < len(source_chiplets) else None
        width = float((source_ch or ch).get("width", 0.0))
        height = float((source_ch or ch).get("height", 0.0))
        x = float(ch.get("x-position", ch.get("x", 0.0)))
        y = float(ch.get("y-position", ch.get("y", 0.0)))
        if width > 0.0 and height > 0.0:
            rects.append((name, width, height, x, y))
    return rects, source_json_path


def _hotspot_cmap():
    from matplotlib.colors import ListedColormap

    palette_rgb = [
        (255, 0, 0),
        (255, 51, 0),
        (255, 102, 0),
        (255, 153, 0),
        (255, 204, 0),
        (255, 255, 0),
        (204, 255, 0),
        (153, 255, 0),
        (102, 255, 0),
        (51, 255, 0),
        (0, 255, 0),
        (0, 255, 51),
        (0, 255, 102),
        (0, 255, 153),
        (0, 255, 204),
        (0, 255, 255),
        (0, 204, 255),
        (0, 153, 255),
        (0, 102, 255),
        (0, 51, 255),
        (0, 0, 255),
    ]
    return ListedColormap([(r / 255.0, g / 255.0, b / 255.0) for r, g, b in reversed(palette_rgb)])


def _plot_thermal_grid_overlay_from_placement_json(
    placement_json_path,
    grid,
    output_image,
    *,
    title=None,
    vmin=None,
    vmax=None,
):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid)
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]

    rects, _source_json_path = _chiplet_rects_from_placement_json(placement_json_path)
    if rects:
        min_x = min(x for _name, _w, _h, x, _y in rects)
        min_y = min(y for _name, _w, _h, _x, y in rects)
        max_x = max(x + w for _name, w, _h, x, _y in rects)
        max_y = max(y + h for _name, _w, h, _x, y in rects)
    else:
        min_x = min_y = 0.0
        max_x = float(grid.shape[-1])
        max_y = float(grid.shape[-2])

    total_w = max(max_x - min_x, 1e-9)
    total_h = max(max_y - min_y, 1e-9)
    side = max(total_w, total_h)
    pad_x = (side - total_w) / 2.0
    pad_y = (side - total_h) / 2.0
    x0, x1 = min_x - pad_x, max_x + pad_x
    y0, y1 = min_y - pad_y, max_y + pad_y

    fig, ax = plt.subplots(1, figsize=(10, 8))
    im = ax.imshow(
        np.flipud(grid),
        cmap=_hotspot_cmap(),
        extent=(x0, x1, y0, y1),
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(axis="both", labelsize=14)

    info = f"Max={float(np.max(grid)):.2f} °C, AVG={float(np.mean(grid)):.2f} °C"
    ax.text(
        0.01,
        0.99,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        color="white",
        path_effects=[withStroke(linewidth=3, foreground="black")],
    )

    for name, width, height, x, y in rects:
        ax.add_patch(patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor="black", facecolor="none"))
        ax.text(
            x + width / 2.0,
            y + height / 2.0,
            name,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_thermal_eval_artifacts(
    *,
    thermal_evaluator,
    x_sample,
    cond,
    placement_json_path,
    thermal_eval_root,
):
    placement_json_path = Path(placement_json_path)
    out_dir = Path(thermal_eval_root) / placement_json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    temp_c, avg_head_c = thermal_evaluator.predict_temperature_c(x_sample, cond)
    temp_np = temp_c.numpy()

    flp_path = out_dir / f"{placement_json_path.stem}.flp"
    npy_path = out_dir / f"{placement_json_path.stem}_temperature_c.npy"
    heatmap_path = out_dir / f"{placement_json_path.stem}_thermal_heatmap.png"
    summary_path = out_dir / f"{placement_json_path.stem}_thermal_summary.json"
    source_json_path = _source_json_for_placement(placement_json_path)

    _write_flp_from_placement_json(str(placement_json_path), str(flp_path))
    np.save(npy_path, temp_np)
    _plot_thermal_grid_overlay_from_placement_json(
        str(placement_json_path),
        temp_c,
        str(heatmap_path),
        title=f"{placement_json_path.stem} thermal prediction",
        vmin=float(temp_np.min()),
        vmax=float(temp_np.max()),
    )

    summary = {
        "placement_json": str(placement_json_path.resolve()),
        "ckpt": str(Path(thermal_evaluator.ckpt_path).resolve()),
        "source_json": None if source_json_path is None else str(source_json_path.resolve()),
        "grid_size_input": int(thermal_evaluator.grid_size),
        "grid_size_output": list(temp_np.shape),
        "temperature_unit": "C",
        "thermal_max_c": float(temp_np.max()),
        "thermal_mean_c": float(temp_np.mean()),
        "thermal_min_c": float(temp_np.min()),
        "thermal_avg_head_c": None if avg_head_c is None else float(avg_head_c),
        "heatmap_png": str(heatmap_path.resolve()),
        "temperature_npy": str(npy_path.resolve()),
        "overlay_flp": str(flp_path.resolve()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _placement_json_data_for_export(placement, cond, *, step):
    source = utils._load_benchmark_input_json(cond)
    if source is not None and "chiplets" in source:
        chiplets = []
        for i, src in enumerate(source["chiplets"]):
            x, y = placement[i]
            chiplets.append(
                {
                    "name": str(src.get("name", f"C{i}")),
                    "x-position": float(x),
                    "y-position": float(y),
                    "width": float(src.get("width", 0.0)),
                    "height": float(src.get("height", 0.0)),
                    "rotation": int(src.get("rotation", 0)),
                    "power": src.get("power", 0.0),
                }
            )
        connections = [dict(conn) for conn in source.get("connections", [])]
    else:
        chiplets, connections = utils._fallback_placement_json_data(placement, cond)

    area, aspect_ratio = utils._placement_bbox_stats(chiplets)
    return {
        "step": int(step),
        "benchmark_name": utils._cond_benchmark_name(cond),
        "chiplets": chiplets,
        "connections": connections,
        "wirelength": utils._placement_wirelength(chiplets, connections),
        "area": area,
        "aspect_ratio": aspect_ratio,
    }


def _export_intermediate_layouts(
    *,
    intermediates,
    intermediate_steps,
    cond_preprocessed,
    cond_output_preprocessed,
    postprocess_fn,
    legalization_fn,
    legalize_intermediates,
    export_root,
    case_stem,
):
    export_root = Path(export_root)
    image_dir = export_root / "images"
    json_dir = export_root / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "case": case_stem,
        "legalized": bool(legalize_intermediates),
        "steps": [],
        "image_dir": str(image_dir.resolve()),
        "json_dir": str(json_dir.resolve()),
    }
    count = min(len(intermediates), len(intermediate_steps))
    for step, intermediate in zip(intermediate_steps[:count], intermediates[:count]):
        step = int(step)
        sample_for_image = intermediate.detach().clone()
        if sample_for_image.dim() == 2:
            sample_for_image = sample_for_image.unsqueeze(0)
        if legalize_intermediates:
            if legalization_fn is None:
                raise ValueError("intermediate_export.legalize=True requires legalization_fn")
            sample_for_image, _, _ = legalization_fn(sample_for_image, cond_preprocessed)
        image = utils.visualize_placement(
            sample_for_image[0],
            cond_output_preprocessed,
            plot_pins=True,
            plot_edges=False,
            img_size=(2048, 2048),
        )
        image_path = image_dir / f"{case_stem}_step_{step:03d}.png"
        Image.fromarray(image).save(image_path)

        sample_for_json = sample_for_image
        cond_for_json = cond_preprocessed
        if postprocess_fn is not None:
            sample_for_json, cond_for_json = postprocess_fn(sample_for_json, cond_preprocessed)
        cond_output_for_json = _original_case_cond_for_output(cond_for_json)
        placement = utils.postprocess_placement(
            sample_for_json.squeeze(dim=0).detach().to(device=cond_output_for_json.x.device),
            cond_output_for_json,
        ).detach().cpu().numpy()
        data = _placement_json_data_for_export(placement, cond_output_for_json, step=step)
        json_path = json_dir / f"{case_stem}_step_{step:03d}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        manifest["steps"].append(
            {
                "step": step,
                "image": str(image_path.resolve()),
                "json": str(json_path.resolve()),
            }
        )

    manifest_path = export_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"exported {len(manifest['steps'])} intermediate layouts to: {export_root}")
    return manifest_path


class ThermalGuidedFlowMatchingModel(models.FlowMatchingModel):
    def __init__(self, *args, thermal_cfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.thermal_cfg = dict(thermal_cfg or {})
        self.thermal_ckpt = self.thermal_cfg.get("ckpt", "none")
        self.thermal_grid_size = int(self.thermal_cfg.get("grid_size", 128))
        self.thermal_rect_sharpness = float(self.thermal_cfg.get("rect_sharpness", 80.0))
        self.thermal_guidance_weight = float(self.thermal_cfg.get("guidance_weight", 0.1))
        self.thermal_guidance_lr = float(self.thermal_cfg.get("guidance_lr", 0.02))
        self.thermal_guidance_steps = int(self.thermal_cfg.get("guidance_steps", 1))
        self.thermal_smooth_max_beta = float(self.thermal_cfg.get("smooth_max_beta", 20.0))
        self.thermal_mean_weight = float(self.thermal_cfg.get("mean_weight", 0.05))
        self.thermal_apply_after = int(self.thermal_cfg.get("apply_after", 0))
        self.thermal_apply_before = int(self.thermal_cfg.get("apply_before", self.max_diffusion_steps + 1))
        self.thermal_hpwl_weight = float(self.thermal_cfg.get("hpwl_weight", 0.0))
        self.thermal_legality_weight = float(self.thermal_cfg.get("legality_weight", 2.0))
        self.thermal_schedule = self.thermal_cfg.get("schedule", None)
        self.thermal_grad_clip = float(self.thermal_cfg.get("grad_clip", 0.05))
        self.thermal_clamp = float(self.thermal_cfg.get("clamp", 2.0))
        self.__dict__["_thermal_guidance_model"] = None
        self.__dict__["_thermal_guidance_stats"] = None

    def reverse_samples(self, B, x_in, cond, num_timesteps=-1, intermediate_every=0, mask_override=None):
        batch_shape = (B, cond.x.shape[0], self.input_shape[1])
        mask_shape = (1, x_in.shape[1], 1)

        if num_timesteps <= 0:
            num_timesteps = self.max_diffusion_steps

        x = self._epsilon_dist.sample(batch_shape).squeeze(dim=-1)
        mask = mask_override.view(*mask_shape) if mask_override is not None else self.get_mask(x_in, cond)
        x = torch.where(mask, x_in, x) if mask is not None else x

        capture_steps = getattr(self, "intermediate_capture_steps", None)
        if capture_steps is not None:
            capture_steps = {int(step) for step in capture_steps}
            intermediates = []
            intermediate_steps = []
            if 0 in capture_steps:
                intermediates.append(x.detach().clone())
                intermediate_steps.append(0)
        else:
            intermediates = [x]
            intermediate_steps = [0]
        dt = 1.0 / num_timesteps
        self.reset_guidance_state(dtype=x.dtype)
        for step in range(num_timesteps, 0, -1):
            t = float(step) / float(num_timesteps)
            t_vec = torch.full((B,), t, device=x.device)
            velocity = self(x, cond, t_vec)

            if self.is_guided_sampling:
                if self.guidance_mode == "sgd":
                    guidance_force = self.reverse_guidance_force(x, cond, t, mask)
                elif self.guidance_mode == "opt":
                    guidance_force = self.reverse_guidance_opt_force(x, cond, t, mask)
                else:
                    raise NotImplementedError
            else:
                guidance_force = 0.0

            x_next = x - dt * velocity
            if self.is_guided_sampling:
                x_next = x_next + guidance_force

            thermal_guidance_weight = self._thermal_guidance_weight_for_step(step, num_timesteps)
            if thermal_guidance_weight > 0.0:
                x_next = self._thermal_guided_step(x_next, cond, mask, thermal_guidance_weight)

            x = torch.where(mask, x_in, x_next) if mask is not None else x_next
            x = torch.clamp(x, -self.thermal_clamp, self.thermal_clamp)
            completed_step = num_timesteps - step + 1
            if capture_steps is not None:
                if completed_step in capture_steps:
                    intermediates.append(x.detach().clone())
                    intermediate_steps.append(completed_step)
            elif intermediate_every and (completed_step % intermediate_every == 0):
                intermediates.append(x)
                intermediate_steps.append(completed_step)

        if capture_steps is None or num_timesteps not in intermediate_steps:
            intermediates.append(x)
            intermediate_steps.append(num_timesteps)
        self._last_intermediate_steps = intermediate_steps
        return x, intermediates

    def _use_thermal_guidance(self, step):
        return self._thermal_guidance_weight_for_step(step, self.max_diffusion_steps) > 0.0

    def _thermal_guidance_weight_for_step(self, step, num_timesteps):
        if self.thermal_guidance_weight <= 0.0:
            return 0.0
        if self.thermal_guidance_lr <= 0.0 or self.thermal_guidance_steps <= 0:
            return 0.0
        if self.thermal_ckpt in (None, "", "none"):
            return 0.0

        if not _cfg_bool(self.thermal_schedule, "enabled", False):
            return self.thermal_guidance_weight if self.thermal_apply_after <= step <= self.thermal_apply_before else 0.0

        progress = (float(num_timesteps) - float(step) + 1.0) / max(float(num_timesteps), 1.0)
        progress = min(1.0, max(0.0, progress))
        start = _cfg_float(self.thermal_schedule, "start", 0.4)
        full = _cfg_float(self.thermal_schedule, "full", 0.8)
        final_weight = _cfg_float(self.thermal_schedule, "final_weight", self.thermal_guidance_weight)
        initial_weight = _cfg_float(self.thermal_schedule, "initial_weight", 0.0)

        if progress <= start:
            return initial_weight
        ramp = _smoothstep((progress - start) / max(full - start, 1e-6))
        if progress <= full:
            return initial_weight + ramp * (final_weight - initial_weight)
        return final_weight

    @torch.enable_grad()
    def _thermal_guided_step(self, x_current, cond, mask=None, thermal_guidance_weight=None):
        thermal_model, stats = self._load_thermal_guidance_model(x_current.device)
        thermal_cond = _thermal_cond(cond)
        x_guided = x_current.detach().clone().requires_grad_(True)
        optimizer = torch.optim.SGD((x_guided,), lr=self.thermal_guidance_lr, momentum=0.0)
        thermal_guidance_weight = self.thermal_guidance_weight if thermal_guidance_weight is None else float(thermal_guidance_weight)

        for _ in range(self.thermal_guidance_steps):
            optimizer.zero_grad()
            power_grid, layout_grid, total_power = _thermal_rasterize(
                x_guided,
                thermal_cond,
                grid_size=self.thermal_grid_size,
                rect_sharpness=self.thermal_rect_sharpness,
                stats=stats,
            )
            temp, avg_temp = _thermal_output_to_grid_and_avg(thermal_model(power_grid, layout_grid, total_power))
            flat = temp.flatten(1)
            thermal_score = torch.logsumexp(flat * self.thermal_smooth_max_beta, dim=1) / self.thermal_smooth_max_beta
            mean_temp = avg_temp.view(-1) if avg_temp is not None else flat.mean(dim=1)
            thermal_score = thermal_score + self.thermal_mean_weight * mean_temp
            objective = thermal_guidance_weight * thermal_score

            if self.thermal_hpwl_weight > 0.0:
                objective = objective + self.thermal_hpwl_weight * guidance.hpwl_guidance_potential(x_guided, cond)
            if self.thermal_legality_weight > 0.0:
                objective = objective + self.thermal_legality_weight * guidance.legality_guidance_potential(
                    x_guided,
                    cond,
                    mask=mask,
                    softmax_factor=10.0,
                )

            objective.sum().backward()
            if mask is not None and x_guided.grad is not None:
                x_guided.grad *= (~mask).float()
            if self.thermal_grad_clip > 0.0 and x_guided.grad is not None:
                x_guided.grad.data.clamp_(min=-self.thermal_grad_clip, max=self.thermal_grad_clip)
            optimizer.step()
            with torch.no_grad():
                x_guided.clamp_(min=-self.thermal_clamp, max=self.thermal_clamp)

        return x_guided.detach()

    def _load_thermal_guidance_model(self, device):
        if self.__dict__.get("_thermal_guidance_model") is not None:
            return self.__dict__["_thermal_guidance_model"], self.__dict__["_thermal_guidance_stats"]

        ckpt = _load_thermal_checkpoint(self.thermal_ckpt)
        thermal_model = _build_thermal_model_from_ckpt(ckpt, device)
        self.__dict__["_thermal_guidance_model"] = thermal_model
        self.__dict__["_thermal_guidance_stats"] = ckpt.get("stats") if isinstance(ckpt.get("stats"), dict) else None
        return self.__dict__["_thermal_guidance_model"], self.__dict__["_thermal_guidance_stats"]


def save_outputs_with_thermal(
    x_in,
    cond,
    model,
    save_folder,
    thermal_evaluator,
    thermal_eval_root=None,
    output_number_offset=0,
    policy="open_loop",
    policy_kwargs={},
    preprocess_fn=None,
    postprocess_fn=None,
    legalization_fn=None,
    intermediate_export=None,
):
    idx = cond.file_idx if "file_idx" in cond else output_number_offset
    placed_stem = utils.output_case_stem(cond, idx, "placed")
    sample_stem = utils.output_case_stem(cond, idx, "sample")
    x_in = torch.unsqueeze(x_in, dim=0).to(model.device)
    original_device = cond.x.device
    cond.to(model.device)
    metrics = {}
    metrics_special = {}

    t0 = time.time()
    x_preprocessed, cond_preprocessed = preprocess_fn(x_in, cond) if preprocess_fn is not None else (x_in, cond)
    tap_intp_size = None
    cond_preprocessed, tap_context = _prepare_tap_expanded_cond(cond_preprocessed)
    cond_bare_preprocessed = _bare_cond_from_expanded(cond_preprocessed)
    cond_output_preprocessed = _original_case_cond_for_output(cond_preprocessed)
    cond_once_expanded_preprocessed = _once_expanded_cond_for_legality(cond_preprocessed)
    case_stem = utils.output_case_stem(cond_output_preprocessed, idx, "trace").replace("_trace", "")

    export_cfg = dict(intermediate_export or {})
    export_enabled = bool(export_cfg.get("enabled", False))
    target_indices = set()
    if "target_idx" in export_cfg and export_cfg["target_idx"] is not None:
        target_indices.add(int(export_cfg["target_idx"]))
    if "target_case" in export_cfg and export_cfg["target_case"] is not None:
        target_indices.add(int(export_cfg["target_case"]) - 1)
    for target_idx in export_cfg.get("target_indices", []) or []:
        target_indices.add(int(target_idx))
    if target_indices and int(idx) not in target_indices:
        export_enabled = False
    export_steps = [int(step) for step in export_cfg.get("steps", [])] if export_enabled else []

    t1 = time.time()
    if cond_preprocessed.num_nodes == 0:
        sample = torch.zeros_like(x_preprocessed)
    else:
        if policy == "open_loop":
            old_capture_steps = getattr(model, "intermediate_capture_steps", None)
            old_last_intermediate_steps = getattr(model, "_last_intermediate_steps", None)
            if export_steps:
                model.intermediate_capture_steps = export_steps
            try:
                sample, intermediates, policy_metrics_special = policies.open_loop(
                    1,
                    model,
                    x_preprocessed,
                    cond_preprocessed,
                    intermediate_every=0,
                    save_videos=policy_kwargs["save_videos"],
                )
                if export_steps:
                    intermediate_steps = getattr(model, "_last_intermediate_steps", [])
                    export_root = export_cfg.get("output_dir")
                    if not export_root:
                        export_root = Path(save_folder).resolve().parent / "intermediate_steps" / case_stem
                    _export_intermediate_layouts(
                        intermediates=intermediates,
                        intermediate_steps=intermediate_steps,
                        cond_preprocessed=cond_preprocessed,
                        cond_output_preprocessed=cond_output_preprocessed,
                        postprocess_fn=postprocess_fn,
                        legalization_fn=legalization_fn,
                        legalize_intermediates=bool(export_cfg.get("legalize", False)),
                        export_root=export_root,
                        case_stem=case_stem,
                    )
            finally:
                if old_capture_steps is None:
                    model.__dict__.pop("intermediate_capture_steps", None)
                else:
                    model.intermediate_capture_steps = old_capture_steps
                if old_last_intermediate_steps is None:
                    model.__dict__.pop("_last_intermediate_steps", None)
                else:
                    model._last_intermediate_steps = old_last_intermediate_steps
            metrics_special.update(policy_metrics_special)
        elif policy == "open_loop_clustered":
            sample, _ = policies.open_loop_clustered(1, model, x_preprocessed, cond_preprocessed, intermediate_every=0)
        elif policy == "iterative_clustering":
            sample, policy_metrics, policy_metrics_special = policies.iterative_clustering(
                1, model, x_preprocessed, cond_preprocessed, **policy_kwargs
            )
            metrics.update(policy_metrics)
            metrics_special.update(policy_metrics_special)
        elif policy == "random":
            sample = policies.random(1, x_preprocessed, cond_preprocessed)
        else:
            raise NotImplementedError
    t2 = time.time()

    image = utils.visualize_placement(sample[0], cond_output_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048))

    if legalization_fn is not None:
        sample, legalization_metrics, legalization_metrics_special = legalization_fn(sample, cond_preprocessed)
        metrics.update(legalization_metrics)
        metrics_special.update(legalization_metrics_special)
        cond_bare_preprocessed = _bare_cond_from_expanded(cond_preprocessed)
        cond_output_preprocessed = _original_case_cond_for_output(cond_preprocessed)
        cond_once_expanded_preprocessed = _once_expanded_cond_for_legality(cond_preprocessed)
        image_legalized = utils.visualize_placement(
            sample[0], cond_output_preprocessed, plot_pins=True, plot_edges=False, img_size=(2048, 2048)
        )
    else:
        image_legalized = image
    utils.debug_plot_img(image_legalized, os.path.join(save_folder, placed_stem))

    sample_unprocessed = sample.detach().clone()
    sample, cond_postprocessed = postprocess_fn(sample, cond_preprocessed)
    cond_bare_postprocessed = _bare_cond_from_expanded(cond_postprocessed)
    cond_output_postprocessed = _original_case_cond_for_output(cond_postprocessed)

    sample = sample.squeeze(dim=0).detach().to(device=cond.x.device)
    sample = utils.postprocess_placement(sample, cond_output_postprocessed).cpu().numpy()
    save_file = os.path.join(save_folder, f"{sample_stem}.pkl")
    with open(save_file, "wb") as f:
        pickle.dump(sample, f)
    placement_json_path = utils.save_placement_json(sample, cond_output_postprocessed, save_folder, idx)
    tap_avg_wirelength = _solve_tap_avg_wirelength_from_placement_json(placement_json_path)
    thermal_eval_summary = None
    if thermal_eval_root:
        thermal_eval_summary = _save_thermal_eval_artifacts(
            thermal_evaluator=thermal_evaluator,
            x_sample=sample_unprocessed[0],
            cond=cond_output_preprocessed,
            placement_json_path=placement_json_path,
            thermal_eval_root=thermal_eval_root,
        )
    t3 = time.time()

    hpwl_normalized, hpwl_rescaled = utils.hpwl_fast(sample_unprocessed[0], cond_output_preprocessed, normalized_hpwl=False)
    macro_hpwl_normalized, macro_hpwl_rescaled = utils.macro_hpwl(
        sample_unprocessed[0], cond_output_preprocessed, normalized_hpwl=False
    )
    legality = utils.check_legality_new(
        sample_unprocessed[0], x_in[0], cond_output_preprocessed, cond_output_preprocessed.is_ports, score=True
    )
    expanded_legality = utils.check_legality_new(
        sample_unprocessed[0],
        x_in[0],
        cond_once_expanded_preprocessed,
        cond_once_expanded_preprocessed.is_ports,
        score=True,
    )
    original_hpwl_normalized = utils.hpwl_fast(x_preprocessed, cond_output_preprocessed, normalized_hpwl=True)
    bbox_metric_values = utils.bbox_metrics(sample_unprocessed[0], cond_output_preprocessed, reference_x=x_preprocessed[0])

    if thermal_eval_summary is not None:
        thermal_metrics = {
            "thermal_max_k": float(thermal_eval_summary["thermal_max_c"]) + 273.15,
            "thermal_mean_k": float(thermal_eval_summary["thermal_mean_c"]) + 273.15,
            "thermal_max_c": float(thermal_eval_summary["thermal_max_c"]),
            "thermal_mean_c": float(thermal_eval_summary["thermal_mean_c"]),
        }
        if thermal_eval_summary.get("thermal_avg_head_c") is not None:
            thermal_metrics["thermal_avg_head_k"] = float(thermal_eval_summary["thermal_avg_head_c"]) + 273.15
            thermal_metrics["thermal_avg_head_c"] = float(thermal_eval_summary["thermal_avg_head_c"])
    else:
        thermal_metrics = thermal_evaluator(sample_unprocessed[0], cond_output_preprocessed)
    t4 = time.time()

    cond.to(original_device)

    all_metrics = {
        **thermal_metrics,
        "idx": idx,
        "hpwl_normalized": hpwl_normalized,
        "hpwl_rescaled": hpwl_rescaled,
        "macro_hpwl_normalized": macro_hpwl_normalized,
        "macro_hpwl_rescaled": macro_hpwl_rescaled,
        "legality_2": legality,
        "expanded_legality_2": expanded_legality,
        "original_hpwl_normalized": original_hpwl_normalized,
        "hpwl_ratio": hpwl_normalized / max(1e-12, original_hpwl_normalized),
        **bbox_metric_values,
        "model_time": t2 - t1,
        "generation_time": t3 - t0,
        "eval_time": t4 - t3,
        "model_vertices": cond_preprocessed.num_nodes,
        "model_edges": cond_preprocessed.num_edges,
    }
    if tap_context is not None:
        if tap_intp_size is not None:
            all_metrics["tap_intp_size"] = float(tap_intp_size)
        all_metrics["tap_avg_wirelength"] = float("nan") if tap_avg_wirelength is None else tap_avg_wirelength
        all_metrics["tap_hubump_mean"] = float(tap_context["hubump"].detach().cpu().mean().item())
        all_metrics["tap_hubump_max"] = float(tap_context["hubump"].detach().cpu().max().item())
    if thermal_eval_summary is not None:
        all_metrics["thermal_heatmap_png"] = thermal_eval_summary["heatmap_png"]
        all_metrics["thermal_summary_json"] = str(
            Path(thermal_eval_summary["heatmap_png"]).with_name(
                Path(thermal_eval_summary["heatmap_png"]).name.replace("_thermal_heatmap.png", "_thermal_summary.json")
            )
        )
    return _core_metrics(all_metrics), metrics_special, image, image_legalized

@hydra.main(version_base=None, config_path="configs", config_name="config_eval")
def main(cfg):
    # Preliminaries
    OmegaConf.set_struct(cfg, True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(cfg.seed)
    thermal_cfg = dict(cfg.get("thermal", {}) or {})

    # Prepare legalization function
    if cfg.legalization.mode in [None, "none", "None", ""]:
        legalize_fn = None
    elif cfg.legalization.mode in ["scheduled", "standard"]:
        def legalize_fn(x, cond):
            return legalization.legalize(
                x, 
                cond,
                thermal_cfg=thermal_cfg,
                **cfg.legalization,
                )
    elif cfg.legalization.mode == "opt":
        def legalize_fn(x, cond):
            return legalization.legalize_opt(
                x, 
                cond,
                **cfg.legalization,
                )
    # Prepare pre and post processing functions. Note that postprocess fns are applied in reverse order
    preprocess_fns = []
    postprocess_fns = []
    if cfg.cluster.is_cluster:
        def cluster_preprocess_fn(x, cond):
            cluster_cond, cluster_x = utils.cluster(cond, cfg.cluster.num_clusters, verbose=cfg.cluster.verbose, placements=x)
            return cluster_x, cluster_cond
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        preprocess_fns.append(cluster_preprocess_fn)
        postprocess_fns.append(cluster_postprocess_fn)
    elif cfg.cluster.cached_clusters:
        def cluster_postprocess_fn(x, cond):
            return utils.uncluster(cond, x, return_cond=True)
        postprocess_fns.append(cluster_postprocess_fn)
    if cfg.sc_halo != 1.0:
        def resize_standard_cells(x, cond):
            _, _, sc_mask = analysis_utils.get_masks(x, cond)
            is_resize = sc_mask.float()
            size_multiplier = (is_resize * cfg.sc_halo) + ((1-is_resize))
            cond.x = cond.x * size_multiplier.unsqueeze(dim=-1)
            return x, cond
        preprocess_fns.append(resize_standard_cells)
    if cfg.edge_dropout > 0.0: # used for debugging
        def edge_dropout(x, cond):
            x, cond = utils.edge_dropout(x, cond, cfg.edge_dropout)
            return x, cond
        preprocess_fns.append(edge_dropout)
    if cfg.macros_only:
        if cfg.cached_macros:
            postprocess_fns.append(utils.add_non_macros)
        else:
            preprocess_fns.append(utils.remove_non_macros)
            postprocess_fns.append(utils.add_non_macros)
    def preprocess_fn(x, cond):
        for preprocess_step in preprocess_fns:
            x, cond = preprocess_step(x, cond)
        return x, cond
    def postprocess_fn(x, cond):
        for i, postprocess_step in enumerate(reversed(postprocess_fns)):
            x, cond = postprocess_step(x, cond)    
        return x, cond

    # Preparing dataset
    train_set, val_set = utils.load_graph_data_with_config(cfg.task, train_data_limit = cfg.train_data_limit, val_data_limit = cfg.val_data_limit)
    sample_shape = val_set[0][0].shape
    dataloader = utils.GraphDataLoader(
        train_set, 
        val_set, 
        cfg.val_batch_size, 
        cfg.val_batch_size, 
        device,
        preprocess_fn = preprocess_fn,
        val_shuffle = False, # Don't shuffle validation set
        )
    with open_dict(cfg):
        if cfg.family in ["cond_diffusion", "continuous_diffusion", "flow_matching", "guided_diffusion", "skip_diffusion", "skip_guided_diffusion", "no_model"]:
            cfg.model.update({
                "num_classes": cfg.num_classes,
                "input_shape": tuple(sample_shape),
                "device": device,
            })
        else:
            raise NotImplementedError

    # Preparing model
    model_types = {
        "cond_diffusion": models.CondDiffusionModel,
        "continuous_diffusion": models.ContinuousDiffusionModel, 
        "flow_matching": ThermalGuidedFlowMatchingModel,
        "guided_diffusion": models.GuidedDiffusionModel,
        "skip_diffusion": models.SkipDiffusionModel,
        "skip_guided_diffusion": models.SkipGuidedDiffusionModel,
        "no_model": models.NoModel,
    }
    if cfg.implementation == "custom":
        if cfg.family == "flow_matching":
            model = model_types[cfg.family](**cfg.model, thermal_cfg=thermal_cfg).to(device)
        else:
            model = model_types[cfg.family](**cfg.model).to(device)
    else:
        raise NotImplementedError

    # Prepare logger
    num_params = sum([param.numel() for param in model.parameters()])
    with open_dict(cfg):  # for eval/debugging
        cfg.update({
            "num_params": num_params,
            "train_dataset": dataloader.get_train_size(),
            "val_dataset": dataloader.get_val_size(),
        })
    outputs = [
        common.logger.TerminalOutput(cfg.logger.filter),
    ]
    if cfg.logger.get("wandb", False):
        wandb_run_name = f"{cfg.task}.{cfg.method}.{cfg.seed}" if not cfg.param_sweep else None
        wandb_output = common.logger.WandBOutput(wandb_run_name, cfg)
        if cfg.param_sweep:
            with open_dict(cfg):  # for eval/debugging
                cfg.update({
                    "method": f"{cfg.method}.{wandb_output._wandb.run.name}",
                })
        else:
            print("WARNING: param_sweep set to true but wandb disabled. Continuing anyways...")
        outputs.append(wandb_output)
    step = common.Counter()
    logger = common.Logger(step, outputs)

    # Create log and output directories
    log_dir = utils.output_log_dir(cfg)
    sample_dir = os.path.join(log_dir, "samples")
    thermal_eval_dir = os.path.join(log_dir, "thermal_eval")
    checkpointer = common.Checkpointer(os.path.join(log_dir, "latest.ckpt"))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(thermal_eval_dir, exist_ok=True)
    print(f"saving eval outputs to: {log_dir}")

    # Output config used
    utils.save_cfg(cfg, os.path.join(log_dir, "config.yaml"))
    utils.write_summary_metrics({}, os.path.join(log_dir, "metrics_summary"))
    print(OmegaConf.to_yaml(cfg))

    # Load checkpoint if exists. Here we only load the model
    checkpointer.register({
        "model": model,
    })
    checkpointer.load(
        utils.resolve_checkpoint_path(cfg, cfg.from_checkpoint)
    )

    # Start training
    print(f"model has {num_params} params")
    print(f"==== Start Eval on Device: {device} ====")
    thermal_evaluator = ThermalEvaluator(thermal_cfg, device)
    report_guidance_enabled = _cfg_bool(thermal_cfg, "report_guidance_enabled", True)

    def report_eval_function(samples, x_val, cond_val):
        sample_metrics = utils.eval_samples(samples, x_val, cond_val)
        for idx, sample_metric in enumerate(sample_metrics):
            sample_metric.update(thermal_evaluator(samples[idx], cond_val))
        return sample_metrics

    if cfg.eval_samples > 0:
        print("generating evaluation report")
        t1 = time.time()
        old_thermal_guidance_weight = getattr(model, "thermal_guidance_weight", None)
        old_thermal_guidance_steps = getattr(model, "thermal_guidance_steps", None)
        old_heat_repulsion_guidance_weight = getattr(model, "heat_repulsion_guidance_weight", None)
        old_heat_repulsion_schedule = {}
        heat_repulsion_schedule_keys = (
            "heat_repulsion_initial_weight",
            "heat_repulsion_final_weight",
        )
        guidance_schedule = getattr(model, "guidance_schedule", None)
        if not report_guidance_enabled and old_thermal_guidance_weight is not None:
            print("thermal guidance disabled for evaluation report; thermal metrics are still computed")
            model.thermal_guidance_weight = 0.0
            model.thermal_guidance_steps = 0
            if old_heat_repulsion_guidance_weight is not None:
                model.heat_repulsion_guidance_weight = 0.0
            if guidance_schedule is not None and hasattr(guidance_schedule, "get"):
                for key in heat_repulsion_schedule_keys:
                    old_heat_repulsion_schedule[key] = guidance_schedule.get(key, None)
                    try:
                        with open_dict(guidance_schedule):
                            guidance_schedule[key] = 0.0
                    except Exception:
                        guidance_schedule[key] = 0.0
        try:
            utils.generate_report(
                cfg.eval_samples, 
                dataloader, 
                model, 
                logger, 
                policy = cfg.eval_policy_algorithm, 
                intermediate_every = cfg.show_intermediate_every,
                eval_function = report_eval_function,
                )
        finally:
            if old_thermal_guidance_weight is not None:
                model.thermal_guidance_weight = old_thermal_guidance_weight
            if old_thermal_guidance_steps is not None:
                model.thermal_guidance_steps = old_thermal_guidance_steps
            if old_heat_repulsion_guidance_weight is not None:
                model.heat_repulsion_guidance_weight = old_heat_repulsion_guidance_weight
            if guidance_schedule is not None and old_heat_repulsion_schedule:
                for key, value in old_heat_repulsion_schedule.items():
                    if value is not None:
                        try:
                            with open_dict(guidance_schedule):
                                guidance_schedule[key] = value
                        except Exception:
                            guidance_schedule[key] = value
        logger.write()
        t2 = time.time()
        print(f"generated report in {t2-t1:.3f} sec")

    # output eval samples
    t3 = time.time()
    print("generating output samples")
    output_metrics = {}
    log_metrics = common.Metrics()
    intermediate_export_cfg = OmegaConf.select(cfg, "intermediate_export")
    intermediate_export = (
        OmegaConf.to_container(intermediate_export_cfg, resolve=True)
        if intermediate_export_cfg is not None
        else None
    )
    num_output_samples = min(int(cfg.num_output_samples), len(val_set))
    if num_output_samples < int(cfg.num_output_samples):
        print(
            f"WARNING: requested {cfg.num_output_samples} output samples, "
            f"but val set only has {len(val_set)}. Generating {num_output_samples} samples."
        )
    for i in range(num_output_samples):
        x, cond = val_set[i]
        metrics, metrics_special, image, image_legalized = save_outputs_with_thermal(
            x, 
            cond, 
            model, 
            save_folder=sample_dir, 
            thermal_evaluator=thermal_evaluator,
            thermal_eval_root=thermal_eval_dir,
            output_number_offset=0, 
            policy=cfg.eval_policy_algorithm,
            policy_kwargs=cfg.eval_policy,
            preprocess_fn=preprocess_fn,
            postprocess_fn=postprocess_fn,
            legalization_fn=legalize_fn,
            intermediate_export=intermediate_export,
        )
        print(f"Finished sample {i+1} of {num_output_samples} \t {metrics}")
        t5 = time.time()
        logger.add({
            "reverse_samples": {
                **metrics,
                **metrics_special,
                "image": utils.logging_image(image_legalized, logger),
                "image_raw": utils.logging_image(image, logger),
                "time_elapsed": t5-t3,
            }
        })
        # update metrics
        for k, v in metrics.items():
            if k in output_metrics:
                output_metrics[k].append(v)
            else:
                output_metrics[k] = [v]
        log_metrics.add(metrics)
    if output_metrics:
        utils.dict_to_csv(output_metrics, os.path.join(log_dir,"metrics.csv"))
        export_wirelength_bbox_results(log_dir)
        if cfg.logger.get("wandb", False):
            for plot_keys in cfg.scatter_plots:
                x_name = plot_keys[0]
                y_name = plot_keys[1]
                if x_name in output_metrics and y_name in output_metrics:
                    scatter_plot = utils.plot_scatter(output_metrics[x_name], output_metrics[y_name], x_title=x_name, y_title=y_name)
                    logger.add({f"{x_name}_vs_{y_name}": scatter_plot})
        summary_metrics = log_metrics.result()
        sweep_metrics = cost(output_metrics)
        utils.write_summary_metrics(
            output_metrics,
            os.path.join(log_dir, "metrics_summary"),
            extra_metrics={f"sweep/{k}": v for k, v in sweep_metrics.items()},
        )
        logger.add(summary_metrics)
        logger.add(sweep_metrics, prefix = "sweep")
        logger.write()

if __name__=="__main__":
    main()
