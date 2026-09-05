#!/usr/bin/env python3
"""Run ThermalGuidanceHRNet on one placement JSON and save a heat map."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from thermalmodel.HRNet import ThermalGuidanceHRNet
from thermalmodel.draw_thermal_fig import plot_thermal_grid_overlay


def _device(prefer: str) -> torch.device:
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _chiplets(data: dict[str, Any], *, require_position: bool = True) -> list[dict[str, Any]]:
    chiplets = list(data.get("chiplets") or [])
    if not chiplets:
        raise ValueError("placement JSON has no chiplets")
    required = ("width", "height")
    if require_position:
        required = ("x-position", "y-position", "width", "height")
    for idx, ch in enumerate(chiplets):
        missing = [k for k in required if k not in ch]
        if missing:
            raise ValueError(f"chiplet {idx} missing fields: {missing}")
    return chiplets


def _infer_source_json(placement_path: Path) -> Path | None:
    match = re.search(r"Case(\d+)", placement_path.stem)
    if not match:
        return None
    candidate = Path(_PROJECT_ROOT) / "benchmark" / "ATPlace_json" / f"Case{match.group(1)}.json"
    return candidate if candidate.is_file() else None


def _with_source_dimensions(
    placement_chiplets: list[dict[str, Any]],
    source_json: str,
) -> list[dict[str, Any]]:
    source_path = Path(source_json)
    source_data = json.loads(source_path.read_text(encoding="utf-8"))
    source_chiplets = _chiplets(source_data, require_position=False)
    if len(source_chiplets) != len(placement_chiplets):
        raise ValueError(
            f"source JSON chiplet count {len(source_chiplets)} != placement count {len(placement_chiplets)}"
        )

    merged = []
    for idx, (placed, source) in enumerate(zip(placement_chiplets, source_chiplets)):
        item = dict(placed)
        item["width"] = float(source["width"])
        item["height"] = float(source["height"])
        item["power"] = float(source.get("power", placed.get("power", 1.0)))
        item["source_name"] = str(source.get("name", f"C{idx}"))
        merged.append(item)
    return merged


def _stats_from_ckpt(ckpt: dict[str, Any]) -> dict[str, float]:
    stats = ckpt.get("stats")
    if not isinstance(stats, dict):
        raise ValueError("checkpoint does not contain normalization stats")
    required = ("power_min", "power_max", "total_power_min", "total_power_max", "temp_min", "temp_max")
    missing = [k for k in required if k not in stats]
    if missing:
        raise ValueError(f"checkpoint stats missing fields: {missing}")
    return {k: float(stats[k]) for k in required}


def _rects(chiplets: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xywh = []
    powers = []
    for ch in chiplets:
        xywh.append(
            [
                float(ch["x-position"]),
                float(ch["y-position"]),
                float(ch["width"]),
                float(ch["height"]),
            ]
        )
        powers.append(float(ch.get("power", 1.0)))
    return np.asarray(xywh, dtype=np.float32), np.asarray(powers, dtype=np.float32), np.asarray(powers).sum()


def _rasterize(
    chiplets: list[dict[str, Any]],
    stats: dict[str, float],
    grid_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    xywh, powers, total_power = _rects(chiplets)
    x0 = float(np.min(xywh[:, 0]))
    y0 = float(np.min(xywh[:, 1]))
    x1 = float(np.max(xywh[:, 0] + xywh[:, 2]))
    y1 = float(np.max(xywh[:, 1] + xywh[:, 3]))
    span_x = max(x1 - x0, 1e-12)
    span_y = max(y1 - y0, 1e-12)

    power_raw = np.zeros((grid_size, grid_size), dtype=np.float32)
    layout = np.zeros((grid_size, grid_size), dtype=np.float32)

    for (x, y, w, h), power in zip(xywh, powers):
        gx0 = int(np.floor(((float(x) - x0) / span_x) * grid_size))
        gx1 = int(np.ceil(((float(x + w) - x0) / span_x) * grid_size))
        gy0 = int(np.floor(((float(y) - y0) / span_y) * grid_size))
        gy1 = int(np.ceil(((float(y + h) - y0) / span_y) * grid_size))
        gx0, gx1 = max(0, gx0), min(grid_size, gx1)
        gy0, gy1 = max(0, gy0), min(grid_size, gy1)
        if gx1 <= gx0 or gy1 <= gy0:
            continue
        area_cells = float((gx1 - gx0) * (gy1 - gy0))
        layout[gy0:gy1, gx0:gx1] = 1.0
        power_raw[gy0:gy1, gx0:gx1] += float(power) / area_cells

    power = (power_raw - stats["power_min"]) / max(stats["power_max"] - stats["power_min"], 1e-6)
    power = np.clip(power, 0.0, 1.0).astype(np.float32)

    # Training total-power scalar is normalized after division by grid area.
    total_power_density = float(total_power) / float(grid_size * grid_size)
    tp_min = stats["total_power_min"] / float(grid_size * grid_size)
    tp_max = stats["total_power_max"] / float(grid_size * grid_size)
    total_power01 = np.clip((total_power_density - tp_min) / max(tp_max - tp_min, 1e-12), 0.0, 1.0)

    meta = {
        "bbox_x0": x0,
        "bbox_y0": y0,
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_width": span_x,
        "bbox_height": span_y,
        "total_power": float(total_power),
        "total_power01": float(total_power01),
    }
    return (
        torch.from_numpy(power).view(1, 1, grid_size, grid_size),
        torch.from_numpy(layout).view(1, 1, grid_size, grid_size),
        torch.tensor([[float(total_power01)]], dtype=torch.float32),
        meta,
    )


def _write_flp(chiplets: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for idx, ch in enumerate(chiplets):
        name = str(ch.get("name", f"C{idx}"))
        w = float(ch["width"]) / 1000.0
        h = float(ch["height"]) / 1000.0
        x = float(ch["x-position"]) / 1000.0
        y = float(ch["y-position"]) / 1000.0
        lines.append(f"{name} {w:.12g} {h:.12g} {x:.12g} {y:.12g}\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement_json", required=True)
    ap.add_argument("--ckpt", required=True, help="path to a trained ThermalGuidanceHRNet checkpoint")
    ap.add_argument("--source_json", default="", help="Original benchmark JSON with physical chiplet width/height")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()

    placement_path = Path(args.placement_json).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else placement_path.parent / "thermal_eval" / placement_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(placement_path.read_text(encoding="utf-8"))
    chiplets = _chiplets(data)
    source_json = args.source_json
    if not source_json:
        inferred = _infer_source_json(placement_path)
        source_json = str(inferred) if inferred is not None else ""
    if source_json:
        chiplets = _with_source_dimensions(chiplets, source_json)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    stats = _stats_from_ckpt(ckpt)
    grid_size = int(ckpt.get("grid_size", 128))
    device = _device(args.device)

    model = ThermalGuidanceHRNet(
        base=int(ckpt.get("base", 64)),
        stages=int(ckpt.get("stages", 4)),
        blocks_per_stage=int(ckpt.get("blocks_per_stage", 2)),
        expand_ratio=int(ckpt.get("expand_ratio", 2)),
        mean_calib=bool(ckpt.get("mean_calib", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    power, layout, total_power, meta = _rasterize(chiplets, stats, grid_size=grid_size)
    with torch.no_grad():
        pred01, pred_avg01 = model(power.to(device), layout.to(device), total_power.to(device))
    temp_c = pred01.detach().cpu()[0, 0] * (stats["temp_max"] - stats["temp_min"]) + stats["temp_min"]
    avg_head_c = pred_avg01.detach().cpu().view(-1)[0] * (stats["temp_max"] - stats["temp_min"]) + stats["temp_min"]

    flp_path = out_dir / f"{placement_path.stem}.flp"
    heatmap_path = out_dir / f"{placement_path.stem}_thermal_heatmap.png"
    npy_path = out_dir / f"{placement_path.stem}_temperature_c.npy"
    summary_path = out_dir / f"{placement_path.stem}_thermal_summary.json"

    _write_flp(chiplets, flp_path)
    np.save(npy_path, temp_c.numpy())
    plot_thermal_grid_overlay(
        str(flp_path),
        temp_c,
        str(heatmap_path),
        title=f"{placement_path.stem} thermal prediction",
        vmin=float(temp_c.min().item()),
        vmax=float(temp_c.max().item()),
    )

    summary = {
        "placement_json": str(placement_path),
        "ckpt": str(Path(args.ckpt).resolve()),
        "source_json": str(Path(source_json).resolve()) if source_json else "",
        "device": str(device),
        "grid_size_input": grid_size,
        "grid_size_output": list(temp_c.shape),
        "temperature_unit": "C",
        "thermal_max_c": float(temp_c.max().item()),
        "thermal_mean_c": float(temp_c.mean().item()),
        "thermal_min_c": float(temp_c.min().item()),
        "thermal_avg_head_c": float(avg_head_c.item()),
        **meta,
        "heatmap_png": str(heatmap_path),
        "temperature_npy": str(npy_path),
        "overlay_flp": str(flp_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
