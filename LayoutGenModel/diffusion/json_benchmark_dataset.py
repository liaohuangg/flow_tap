"""Load standalone chiplet benchmark JSON files for Flow Matching inference."""

from __future__ import annotations

import json
import itertools
import math
import os
from pathlib import Path

import torch
from torch_geometric.data import Data


def _parse_area_ratio(value):
    text = str(value).strip()
    if text.endswith("%"):
        ratio = float(text[:-1]) / 100.0
    else:
        ratio = float(text)
        if ratio > 1.0:
            ratio /= 100.0
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"chiplet area ratio must be in (0, 1], got {value!r}")
    return ratio


def _resolve_path(value, config_dir):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (Path(config_dir) / path, repo_root / path, Path.cwd() / path)
    return next((item.resolve() for item in candidates if item.exists()), candidates[1].resolve())


def _shelf_place(footprints, initial_side, padding):
    side = float(initial_side)
    margin = float(padding) / 2.0
    indices = list(range(len(footprints)))
    candidate_orders = [
        sorted(indices, key=lambda i: (-float(footprints[i, 1]), -float(footprints[i, 0]))),
        sorted(indices, key=lambda i: (-float(footprints[i, 0]), -float(footprints[i, 1]))),
        sorted(indices, key=lambda i: -float(footprints[i, 0] * footprints[i, 1])),
        sorted(indices, key=lambda i: -float(max(footprints[i, 0], footprints[i, 1]))),
    ]
    if len(indices) <= 8:
        candidate_orders.extend(itertools.permutations(indices))

    for order in candidate_orders:
        placed = torch.zeros_like(footprints)
        cursor_x = margin
        cursor_y = margin
        row_height = 0.0
        fits = True
        for index in order:
            width, height = (float(value) for value in footprints[index])
            if cursor_x > margin and cursor_x + width > side - margin:
                cursor_x = margin
                cursor_y += row_height
                row_height = 0.0
            if cursor_y + height > side - margin:
                fits = False
                break
            placed[index] = torch.tensor([cursor_x, cursor_y], dtype=footprints.dtype)
            cursor_x += width
            row_height = max(row_height, height)
        if fits:
            return placed, side
    raise ValueError(
        f"hubump-expanded footprints do not fit the requested {side:.6f} square canvas "
        f"with {padding:.6f} mm total edge padding"
    )


class JsonBenchmarkDataset:
    def __init__(self, source_dir, target_utilization=0.60, canvas_padding_mm=1.0):
        self.source_dir = Path(source_dir)
        self.files = sorted(self.source_dir.glob("*.json"))
        if not self.files:
            raise ValueError(f"no benchmark JSON files found in {self.source_dir}")
        self.target_utilization = _parse_area_ratio(target_utilization)
        self.canvas_padding_mm = float(canvas_padding_mm)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[int(index)]
        record = json.loads(path.read_text(encoding="utf-8"))
        chiplets = record["chiplets"]
        names = [str(chiplet.get("name", i)) for i, chiplet in enumerate(chiplets)]
        name_to_index = {name: i for i, name in enumerate(names)}
        sizes = torch.tensor([[float(c["width"]), float(c["height"])] for c in chiplets], dtype=torch.float32)
        hubump = torch.tensor([float(c.get("hubump", 0.0)) for c in chiplets], dtype=torch.float32)
        powers = torch.tensor([float(c.get("power", 0.0)) for c in chiplets], dtype=torch.float32)
        footprints = sizes + 2.0 * hubump.view(-1, 1)
        # Define the external square canvas from chiplet body area.  Hubump
        # remains part of the occupied footprint used by legality/reference
        # placement, but it is not chiplet body area for the utilization ratio.
        chiplet_area = float((sizes[:, 0] * sizes[:, 1]).sum())
        side = math.sqrt(chiplet_area / max(self.target_utilization, 1e-6))
        if float(footprints.max()) + self.canvas_padding_mm > side:
            raise ValueError(
                f"{path.name}: the requested chiplet utilization cannot fit the largest "
                f"hubump-expanded footprint in a {side:.6f} square canvas"
            )
        footprint_lower, side = _shelf_place(footprints, side, self.canvas_padding_mm)
        body_lower = footprint_lower + hubump.view(-1, 1)

        pairs, attrs, weights = [], [], []
        for connection in record.get("connections", []):
            source = name_to_index[str(connection["node1"])]
            target = name_to_index[str(connection["node2"])]
            if source == target:
                continue
            wire_count = float(connection.get("wireCount", 1.0))
            bump_width = float(connection.get("EMIB_bump_width", 0.5))
            metadata = torch.tensor([wire_count / 1024.0, bump_width, 0.0, 1.0], dtype=torch.float32)
            attr = torch.cat((torch.zeros(4, dtype=torch.float32), metadata))
            pairs.extend(((source, target), (target, source)))
            attrs.extend((attr.clone(), attr.clone()))
            weights.extend((wire_count, wire_count))
        edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous() if pairs else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.stack(attrs) if attrs else torch.empty((0, 8), dtype=torch.float32)
        edge_weight = torch.tensor(weights, dtype=torch.float32)

        scale = torch.tensor([side, side], dtype=torch.float32)
        cond = Data(
            x=2.0 * sizes / scale,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            is_ports=torch.zeros(len(chiplets), dtype=torch.bool),
            is_macros=torch.ones(len(chiplets), dtype=torch.bool),
            node_power=powers,
            chip_size=torch.tensor([0.0, 0.0, side, side], dtype=torch.float32),
            tap_hubump=hubump,
            tap_source_chiplet_sizes=sizes.clone(),
            file_idx=int(index),
            benchmark_name=str(record.get("case", path.stem)),
            source_json_path=str(path.resolve()),
        )
        if edge_index.numel():
            cond.edge_attr[:, :2] = -cond.x[edge_index[0]] / 2.0
            cond.edge_attr[:, 2:4] = -cond.x[edge_index[1]] / 2.0
        placement = 2.0 * (body_lower + sizes / 2.0) / scale - 1.0
        return placement, cond


def load_json_benchmark_datasets(config, config_dir, train_limit=None, val_limit=None):
    source_dir = _resolve_path(config.benchmark_json_dir, config_dir)
    target_utilization = os.environ.get(
        "FLOW_TAP_CHIPLET_AREA_RATIO",
        config.get("target_utilization", 0.60),
    )
    dataset = JsonBenchmarkDataset(
        source_dir,
        target_utilization=target_utilization,
        canvas_padding_mm=float(config.get("canvas_padding_mm", 1.0)),
    )
    if val_limit not in (None, "none", "None"):
        from torch.utils.data import Subset
        dataset = Subset(dataset, range(min(int(val_limit), len(dataset))))
    return [], dataset
