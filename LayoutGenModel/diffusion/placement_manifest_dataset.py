"""Lazy Flow Matching dataset backed by placement JSON split manifests."""

from __future__ import annotations

import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from torch_geometric.data import Data


def _resolve_repo_path(value, config_dir):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (Path(config_dir) / path, repo_root / path, Path.cwd() / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[1].resolve())


class PlacementManifestDataset:
    """Convert placement records to normalized PyG graphs on demand.

    The source files contain 5,000 systems each. A small LRU cache keeps parsed
    chunks in memory, while ``sample_index`` reuses one chunk for several random
    training samples to avoid making JSON parsing the training bottleneck.
    """

    def __init__(
        self,
        placement_dir,
        manifest_dir,
        split,
        limit=None,
        seed=42,
        cache_chunks=2,
        samples_per_chunk=256,
        canvas_padding_mm=1.0,
    ):
        self.placement_dir = Path(placement_dir)
        self.manifest_dir = Path(manifest_dir)
        self.split = str(split)
        ids = [
            int(line)
            for line in (self.manifest_dir / f"{self.split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit not in (None, "none", "None"):
            ids = ids[: int(limit)]
        if not ids:
            raise ValueError(f"placement split is empty: {self.split}")
        self.ids = ids
        self.cache_chunks = max(1, int(cache_chunks))
        self.samples_per_chunk = max(1, int(samples_per_chunk))
        self.canvas_padding_mm = float(canvas_padding_mm)
        self._cache = OrderedDict()
        self._rng = random.Random(int(seed))
        self._sample_chunk = None
        self._sample_chunk_remaining = 0
        self._indices_by_chunk = defaultdict(list)
        for index, system_id in enumerate(self.ids):
            self._indices_by_chunk[self._chunk_index(system_id)].append(index)
        self._chunks = sorted(self._indices_by_chunk)

    @staticmethod
    def _chunk_index(system_id):
        return (int(system_id) - 1) // 5000 + 1

    def __len__(self):
        return len(self.ids)

    def sample_index(self):
        if self._sample_chunk_remaining <= 0 or self._sample_chunk is None:
            self._sample_chunk = self._rng.choice(self._chunks)
            self._sample_chunk_remaining = self.samples_per_chunk
        self._sample_chunk_remaining -= 1
        return self._rng.choice(self._indices_by_chunk[self._sample_chunk])

    def _load_chunk(self, chunk_index):
        if chunk_index in self._cache:
            self._cache.move_to_end(chunk_index)
            return self._cache[chunk_index]
        path = self.placement_dir / f"chiplet_dataset_{chunk_index}.json"
        with path.open("r", encoding="utf-8") as handle:
            systems = json.load(handle)
        self._cache[chunk_index] = systems
        self._cache.move_to_end(chunk_index)
        while len(self._cache) > self.cache_chunks:
            self._cache.popitem(last=False)
        return systems

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = int(index.view(-1)[0].item())
        system_id = self.ids[int(index)]
        systems = self._load_chunk(self._chunk_index(system_id))
        record = systems[f"system_{system_id}"]
        return self._to_flow_graph(system_id, record)

    def _to_flow_graph(self, system_id, record):
        chiplets = record["chiplets"]
        node_count = len(chiplets)
        if node_count == 0:
            raise ValueError(f"system_{system_id} contains no chiplets")
        names = [str(chiplet.get("name", index)) for index, chiplet in enumerate(chiplets)]
        name_to_index = {name: index for index, name in enumerate(names)}

        sizes = torch.tensor(
            [[float(chiplet["width"]), float(chiplet["height"])] for chiplet in chiplets],
            dtype=torch.float32,
        )
        lower = torch.tensor(
            [[float(chiplet["x-position"]), float(chiplet["y-position"])] for chiplet in chiplets],
            dtype=torch.float32,
        )
        hubump = torch.tensor(
            [float(chiplet.get("hubump", 0.0)) for chiplet in chiplets],
            dtype=torch.float32,
        )
        powers = torch.tensor(
            [float(chiplet.get("power", 0.0)) for chiplet in chiplets],
            dtype=torch.float32,
        )

        outer_lower = lower - hubump.view(-1, 1)
        outer_upper = lower + sizes + hubump.view(-1, 1)
        bbox_lower = outer_lower.amin(dim=0)
        spans = outer_upper.amax(dim=0) - bbox_lower
        side = spans.amax() + self.canvas_padding_mm
        shift = (side - spans) / 2.0 - bbox_lower
        lower = lower + shift.view(1, 2)

        pairs = []
        attrs = []
        weights = []
        for connection in record.get("connections", []):
            source_name = str(connection["node1"])
            target_name = str(connection["node2"])
            if source_name not in name_to_index or target_name not in name_to_index:
                continue
            source = name_to_index[source_name]
            target = name_to_index[target_name]
            if source == target:
                continue
            wire_count = float(connection.get("wireCount", 1.0))
            bump_width = float(connection.get("EMIB_bump_width", 0.5))
            edge_type = float(connection.get("edge_type", 0.0))
            metadata = torch.tensor(
                [wire_count / 1024.0, bump_width, edge_type, 1.0],
                dtype=torch.float32,
            )
            # The original Flow Matching graph builder places unspecified pins
            # at each chiplet's lower-left corner. preprocess_graph then turns
            # these into offsets relative to the chiplet centre.
            attr = torch.cat((torch.zeros(4, dtype=torch.float32), metadata))
            pairs.extend(((source, target), (target, source)))
            attrs.extend((attr.clone(), attr.clone()))
            weights.extend((wire_count, wire_count))
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(attrs).to(dtype=torch.float32)
            edge_weight = torch.tensor(weights, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 8), dtype=torch.float32)
            edge_weight = torch.empty((0,), dtype=torch.float32)

        cond = Data(
            x=sizes.clone(),
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            is_ports=torch.zeros(node_count, dtype=torch.bool),
            is_macros=torch.ones(node_count, dtype=torch.bool),
            node_power=powers,
            chip_size=torch.tensor([0.0, 0.0, float(side), float(side)], dtype=torch.float32),
            tap_hubump=hubump,
            tap_source_chiplet_sizes=sizes.clone(),
            file_idx=int(system_id),
            system_id=int(system_id),
            benchmark_name=f"system_{system_id}",
        )

        # Equivalent to utils.preprocess_graph for centre-pin pairwise edges.
        scale = torch.tensor([float(side), float(side)], dtype=torch.float32)
        cond.x = 2.0 * cond.x / scale.view(1, 2)
        cond.edge_attr[:, :2] = 2.0 * cond.edge_attr[:, :2] / scale.view(1, 2)
        cond.edge_attr[:, 2:4] = 2.0 * cond.edge_attr[:, 2:4] / scale.view(1, 2)
        placement = 2.0 * lower / scale.view(1, 2) - 1.0
        placement = placement + cond.x / 2.0
        if cond.edge_index.numel():
            cond.edge_attr[:, :2] -= cond.x[cond.edge_index[0]] / 2.0
            cond.edge_attr[:, 2:4] -= cond.x[cond.edge_index[1]] / 2.0
        return placement, cond


def load_placement_manifest_datasets(config, config_dir, train_limit=None, val_limit=None):
    manifest_dir = _resolve_repo_path(config.placement_manifest, config_dir)
    placement_dir = _resolve_repo_path(config.placement_dir, config_dir)
    common = {
        "placement_dir": placement_dir,
        "manifest_dir": manifest_dir,
        "seed": int(config.get("split_seed", 42)),
        "cache_chunks": int(config.get("cache_chunks", 2)),
        "samples_per_chunk": int(config.get("samples_per_chunk", 256)),
        "canvas_padding_mm": float(config.get("canvas_padding_mm", 1.0)),
    }
    train = PlacementManifestDataset(split="train", limit=train_limit, **common)
    val = PlacementManifestDataset(split="val", limit=val_limit, **common)
    return train, val
