"""Differentiable adapter from Flow Matching placements to WirelengthGNN.

Flow Matching stores normalized chiplet centre coordinates, while WirelengthGNN
was trained from physical bottom-left coordinates and derived congestion
features.  This module is the single conversion boundary between those schemas.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent
_LAYOUT_ROOT = _THIS_DIR.parent
_REPO_ROOT = _LAYOUT_ROOT.parent


def _resolve_path(value, default=None):
    value = default if value in (None, "", "none", "None") else value
    if value in (None, "", "none", "None"):
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, _LAYOUT_ROOT / path, _REPO_ROOT / path)
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[1].resolve())


def _cfg_dict(value):
    if value is None:
        return {}
    return dict(value)


def _checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _unique_undirected_edges(edge_index):
    """Return one edge id for each unordered pair, independent of edge order."""
    keep = []
    seen = set()
    for edge_id, (src, dst) in enumerate(edge_index.detach().cpu().t().tolist()):
        key = (src, dst) if src <= dst else (dst, src)
        if src != dst and key not in seen:
            seen.add(key)
            keep.append(edge_id)
    return torch.as_tensor(keep, dtype=torch.long, device=edge_index.device)


class WirelengthSurrogate:
    """Frozen WirelengthGNN with a differentiable Flow Matching input adapter."""

    def __init__(self, cfg, device):
        from wirelengthmodel.dataloader import Normalizer, fit_normalizer_from_placement_files
        from wirelengthmodel.wlmodel import WirelengthGNN

        self.cfg = _cfg_dict(cfg)
        self.device = torch.device(device)
        checkpoint_path = _resolve_path(
            self.cfg.get("ckpt"),
            _REPO_ROOT / "wirelengthmodel" / "checkpoint" / "best_wlmodel_total_60k.pt",
        )
        if checkpoint_path is None or not checkpoint_path.exists():
            raise FileNotFoundError(f"wirelength checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        model_cfg = {
            "node_dim": 18,
            "edge_dim": 2,
            "global_dim": 4,
            "cong_dim": 8,
            "hidden": 256,
            "num_layers": 6,
            "heads": 4,
            "dropout": 0.0,
            "use_residual": True,
            "use_global": True,
        }
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
            model_cfg.update(checkpoint["model_config"])
        model_cfg.update(_cfg_dict(self.cfg.get("model")))
        self.model = WirelengthGNN(**model_cfg).to(self.device)
        self.model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        normalizer_value = checkpoint.get("normalizer") if isinstance(checkpoint, dict) else None
        normalizer_path = _resolve_path(self.cfg.get("normalizer"))
        if normalizer_value is None and normalizer_path is not None and normalizer_path.exists():
            if normalizer_path.suffix.lower() == ".json":
                normalizer_value = json.loads(normalizer_path.read_text(encoding="utf-8"))
            else:
                loaded = torch.load(str(normalizer_path), map_location="cpu")
                normalizer_value = loaded.get("normalizer", loaded) if isinstance(loaded, dict) else loaded

        if normalizer_value is None:
            if not bool(self.cfg.get("auto_fit_normalizer", False)):
                raise ValueError(
                    "legacy wirelength checkpoint has no normalizer; provide wirelength.normalizer "
                    "or enable wirelength.auto_fit_normalizer"
                )
            file_indices = self.cfg.get("normalizer_files", list(range(69, 81)))
            allow_partial = bool(self.cfg.get("allow_partial_normalizer", False))
            placement_dir = _resolve_path(
                self.cfg.get("placement_dir"),
                _REPO_ROOT / "Dataset" / "dataset" / "placement_dataset" / "placement_dataset_tw",
            )
            warnings.warn(
                "rebuilding the legacy wirelength normalizer from placement JSON files; "
                "package it with the checkpoint for reproducible and faster startup",
                RuntimeWarning,
            )
            normalizer = fit_normalizer_from_placement_files(
                placement_dir=placement_dir,
                file_indices=file_indices,
                seed=int(self.cfg.get("normalizer_seed", 42)),
                train_ratio=float(self.cfg.get("normalizer_train_ratio", 0.8)),
                allow_partial=allow_partial,
            )
            cache_path = _resolve_path(self.cfg.get("normalizer_cache"))
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(normalizer.to_dict(), indent=2), encoding="utf-8")
        elif isinstance(normalizer_value, Normalizer):
            normalizer = normalizer_value
        else:
            if bool(normalizer_value.get("approximate", False)):
                warnings.warn(
                    "using a compatibility wirelength normalizer reconstructed from a subset of the "
                    "legacy training distribution; replace it with the original packaged normalizer when available",
                    RuntimeWarning,
                )
            normalizer = Normalizer.from_dict(normalizer_value)

        self.node_mean = normalizer.node_mean.to(self.device)
        self.node_std = normalizer.node_std.to(self.device)
        self.edge_mean = normalizer.edge_mean.to(self.device)
        self.edge_std = normalizer.edge_std.to(self.device)
        if self.node_mean.numel() != 18 or self.edge_mean.numel() != 2:
            raise ValueError(
                f"wirelength normalizer schema mismatch: node={self.node_mean.numel()}, "
                f"edge={self.edge_mean.numel()}"
            )
        self.objective = str(self.cfg.get("objective", "log_total"))
        self.preferred_side_temperature = float(self.cfg.get("preferred_side_temperature", 0.25))

    @staticmethod
    def _canvas(cond, device, dtype):
        if "chip_size" not in cond:
            return torch.ones(2, device=device, dtype=dtype), torch.zeros(2, device=device, dtype=dtype)
        value = torch.as_tensor(cond.chip_size, device=device, dtype=dtype).view(-1)
        if value.numel() == 4:
            return (value[2:] - value[:2]).clamp_min(1e-6), value[:2]
        return value[:2].clamp_min(1e-6), torch.zeros(2, device=device, dtype=dtype)

    @staticmethod
    def _clumps(node_geom):
        x, y, w, h, hubump = node_geom.unbind(dim=-1)
        cx, cy = x + w / 2.0, y + h / 2.0
        return torch.stack(
            (
                torch.stack((x - hubump / 2.0, cy), dim=-1),
                torch.stack((cx, y + h + hubump / 2.0), dim=-1),
                torch.stack((x + w + hubump / 2.0, cy), dim=-1),
                torch.stack((cx, y - hubump / 2.0), dim=-1),
            ),
            dim=-2,
        )

    @staticmethod
    def _derive_hubump(width, height, demand):
        # Same discrete 45 um search used by TAP-2.5D preprocessing.
        values = []
        for w, h, required in zip(width.detach().cpu(), height.detach().cpu(), demand.detach().cpu()):
            rows = 1
            ring = 0.045
            while ((float(w) + float(h)) * 2.0 * ring + 4.0 * ring * ring) / (0.045 ** 2) < float(required):
                rows += 1
                ring = 0.045 * rows
                if rows > 1000:
                    raise RuntimeError("infeasible microbump demand while building wirelength features")
            values.append(ring)
        return torch.tensor(values, device=width.device, dtype=width.dtype)

    @staticmethod
    def _capacities(width, height, hubump):
        rows = torch.floor(hubump / 0.045).clamp_min(0.0)
        vertical = rows * torch.floor((height + hubump) / 0.045)
        horizontal = rows * torch.floor((width + hubump) / 0.045)
        side = torch.stack((vertical, horizontal, vertical, horizontal), dim=-1)
        return side, side.sum(dim=-1)

    def _topology(self, cond, device, dtype):
        edge_index_all = cond.edge_index.to(device=device, dtype=torch.long)
        edge_ids = _unique_undirected_edges(edge_index_all)
        if edge_ids.numel() == 0:
            raise ValueError("wirelength surrogate requires at least one chiplet connection")
        pairs = edge_index_all[:, edge_ids]
        if "edge_weight" not in cond:
            raise ValueError("wirelength surrogate requires cond.edge_weight with original wireCount values")
        weights = cond.edge_weight[edge_ids].to(device=device, dtype=dtype).view(-1)
        directed = torch.cat((pairs, pairs.flip(0)), dim=1)
        directed_weights = torch.cat((weights, weights), dim=0)
        return pairs, weights, directed, directed_weights

    def _build_batch(self, placement, cond, smooth_preferred=False):
        if placement.dim() == 2:
            placement = placement.unsqueeze(0)
        batch_size, num_nodes, _ = placement.shape
        device, dtype = placement.device, placement.dtype
        canvas, origin = self._canvas(cond, device, dtype)

        size_norm = cond.tap_original_x if "tap_original_x" in cond else cond.x
        size = size_norm[:, :2].to(device=device, dtype=dtype) * canvas.view(1, 2) / 2.0
        center = (placement[..., :2] + 1.0) * canvas.view(1, 1, 2) / 2.0 + origin.view(1, 1, 2)
        lower = center - size.view(1, num_nodes, 2) / 2.0

        pairs, weights, directed, directed_weights = self._topology(cond, device, dtype)
        demand = torch.zeros(num_nodes, device=device, dtype=dtype)
        demand.index_add_(0, pairs[0], weights)
        demand.index_add_(0, pairs[1], weights)
        directed_demand = 2.0 * demand

        if "tap_hubump" in cond:
            hubump = cond.tap_hubump.to(device=device, dtype=dtype).view(-1)
        else:
            hubump = self._derive_hubump(size[:, 0], size[:, 1], directed_demand)
        side_capacity, total_capacity = self._capacities(size[:, 0], size[:, 1], hubump)

        geom = torch.cat(
            (
                lower,
                size.view(1, num_nodes, 2).expand(batch_size, -1, -1),
                hubump.view(1, num_nodes, 1).expand(batch_size, -1, -1),
            ),
            dim=-1,
        )
        clumps = self._clumps(geom)
        src, dst = pairs
        pair_dist = (clumps[:, src, :, None, :] - clumps[:, dst, None, :, :]).abs().sum(dim=-1)
        if smooth_preferred and self.preferred_side_temperature > 0.0:
            pair_probability = torch.softmax(
                -pair_dist.flatten(start_dim=-2) / self.preferred_side_temperature,
                dim=-1,
            ).view(batch_size, pairs.shape[1], 4, 4)
            src_probability = pair_probability.sum(dim=-1)
            dst_probability = pair_probability.sum(dim=-2)
            pref_graphs = []
            for graph_id in range(batch_size):
                graph_pref = torch.zeros(num_nodes, 4, device=device, dtype=dtype)
                graph_pref = graph_pref.index_add(0, src, weights.view(-1, 1) * src_probability[graph_id])
                graph_pref = graph_pref.index_add(0, dst, weights.view(-1, 1) * dst_probability[graph_id])
                pref_graphs.append(graph_pref)
            pref = torch.stack(pref_graphs, dim=0)
        else:
            flat_choice = pair_dist.detach().flatten(start_dim=-2).argmin(dim=-1)
            src_side = torch.div(flat_choice, 4, rounding_mode="floor")
            dst_side = flat_choice.remainder(4)
            pref = torch.zeros(batch_size, num_nodes, 4, device=device, dtype=dtype)
            for graph_id in range(batch_size):
                pref[graph_id].index_put_((src, src_side[graph_id]), weights, accumulate=True)
                pref[graph_id].index_put_((dst, dst_side[graph_id]), weights, accumulate=True)

        power = (
            cond.node_power.to(device=device, dtype=dtype).view(-1)
            if "node_power" in cond
            else torch.zeros(num_nodes, device=device, dtype=dtype)
        )
        rotation = torch.zeros(num_nodes, device=device, dtype=dtype)
        static = torch.stack(
            (
                size[:, 0], size[:, 1], rotation, power, hubump,
                torch.log1p(directed_demand), torch.log1p(total_capacity),
                directed_demand / total_capacity.clamp_min(1.0),
            ),
            dim=-1,
        )
        static = static.view(1, num_nodes, 8).expand(batch_size, -1, -1)
        capacity_features = torch.log1p(side_capacity).view(1, num_nodes, 4).expand(batch_size, -1, -1)
        raw_nodes = torch.cat((lower, static[..., :5], static[..., 5:], capacity_features, torch.log1p(pref)), dim=-1)
        nodes = (raw_nodes - self.node_mean.to(dtype).view(1, 1, -1)) / self.node_std.to(dtype).view(1, 1, -1)

        directed_src, directed_dst = directed
        directed_pair_dist = (
            clumps[:, directed_src, :, None, :] - clumps[:, directed_dst, None, :, :]
        ).abs().sum(dim=-1)
        dmin = directed_pair_dist.flatten(start_dim=-2).amin(dim=-1)
        raw_edges = torch.stack(
            (torch.log1p(directed_weights).view(1, -1).expand(batch_size, -1), dmin),
            dim=-1,
        )
        edge_attr = (raw_edges - self.edge_mean.to(dtype).view(1, 1, -1)) / self.edge_std.to(dtype).view(1, 1, -1)

        left = lower[..., 0].amin(dim=1)
        bottom = lower[..., 1].amin(dim=1)
        right = (lower[..., 0] + size[:, 0].view(1, -1)).amax(dim=1)
        top = (lower[..., 1] + size[:, 1].view(1, -1)).amax(dim=1)
        die_area = (right - left).clamp_min(1e-6) * (top - bottom).clamp_min(1e-6)
        wcount = weights.sum().expand(batch_size)
        global_attr = torch.stack(
            (
                torch.full_like(die_area, math.log1p(num_nodes)),
                torch.full_like(die_area, math.log1p(pairs.shape[1])),
                torch.log1p(wcount),
                torch.log1p(die_area),
            ),
            dim=-1,
        )

        edge_copies = [directed + graph_id * num_nodes for graph_id in range(batch_size)]
        return {
            "x": nodes.reshape(batch_size * num_nodes, 18),
            "node_geom": geom.reshape(batch_size * num_nodes, 5),
            "edge_index": torch.cat(edge_copies, dim=1),
            "edge_attr": edge_attr.reshape(-1, 2),
            "edge_weight": directed_weights.repeat(batch_size),
            "cong": torch.cat((capacity_features, torch.log1p(pref)), dim=-1).reshape(batch_size * num_nodes, 8),
            "global_attr": global_attr,
            "batch": torch.arange(batch_size, device=device).repeat_interleave(num_nodes),
            "wcount": wcount,
        }

    def predict(self, placement, cond, smooth_preferred=False):
        inputs = self._build_batch(placement, cond, smooth_preferred=smooth_preferred)
        total, edge_distance, side_flow = self.model(
            inputs["x"], inputs["edge_index"], inputs["edge_attr"], inputs["edge_weight"],
            inputs["node_geom"], inputs["batch"], inputs["global_attr"], inputs["cong"],
        )
        average = total / (2.0 * inputs["wcount"].clamp_min(1e-6))
        return {"total": total, "average": average, "edge_distance": edge_distance, "side_flow": side_flow}

    def pair_distance_matrix(self, placement, cond, smooth_preferred=True):
        """Return a differentiable dense pair-distance feature for GeometryAttention.

        Connected pairs receive WirelengthGNN's edge-level routed-distance
        prediction (microbump-side Manhattan lower bound plus learned congestion
        correction).  The wirelength model has no supervised meaning for absent
        netlist edges, so those entries retain center Euclidean distance rather
        than being represented as misleading zero-distance pairs.

        WirelengthGNN operates in millimetres while the flow model uses roughly
        [-1, 1] canvas coordinates.  Dividing by half the mean canvas side puts
        the prediction on the same dimensionless scale as the original fifth
        GeometryAttention feature.
        """
        if placement.dim() == 2:
            placement = placement.unsqueeze(0)
        batch_size, num_nodes, _ = placement.shape
        inputs = self._build_batch(placement, cond, smooth_preferred=smooth_preferred)
        _, edge_distance, _ = self.model(
            inputs["x"], inputs["edge_index"], inputs["edge_attr"], inputs["edge_weight"],
            inputs["node_geom"], inputs["batch"], inputs["global_attr"], inputs["cong"],
        )

        # Dense geometry attention also covers non-netlist pairs.  Preserve the
        # original geometric meaning for them and replace connected entries only.
        delta = placement[:, :, None, :2] - placement[:, None, :, :2]
        pair_distance = torch.linalg.vector_norm(delta, dim=-1)

        directed_edges_per_graph = edge_distance.numel() // batch_size
        predicted = edge_distance.view(batch_size, directed_edges_per_graph)
        local_edges = inputs["edge_index"][:, :directed_edges_per_graph]
        src = local_edges[0]
        dst = local_edges[1]
        canvas, _ = self._canvas(cond, placement.device, placement.dtype)
        coordinate_scale = (0.5 * canvas.mean()).clamp_min(1e-6)
        predicted = predicted / coordinate_scale

        pair_distance = pair_distance.clone()
        pair_distance[:, src, dst] = predicted
        return pair_distance

    def potential(self, placement, cond):
        prediction = self.predict(placement, cond, smooth_preferred=True)
        if self.objective == "log_total":
            return torch.log(prediction["total"].clamp_min(1e-8))
        if self.objective == "log_average":
            return torch.log(prediction["average"].clamp_min(1e-8))
        if self.objective == "total":
            return prediction["total"]
        if self.objective == "average":
            return prediction["average"]
        raise ValueError(f"unknown wirelength objective: {self.objective}")
