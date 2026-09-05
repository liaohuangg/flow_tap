#!/usr/bin/env python3
"""gen_legal_pla_greedy.py

Deterministic fast heuristic placement generator.

- Hard constraint: ONLY process system_i where i >= 10425.
- Coordinate grid: x/y snapped to 0.001.
- Algorithm: connectivity-aware clustering + size-first greedy legalization + force-directed refinement.

Input schema: Dataset/dataset/input_test/system_i.json
Output schema: Dataset/dataset/output/placement/system_i.json (via tool.generate_placement_json_with_EMIB)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# tool.py imports a top-level "placement" package; when this script is executed via an
# absolute path, the repo root (e.g. /root) may not be on sys.path. Add it deterministically.
_script_dir = Path(__file__).resolve().parent
for _anc in [_script_dir, *_script_dir.parents]:
    if (_anc / "placement").is_dir():
        sys.path.insert(0, str(_anc))
        break

from tool import ChipletNode, generate_placement_json_with_EMIB, load_emib_placement_json

MIN_START_ID = 1
GRID = 0.001

# Layout compactness defaults
DEFAULT_MIN_UTIL = 0.55
DEFAULT_MIN_ASPECT = 0.40
DEFAULT_OUTLINE_SLACK = 1.02
DEFAULT_BBOX_AREA_SLACK = 0.05
DEFAULT_COMPACT_PASSES = 8


def snap001(v: float) -> float:
    # snap to 0.001 grid deterministically
    return float(int(round(v * 1000.0))) / 1000.0


def clip(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


@dataclass(frozen=True)
class Rect:
    name: str
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def _compute_initial_bbox(nodes: List[ChipletNode]) -> Tuple[float, float]:
    # Legacy heuristic: keep for reference. The actual compactness control is handled
    # by outline-driven retries in _place_one_case().
    total_area = 0.0
    max_w = 0.0
    max_h = 0.0
    for n in nodes:
        w = float(n.dimensions.get("x", 0.0) or 0.0)
        h = float(n.dimensions.get("y", 0.0) or 0.0)
        total_area += w * h
        max_w = max(max_w, w, h)
        max_h = max(max_h, w, h)
    estimated_side = int(math.ceil(math.sqrt(total_area * 2.0))) if total_area > 0 else 1
    W = float(estimated_side * 3)
    H = float(estimated_side * 3)
    W = max(W, max_w * 2.0)
    H = max(H, max_h * 2.0)
    return W, H


def _initial_outline_from_area(
    nodes: List[ChipletNode],
    min_util: float,
) -> Tuple[float, float, float, float, float]:
    """Return (W, H, total_area, area_limit, min_dim_guard).

    We target a roughly square outline whose area is bounded by total_area/min_util.
    """

    total_area = 0.0
    max_w = 0.0
    max_h = 0.0
    for n in nodes:
        w = float(n.dimensions.get("x", 0.0) or 0.0)
        h = float(n.dimensions.get("y", 0.0) or 0.0)
        total_area += w * h
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    util = float(min_util or DEFAULT_MIN_UTIL)
    util = clip(util, 0.05, 0.98)
    area_limit = total_area / util if total_area > 0 else 1.0

    # Prefer square to avoid very small aspect ratios.
    side = math.ceil(math.sqrt(area_limit)) if area_limit > 0 else 1.0
    side = float(max(1.0, side))

    # Guard: outline must fit the largest chiplet.
    guard = float(max(max_w, max_h, 1.0))
    W = max(side, guard)
    H = max(side, guard)
    return float(W), float(H), float(total_area), float(area_limit), float(guard)


def _tight_bbox(rects: Dict[str, Rect]) -> Tuple[float, float]:
    if not rects:
        return 0.0, 0.0
    bw = max(r.x + r.w for r in rects.values())
    bh = max(r.y + r.h for r in rects.values())
    return snap001(bw), snap001(bh)


def _aspect_ratio(w: float, h: float) -> float:
    if w <= 0 or h <= 0:
        return 0.0
    return float(min(w, h) / max(w, h))


def _wirelength_from_layout(
    layout: Dict[str, Tuple[float, float]],
    chiplet_dims: Dict[str, Tuple[float, float]],
    edge_map: Dict[Tuple[str, str], dict],
) -> float:
    wl = 0.0
    for (a, b), edge in edge_map.items():
        n1, n2 = edge.get("node1", a), edge.get("node2", b)
        x1, y1 = layout.get(n1, (0.0, 0.0))
        x2, y2 = layout.get(n2, (0.0, 0.0))
        w1, h1 = chiplet_dims.get(n1, (0.0, 0.0))
        w2, h2 = chiplet_dims.get(n2, (0.0, 0.0))
        cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
        cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
        wl += float(edge.get("wireCount", 1)) * (abs(cx1 - cx2) + abs(cy1 - cy2))
    return float(wl)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    # 边界贴合也算重叠：只有严格分离才算不重叠
    if a.x + a.w < b.x:
        return False
    if b.x + b.w < a.x:
        return False
    if a.y + a.h < b.y:
        return False
    if b.y + b.h < a.y:
        return False
    return True


def _overlap_amount(a: Rect, b: Rect) -> Tuple[float, float]:
    ox = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    oy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    return (ox, oy)


def _y_intervals_overlap(a: Rect, b: Rect) -> bool:
    # 边界贴合也算 overlap
    return not (a.y + a.h < b.y or b.y + b.h < a.y)


def _x_intervals_overlap(a: Rect, b: Rect) -> bool:
    # 边界贴合也算 overlap
    return not (a.x + a.w < b.x or b.x + b.w < a.x)


def _has_any_overlap(placed: Dict[str, Rect]) -> bool:
    names = sorted(placed.keys())
    for i in range(len(names)):
        a = placed[names[i]]
        for j in range(i + 1, len(names)):
            b = placed[names[j]]
            if _rects_overlap(a, b):
                return True
    return False


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent: Dict[str, str] = {x: x for x in items}
        self.rank: Dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def _build_neighbor_map(edge_map: Dict[Tuple[str, str], dict]) -> Dict[str, List[Tuple[str, float]]]:
    nbrs: Dict[str, List[Tuple[str, float]]] = {}
    for (a, b), e in edge_map.items():
        w = float(e.get("wireCount", 1.0) or 1.0)
        nbrs.setdefault(a, []).append((b, w))
        nbrs.setdefault(b, []).append((a, w))
    for k in nbrs:
        nbrs[k].sort(key=lambda t: (t[0], -t[1]))
    return nbrs


def _clusters(nodes: List[ChipletNode], edge_map: Dict[Tuple[str, str], dict]) -> Dict[str, int]:
    names = [n.name for n in nodes]
    uf = UnionFind(names)

    weights: List[float] = []
    edges: List[Tuple[str, str, float]] = []
    for (a, b), e in edge_map.items():
        w = float(e.get("wireCount", 0.0) or 0.0)
        if w <= 0:
            continue
        edges.append((a, b, w))
        weights.append(w)

    if not weights:
        return {n: i for i, n in enumerate(sorted(names))}

    weights.sort()
    idx = int(round(0.80 * (len(weights) - 1)))
    idx = max(0, min(idx, len(weights) - 1))
    T = weights[idx]

    # deterministic: iterate edges in sorted (w desc, a, b)
    edges.sort(key=lambda t: (-t[2], t[0], t[1]))
    for a, b, w in edges:
        if w >= T:
            uf.union(a, b)

    root_to_members: Dict[str, List[str]] = {}
    for n in names:
        r = uf.find(n)
        root_to_members.setdefault(r, []).append(n)

    # deterministically assign cluster ids by sorted member lists
    clusters = sorted(
        (sorted(members), root) for root, members in root_to_members.items()
    )
    name_to_cluster: Dict[str, int] = {}
    for cid, (members, _root) in enumerate(clusters):
        for m in members:
            name_to_cluster[m] = cid
    return name_to_cluster


def _cluster_metrics(
    nodes: List[ChipletNode],
    edge_map: Dict[Tuple[str, str], dict],
    name_to_cluster: Dict[str, int],
) -> Dict[int, Tuple[float, float]]:
    # returns cid -> (cluster_area, cluster_degree)
    area: Dict[int, float] = {}
    degree: Dict[int, float] = {}

    for n in nodes:
        cid = name_to_cluster[n.name]
        w = float(n.dimensions.get("x", 0.0) or 0.0)
        h = float(n.dimensions.get("y", 0.0) or 0.0)
        area[cid] = area.get(cid, 0.0) + w * h

    for (a, b), e in edge_map.items():
        w = float(e.get("wireCount", 0.0) or 0.0)
        if w <= 0:
            continue
        ca = name_to_cluster.get(a)
        cb = name_to_cluster.get(b)
        if ca is None or cb is None:
            continue
        degree[ca] = degree.get(ca, 0.0) + w
        degree[cb] = degree.get(cb, 0.0) + w

    out: Dict[int, Tuple[float, float]] = {}
    for cid in set(name_to_cluster.values()):
        out[cid] = (area.get(cid, 0.0), degree.get(cid, 0.0))
    return out


def _anchor_points(W: float, H: float) -> List[Tuple[float, float]]:
    # positions are centers (not lower-left), deterministic order
    xs = [0.5, 0.25, 0.75, 0.5, 0.5, 0.25, 0.75, 0.25, 0.75]
    ys = [0.5, 0.25, 0.25, 0.75, 0.25, 0.5, 0.5, 0.75, 0.75]
    pts = [(W * x, H * y) for x, y in zip(xs, ys)]

    # add a 3x3 grid excluding duplicates
    for gx in [0.1666667, 0.5, 0.8333333]:
        for gy in [0.1666667, 0.5, 0.8333333]:
            pts.append((W * gx, H * gy))
    # stable dedup with rounding
    seen = set()
    out = []
    for x, y in pts:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append((x, y))
    return out


def _infer_range_end(input_dir: Path) -> int:
    max_i: Optional[int] = None
    for p in input_dir.glob("system_*.json"):
        stem = p.stem
        try:
            i_str = stem.split("_", 1)[1]
            i_val = int(i_str)
        except Exception:
            continue
        if max_i is None or i_val > max_i:
            max_i = i_val
    if max_i is None:
        raise SystemExit(f"No system_*.json files found in: {input_dir}")
    return max_i


def _fallback_load_no_connections(json_path: str) -> Tuple[List[ChipletNode], List[dict], Dict[Tuple[str, str], dict], Dict[str, int]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chiplets = data.get("chiplets", [])
    if not isinstance(chiplets, list):
        raise ValueError(f"JSON文件格式错误：必须包含 'chiplets' 列表字段。文件: {json_path}")

    nodes: List[ChipletNode] = []
    for i, c in enumerate(chiplets):
        if not isinstance(c, dict):
            raise ValueError(f"chiplet 格式错误：索引 {i}: {c}")
        name = c.get("name")
        width = c.get("width")
        height = c.get("height")
        power = c.get("power")
        if name is None or width is None or height is None or power is None:
            raise ValueError(f"chiplet 格式错误：索引 {i}: {c}")
        nodes.append(
            ChipletNode(
                name=str(name),
                dimensions={"x": float(width), "y": float(height)},
                phys=[],
                power=float(power),
            )
        )

    name_to_idx = {n.name: i for i, n in enumerate(nodes)}
    return nodes, [], {}, name_to_idx


def _load_case(json_path: Path) -> Tuple[List[ChipletNode], List[dict], Dict[Tuple[str, str], dict], Dict[str, int]]:
    try:
        return load_emib_placement_json(str(json_path))
    except Exception as e:
        msg = str(e)
        if "connections 不能为空" in msg or "must包含 'connections'" in msg or "connections" in msg:
            # allow cases with empty connections (treat as independent packing)
            return _fallback_load_no_connections(str(json_path))
        raise


def _place_one_case(
    nodes: List[ChipletNode],
    edge_map: Dict[Tuple[str, str], dict],
    output_path: Path,
) -> None:
    # Compactness constraints (defaults picked to match user requirements)
    min_util = float(DEFAULT_MIN_UTIL)
    min_aspect = float(DEFAULT_MIN_ASPECT)
    outline_slack = float(DEFAULT_OUTLINE_SLACK)
    bbox_area_slack = float(DEFAULT_BBOX_AREA_SLACK)
    compact_passes = int(DEFAULT_COMPACT_PASSES)

    outline_grow = 1.08
    inner_max_attempts = 30
    max_retries = 10

    W0, H0, _total_area, area_limit, guard = _initial_outline_from_area(nodes, min_util=min_util)

    # Retry loop: if strict overlap remains, grow outline and retry.
    last_best: Optional[dict] = None
    last_satisfied = False

    for retry in range(max_retries):
        W = max(W0 * outline_slack, guard) * (outline_grow ** retry)
        H = max(H0 * outline_slack, guard) * (outline_grow ** retry)

        chiplet_dims0: Dict[str, Tuple[float, float]] = {
            n.name: (
                float(n.dimensions.get("x", 0.0) or 0.0),
                float(n.dimensions.get("y", 0.0) or 0.0),
            )
            for n in nodes
        }

        name_to_cluster = _clusters(nodes, edge_map)
        metrics = _cluster_metrics(nodes, edge_map, name_to_cluster)

        # cluster ordering
        cluster_ids = sorted(
            set(name_to_cluster.values()),
            key=lambda cid: (
                -metrics.get(cid, (0.0, 0.0))[1],
                -metrics.get(cid, (0.0, 0.0))[0],
                cid,
            ),
        )

        nbrs = _build_neighbor_map(edge_map)

        # stable chiplet ordering: by cluster order, then area desc, then name
        node_area: Dict[str, float] = {
            n.name: float(n.dimensions.get("x", 0.0)) * float(n.dimensions.get("y", 0.0)) for n in nodes
        }
        cluster_rank = {cid: r for r, cid in enumerate(cluster_ids)}

        ordered = sorted(
            nodes,
            key=lambda n: (
                cluster_rank.get(name_to_cluster[n.name], 10**9),
                -node_area.get(n.name, 0.0),
                n.name,
            ),
        )

        def _try_place_with_outline(outW: float, outH: float) -> dict:
            # anchors depend on outline (deterministic variation across attempts)
            anchors = _anchor_points(outW, outH)
            cluster_center: Dict[int, Tuple[float, float]] = {}
            for k, cid in enumerate(cluster_ids):
                ax, ay = anchors[min(k, len(anchors) - 1)]
                cluster_center[cid] = (ax, ay)

            layout: Dict[str, Tuple[float, float]] = {}
            rotations: Dict[str, bool] = {}
            placed: Dict[str, Rect] = {}
            bbox_w = 0.0
            bbox_h = 0.0

            def can_place(x: float, y: float, w: float, h: float) -> bool:
                r = Rect("_", x, y, w, h)
                for pr in placed.values():
                    if _rects_overlap(r, pr):
                        return False
                return True

            def neighbor_cost(name: str, cx: float, cy: float) -> float:
                cost = 0.0
                for nb, wt in nbrs.get(name, []):
                    pr = placed.get(nb)
                    if pr is None:
                        continue
                    cost += wt * (abs(cx - pr.cx) + abs(cy - pr.cy))
                return cost

            def bbox_penalty(x: float, y: float, w: float, h: float) -> float:
                # Penalize bbox growth, imbalance, and aspect deficiency.
                new_w = max(bbox_w, x + w)
                new_h = max(bbox_h, y + h)
                dw = new_w - bbox_w
                dh = new_h - bbox_h

                old_area = bbox_w * bbox_h
                new_area = new_w * new_h
                area_growth = new_area - old_area
                imbalance = abs(new_w - new_h)
                asp = _aspect_ratio(new_w, new_h)
                asp_def = max(0.0, min_aspect - asp)

                # Candidate hard-ish cutoff: avoid obviously over-large bboxes.
                if area_limit > 0 and new_area > area_limit * (1.0 + bbox_area_slack + 0.10):
                    return 1e15

                # Deterministic weighted sum.
                return (
                    0.4 * (dw + dh)
                    + 1.5 * area_growth
                    + 0.2 * imbalance
                    + 5000.0 * asp_def
                )

            def candidate_positions(px: float, py: float, w: float, h: float) -> List[Tuple[float, float]]:
                xs: List[float] = [px, 0.0, outW - w]
                ys: List[float] = [py, 0.0, outH - h]
                for pr in placed.values():
                    xs.extend([pr.x, pr.x + pr.w, pr.x - w, pr.x + pr.w - w])
                    ys.extend([pr.y, pr.y + pr.h, pr.y - h, pr.y + pr.h - h])

                # snap + clip + dedup
                xset = []
                seenx = set()
                for x in xs:
                    xx = snap001(clip(x, 0.0, max(0.0, outW - w)))
                    if xx in seenx:
                        continue
                    seenx.add(xx)
                    xset.append(xx)
                yset = []
                seeny = set()
                for y in ys:
                    yy = snap001(clip(y, 0.0, max(0.0, outH - h)))
                    if yy in seeny:
                        continue
                    seeny.add(yy)
                    yset.append(yy)

                xset.sort(key=lambda x: (abs(x - px), x))
                yset.sort(key=lambda y: (abs(y - py), y))
                K = 18
                xset = xset[:K]
                yset = yset[:K]

                out: List[Tuple[float, float]] = []
                for x in xset:
                    for y in yset:
                        out.append((x, y))
                return out

            def resolve_current(name: str, x: float, y: float, w: float, h: float) -> Tuple[float, float]:
                # push current rect away from overlaps (strict: edge-touch counts as overlap)
                cur = Rect(name, x, y, w, h)
                for _ in range(200):
                    moved = False
                    for pr in placed.values():
                        if not _rects_overlap(cur, pr):
                            continue

                        # Strict overlap: if they overlap OR just touch, we must separate by >= GRID.
                        ox, oy = _overlap_amount(cur, pr)

                        # If either axis is disjoint, no move needed.
                        if ox < 0.0 or oy < 0.0:
                            continue

                        # Choose axis with smaller required move.
                        if ox <= oy:
                            step = ox + GRID
                            if cur.cx < pr.cx:
                                nx = cur.x - step
                            else:
                                nx = cur.x + step
                            nx = snap001(clip(nx, 0.0, max(0.0, outW - w)))
                            cur = Rect(name, nx, cur.y, w, h)
                        else:
                            step = oy + GRID
                            if cur.cy < pr.cy:
                                ny = cur.y - step
                            else:
                                ny = cur.y + step
                            ny = snap001(clip(ny, 0.0, max(0.0, outH - h)))
                            cur = Rect(name, cur.x, ny, w, h)

                        moved = True
                    if not moved:
                        break
                return cur.x, cur.y

            # Pre-place up to 5 largest chiplets with anchor search
            top_anchors = _anchor_points(outW, outH)
            topN = min(5, len(ordered))
            top_names = [ordered[i].name for i in range(topN)]

            for n in ordered:
                name = n.name
                cid = name_to_cluster[name]
                w0, h0 = chiplet_dims0.get(name, (0.0, 0.0))

                # desired center
                if name in top_names:
                    desired_centers = top_anchors[:12]
                else:
                    desired_centers = [cluster_center.get(cid, (outW / 2.0, outH / 2.0))]

                # if neighbors already placed: use barycenter
                if name in nbrs and any(nb in placed for nb, _w in nbrs[name]):
                    sx = 0.0
                    sy = 0.0
                    sw = 0.0
                    for nb, wt in nbrs.get(name, []):
                        pr = placed.get(nb)
                        if pr is None:
                            continue
                        sx += wt * pr.cx
                        sy += wt * pr.cy
                        sw += wt
                    if sw > 0:
                        desired_centers = [(sx / sw, sy / sw)]

                best: Optional[Tuple[float, float, bool, float]] = None
                for dcx, dcy in desired_centers:
                    for rot in (False, True):
                        w, h = (h0, w0) if rot else (w0, h0)
                        px = snap001(clip(dcx - w / 2.0, 0.0, max(0.0, outW - w)))
                        py = snap001(clip(dcy - h / 2.0, 0.0, max(0.0, outH - h)))

                        for x, y in candidate_positions(px, py, w, h):
                            if not can_place(x, y, w, h):
                                continue
                            cx = x + w / 2.0
                            cy = y + h / 2.0
                            cost = neighbor_cost(name, cx, cy) + bbox_penalty(x, y, w, h)
                            cost += 0.001 * (abs(x - px) + abs(y - py))
                            cand = (x, y, rot, cost)
                            if best is None or cand[3] < best[3] or (cand[3] == best[3] and cand[:3] < best[:3]):
                                best = cand

                    if best is not None:
                        break

                # fallback: if no non-overlap candidate found, allow overlap then resolve
                if best is None:
                    rot = False
                    w, h = w0, h0
                    x = snap001(clip(outW / 2.0 - w / 2.0, 0.0, max(0.0, outW - w)))
                    y = snap001(clip(outH / 2.0 - h / 2.0, 0.0, max(0.0, outH - h)))
                    x, y = resolve_current(name, x, y, w, h)
                    best = (x, y, rot, 0.0)

                x, y, rot, _c = best
                w, h = (h0, w0) if rot else (w0, h0)
                x, y = resolve_current(name, x, y, w, h)

                placed[name] = Rect(name, x, y, w, h)
                layout[name] = (x, y)
                rotations[name] = bool(rot)
                bbox_w = max(bbox_w, x + w)
                bbox_h = max(bbox_h, y + h)

            def compact_left() -> bool:
                moved = False
                names_sorted = sorted(placed.keys(), key=lambda nm: (placed[nm].x, placed[nm].y, nm))
                for nm in names_sorted:
                    r = placed[nm]
                    block = 0.0
                    for onm, o in placed.items():
                        if onm == nm:
                            continue
                        if not _y_intervals_overlap(r, o):
                            continue
                        if o.x + o.w + GRID <= r.x:
                            block = max(block, o.x + o.w + GRID)
                    nx = snap001(clip(block, 0.0, max(0.0, outW - r.w)))
                    if nx != r.x:
                        nr = Rect(r.name, nx, r.y, r.w, r.h)
                        if all((onm == nm) or (not _rects_overlap(nr, o)) for onm, o in placed.items()):
                            placed[nm] = nr
                            layout[nm] = (nr.x, nr.y)
                            moved = True
                return moved

            def compact_down() -> bool:
                moved = False
                names_sorted = sorted(placed.keys(), key=lambda nm: (placed[nm].y, placed[nm].x, nm))
                for nm in names_sorted:
                    r = placed[nm]
                    block = 0.0
                    for onm, o in placed.items():
                        if onm == nm:
                            continue
                        if not _x_intervals_overlap(r, o):
                            continue
                        if o.y + o.h + GRID <= r.y:
                            block = max(block, o.y + o.h + GRID)
                    ny = snap001(clip(block, 0.0, max(0.0, outH - r.h)))
                    if ny != r.y:
                        nr = Rect(r.name, r.x, ny, r.w, r.h)
                        if all((onm == nm) or (not _rects_overlap(nr, o)) for onm, o in placed.items()):
                            placed[nm] = nr
                            layout[nm] = (nr.x, nr.y)
                            moved = True
                return moved

            def compact_until_stable() -> None:
                for _ in range(max(1, compact_passes)):
                    moved = compact_left()
                    moved = compact_down() or moved
                    if not moved:
                        break

            compact_until_stable()

            # force-directed refinement (clipped to outline), then re-compact
            names = sorted(placed.keys())
            if edge_map and len(names) >= 2:
                k_attr = 1e-6
                k_rep = 0.05
                max_step = 1.0
                for _step in range(30):
                    forces: Dict[str, Tuple[float, float]] = {nm: (0.0, 0.0) for nm in names}

                    for (a, b), e in sorted(edge_map.items(), key=lambda t: (t[0][0], t[0][1])):
                        wa = float(e.get("wireCount", 1.0) or 1.0)
                        ra = placed.get(a)
                        rb = placed.get(b)
                        if ra is None or rb is None:
                            continue
                        dx = rb.cx - ra.cx
                        dy = rb.cy - ra.cy
                        fa = forces[a]
                        fb = forces[b]
                        forces[a] = (fa[0] + wa * dx, fa[1] + wa * dy)
                        forces[b] = (fb[0] - wa * dx, fb[1] - wa * dy)

                    for i in range(len(names)):
                        ri = placed[names[i]]
                        for j in range(i + 1, len(names)):
                            rj = placed[names[j]]
                            dx = ri.cx - rj.cx
                            dy = ri.cy - rj.cy
                            fx_i, fy_i = forces[ri.name]
                            fx_j, fy_j = forces[rj.name]

                            if _rects_overlap(ri, rj):
                                ox, oy = _overlap_amount(ri, rj)
                                # strict: if overlap OR touch, separate by at least GRID
                                if ox >= 0.0 and oy >= 0.0:
                                    if ox <= oy:
                                        s = 1.0 if dx >= 0 else -1.0
                                        push = k_rep * (ox + GRID)
                                        fx_i += s * push
                                        fx_j -= s * push
                                    else:
                                        s = 1.0 if dy >= 0 else -1.0
                                        push = k_rep * (oy + GRID)
                                        fy_i += s * push
                                        fy_j -= s * push
                            else:
                                dist2 = dx * dx + dy * dy
                                if dist2 > 1e-9 and dist2 < 400.0:
                                    inv = 1.0 / math.sqrt(dist2)
                                    fx_i += k_rep * dx * inv
                                    fy_i += k_rep * dy * inv
                                    fx_j -= k_rep * dx * inv
                                    fy_j -= k_rep * dy * inv

                            forces[ri.name] = (fx_i, fy_i)
                            forces[rj.name] = (fx_j, fy_j)

                    moved_any = False
                    for nm in names:
                        r = placed[nm]
                        fx, fy = forces[nm]
                        dcx = clip(fx * k_attr, -max_step, max_step)
                        dcy = clip(fy * k_attr, -max_step, max_step)
                        if abs(dcx) < 1e-12 and abs(dcy) < 1e-12:
                            continue
                        ncx = r.cx + dcx
                        ncy = r.cy + dcy
                        nx = snap001(clip(ncx - r.w / 2.0, 0.0, max(0.0, outW - r.w)))
                        ny = snap001(clip(ncy - r.h / 2.0, 0.0, max(0.0, outH - r.h)))
                        if nx != r.x or ny != r.y:
                            placed[nm] = Rect(r.name, nx, ny, r.w, r.h)
                            layout[nm] = (nx, ny)
                            moved_any = True

                    if not moved_any:
                        break

                    # quick legalization
                    for nm in names:
                        r = placed[nm]
                        nx, ny = resolve_current(nm, r.x, r.y, r.w, r.h)
                        if nx != r.x or ny != r.y:
                            placed[nm] = Rect(nm, nx, ny, r.w, r.h)
                            layout[nm] = (nx, ny)

            compact_until_stable()

            # evaluate
            bbox_tw, bbox_th = _tight_bbox(placed)
            asp = _aspect_ratio(bbox_tw, bbox_th)
            bbox_area = float(bbox_tw * bbox_th)

            overlap_any = _has_any_overlap(placed)

            dims_final = {nm: (placed[nm].w, placed[nm].h) for nm in placed}
            wl = _wirelength_from_layout(layout=layout, chiplet_dims=dims_final, edge_map=edge_map) if edge_map else 0.0

            return {
                "layout": layout,
                "rotations": rotations,
                "placed": placed,
                "bbox_w": bbox_tw,
                "bbox_h": bbox_th,
                "bbox_area": bbox_area,
                "aspect": asp,
                "wirelength": float(wl),
                "overlap": bool(overlap_any),
                "W": float(outW),
                "H": float(outH),
            }

        best: Optional[dict] = None
        best_score: Optional[Tuple[float, float, float]] = None
        satisfied = False

        for _att in range(inner_max_attempts):
            attempt = _try_place_with_outline(W, H)

            score = (
                float(attempt["bbox_area"]),
                -float(attempt["aspect"]),
                float(attempt["wirelength"]),
            )
            if best is None or score < (best_score or score):
                best = attempt
                best_score = score

            ok = (
                (not attempt["overlap"])
                and (attempt["bbox_area"] <= area_limit * (1.0 + bbox_area_slack))
                and (attempt["aspect"] >= min_aspect)
            )
            if ok:
                satisfied = True
                break

            # deterministic outline growth: if aspect is too small, grow the tighter (smaller) side
            bw = float(attempt["bbox_w"])
            bh = float(attempt["bbox_h"])
            asp = float(attempt["aspect"])

            if asp > 0.0 and asp < min_aspect and bw > 0.0 and bh > 0.0:
                if bw >= bh:
                    H *= outline_grow
                else:
                    W *= outline_grow
            else:
                W *= outline_grow
                H *= outline_grow

            W = max(W, guard)
            H = max(H, guard)

        if best is None:
            raise RuntimeError("placement failed: no attempts")

        last_best = best
        last_satisfied = satisfied

        # strict legality gate: stop immediately once legal
        if not best["overlap"]:
            break

    if last_best is None:
        raise RuntimeError("placement failed: no attempts")

    if last_best["overlap"]:
        raise RuntimeError(f"placement failed: still overlap after {max_retries} retries")

    if not last_satisfied:
        # Keep output stable and quiet: only one warning per case.
        print(
            f"[gen_legal_pla_greedy] WARNING: constraints not met; "
            f"best bbox_area={last_best['bbox_area']:.3f} (limit {area_limit:.3f}), "
            f"aspect={last_best['aspect']:.3f} (min {min_aspect:.3f})."
        )

    class _Result:
        pass

    res = _Result()
    res.layout = last_best["layout"]
    res.rotations = last_best["rotations"]
    res.bounding_box = (float(last_best["bbox_w"]), float(last_best["bbox_h"]))
    res.placements = []

    generate_placement_json_with_EMIB(
        res,
        post=None,
        nodes=nodes,
        edge_map=edge_map,
        output_path=str(output_path),
        ctx=None,
    )


def generate_for_directory(
    input_dir: Path,
    output_dir: Path,
    start_idx: int,
    end_idx: Optional[int],
) -> None:
    if start_idx < MIN_START_ID:
        raise SystemExit(f"Hard constraint: --start must be >= {MIN_START_ID}")

    if end_idx is None:
        end_idx = _infer_range_end(input_dir)

    if end_idx < MIN_START_ID:
        raise SystemExit(f"Hard constraint: --end must be >= {MIN_START_ID}")

    if end_idx < start_idx:
        raise SystemExit("--end must be >= --start")

    total = end_idx - start_idx + 1
    print(f"[gen_legal_pla_greedy] Input dir: {input_dir}")
    print(f"[gen_legal_pla_greedy] Output dir: {output_dir}")
    print(f"[gen_legal_pla_greedy] Range: system_i, i={start_idx}..{end_idx} (total {total})")

    for i in range(start_idx, end_idx + 1):
        json_path = input_dir / f"system_{i}.json"
        if not json_path.exists():
            raise SystemExit(f"Missing input JSON: {json_path}")

        out_path = output_dir / f"system_{i}.json"
        pos = i - start_idx + 1
        print(f"[{pos}/{total}] Placing: {json_path.name}")

        nodes, _edges, edge_map, _name_to_idx = _load_case(json_path)
        _place_one_case(nodes=nodes, edge_map=edge_map, output_path=out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic greedy + force placement generator.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing input system_*.json files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to write output placements.")
    parser.add_argument("--start", type=int, default=MIN_START_ID, help=f"Start idx (inclusive), must be >= {MIN_START_ID}.")
    parser.add_argument("--end", type=int, default=None, help="End idx (inclusive). If omitted, inferred from input dir.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    # 脚本位于 Dataset/ 顶层,数据仍在 Dataset/dataset/ 下
    input_dir = Path(args.input_dir).resolve() if args.input_dir else (script_dir / "dataset" / "input_test")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (script_dir / "dataset" / "output" / "placement")

    generate_for_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        start_idx=args.start,
        end_idx=args.end,
    )


if __name__ == "__main__":
    main()
'''

python gen_legal_pla_greedy.py --start 65999 --end 100000 
'''