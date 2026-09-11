"""Local chiplet data model and geometry helpers for the standalone RL package."""

from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional, Tuple


MIN_OVERLAP = 1
EPSILON = 1e-9


class _EdgeView:
    def __init__(self, graph: "SimpleGraph"):
        self._graph = graph

    def __call__(self, data: bool = False):
        return list(self._graph.iter_edges(data=data))

    def __iter__(self):
        return iter(self._graph.iter_edges(data=False))

    def __len__(self) -> int:
        return self._graph.number_of_edges()


class SimpleGraph:
    """Small undirected graph API compatible with the calls used by RL."""

    def __init__(self):
        self._adj: Dict[str, Dict[str, Dict]] = {}
        self.edges = _EdgeView(self)

    def add_node(self, node: str) -> None:
        self._adj.setdefault(node, {})

    def add_edge(self, node1: str, node2: str, **attrs) -> None:
        self.add_node(node1)
        self.add_node(node2)
        edge_attrs = dict(attrs)
        self._adj[node1][node2] = edge_attrs
        self._adj[node2][node1] = edge_attrs

    def neighbors(self, node: str) -> Iterator[str]:
        return iter(self._adj.get(node, {}))

    def has_edge(self, node1: str, node2: str) -> bool:
        return node2 in self._adj.get(node1, {})

    def number_of_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self._adj.values()) // 2

    def iter_edges(self, data: bool = False):
        seen = set()
        for node1, neighbors in self._adj.items():
            for node2, attrs in neighbors.items():
                edge_key = tuple(sorted((node1, node2)))
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                if data:
                    yield node1, node2, attrs
                else:
                    yield node1, node2

    def __contains__(self, node: str) -> bool:
        return node in self._adj

    def __getitem__(self, node: str) -> Dict[str, Dict]:
        return self._adj[node]


class Chiplet:
    """A rectangular chiplet with position, size, rotation, and power.

    ``footprint_w`` / ``footprint_h`` are the *declared* occupied (bump-expanded)
    envelope of the unrotated body, as authored in the benchmark JSON. They are
    the authoritative placement envelope when present: the microbump halo is then
    ``(footprint - body) / 2`` per axis. ``hubump`` stays the symmetric ring
    width used by FastTM and is the fallback when no footprint is declared.
    """

    def __init__(
        self,
        chip_id: str,
        width: float,
        height: float,
        x: float = 0.0,
        y: float = 0.0,
        rotation: bool = 0,
        power: float = 0.0,
        hubump: float = 0.0,
        footprint_w: Optional[float] = None,
        footprint_h: Optional[float] = None,
    ):
        self.id = chip_id
        self.width = float(width)
        self.height = float(height)
        self.x = float(x)
        self.y = float(y)
        self.rotation = rotation
        self.power = float(power)
        self.hubump = float(hubump)
        self.footprint_w = None if footprint_w is None else float(footprint_w)
        self.footprint_h = None if footprint_h is None else float(footprint_h)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def has_declared_footprint(self) -> bool:
        return self.footprint_w is not None and self.footprint_h is not None

    def get_footprint(self) -> Optional[Tuple[float, float]]:
        """Declared occupied size (unrotated axes), or None when not declared."""
        if not self.has_declared_footprint():
            return None
        return (self.footprint_w, self.footprint_h)

    def __repr__(self) -> str:
        return (
            f"Chiplet(id='{self.id}', width={self.width}, height={self.height}, "
            f"x={self.x}, y={self.y}, rotation={self.rotation}, power={self.power}, "
            f"hubump={self.hubump}, footprint=({self.footprint_w}, {self.footprint_h}))"
        )


class LayoutProblem:
    """Represents a complete chiplet placement problem."""

    def __init__(self):
        self.chiplets: Dict[str, Chiplet] = {}
        self.connection_graph = SimpleGraph()
        self.all_connections: List[Dict] = []

    def add_chiplet(self, chiplet: Chiplet) -> None:
        self.chiplets[chiplet.id] = chiplet
        self.connection_graph.add_node(chiplet.id)

    def add_connection(self, chip_id1: str, chip_id2: str, weight: float = 1.0) -> None:
        if chip_id1 not in self.chiplets or chip_id2 not in self.chiplets:
            raise ValueError(f"Chiplet '{chip_id1}' or '{chip_id2}' not found")
        self.connection_graph.add_edge(chip_id1, chip_id2, weight=weight)

    def get_chiplet(self, chip_id: str) -> Optional[Chiplet]:
        return self.chiplets.get(chip_id)

    def get_neighbors(self, chip_id: str) -> List[str]:
        if chip_id in self.connection_graph:
            return list(self.connection_graph.neighbors(chip_id))
        return []

    def get_connection_weight(self, chip_id1: str, chip_id2: str) -> Optional[float]:
        if self.connection_graph.has_edge(chip_id1, chip_id2):
            return self.connection_graph[chip_id1][chip_id2].get("weight", 1.0)
        return None

    def is_cyclic(self) -> bool:
        visited = set()

        def visit(node: str, parent: Optional[str]) -> bool:
            visited.add(node)
            for neighbor in self.connection_graph.neighbors(node):
                if neighbor == parent:
                    continue
                if neighbor in visited or visit(neighbor, node):
                    return True
            return False

        return any(visit(node, None) for node in self.chiplets if node not in visited)

    def __repr__(self) -> str:
        return (
            f"LayoutProblem(chiplets={len(self.chiplets)}, "
            f"connections={self.connection_graph.number_of_edges()})"
        )


def load_problem_from_json(file_path: str) -> LayoutProblem:
    """Load a LayoutProblem from JSON using either chiplets or dies schema."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    problem = LayoutProblem()
    chiplets_data = data.get("chiplets", data.get("dies", []))
    if not chiplets_data:
        raise KeyError("JSON must contain a 'chiplets' or 'dies' field")

    for chiplet_data in chiplets_data:
        chip_id = chiplet_data.get("name") or chiplet_data.get("id")
        if chip_id is None:
            raise KeyError("Each chiplet/die must have a 'name' or 'id' field")
        problem.add_chiplet(
            Chiplet(
                chip_id=chip_id,
                width=chiplet_data["width"],
                height=chiplet_data["height"],
                x=chiplet_data.get("x", 0.0),
                y=chiplet_data.get("y", 0.0),
                power=chiplet_data.get("power", 0.0),
                hubump=chiplet_data.get("hubump", 0.0),
                footprint_w=chiplet_data.get("footprint_w"),
                footprint_h=chiplet_data.get("footprint_h"),
            )
        )

    for connection in data.get("connections", []):
        if isinstance(connection, dict):
            chip_id1 = connection.get("node1") or connection.get("source") or connection.get("from")
            chip_id2 = connection.get("node2") or connection.get("target") or connection.get("to")
            weight = float(connection.get("weight", 1.0))
            if chip_id1 is None or chip_id2 is None:
                raise KeyError("Connection dict must contain node1/node2 fields")
            problem.add_connection(chip_id1, chip_id2, weight)
            problem.all_connections.append(dict(connection))
            edge_data = problem.connection_graph[chip_id1][chip_id2]
            for key, value in connection.items():
                edge_data[key] = value
        elif len(connection) == 2:
            problem.add_connection(connection[0], connection[1])
        elif len(connection) >= 3:
            problem.add_connection(connection[0], connection[1], float(connection[2]))
        else:
            raise ValueError("Each connection must contain at least two chiplet ids")

    return problem


def has_overlap(chip1: Chiplet, chip2: Chiplet) -> bool:
    """Return True if the two chiplets overlap."""
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()
    x_overlap = not (x1_max <= x2_min + EPSILON or x2_max <= x1_min + EPSILON)
    y_overlap = not (y1_max <= y2_min + EPSILON or y2_max <= y1_min + EPSILON)
    return x_overlap and y_overlap


def get_adjacency_info(chip1: Chiplet, chip2: Chiplet) -> Tuple[bool, float, str]:
    """
    Check whether two chiplets are adjacent and return shared edge length.

    Direction is from chip1's perspective: right, left, top, bottom, or none.
    """
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()

    if abs(x1_max - x2_min) < EPSILON:
        overlap_length = min(y1_max, y2_max) - max(y1_min, y2_min)
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, "right"

    if abs(x1_min - x2_max) < EPSILON:
        overlap_length = min(y1_max, y2_max) - max(y1_min, y2_min)
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, "left"

    if abs(y1_max - y2_min) < EPSILON:
        overlap_length = min(x1_max, x2_max) - max(x1_min, x2_min)
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, "top"

    if abs(y1_min - y2_max) < EPSILON:
        overlap_length = min(x1_max, x2_max) - max(x1_min, x2_min)
        if overlap_length >= MIN_OVERLAP - EPSILON:
            return True, overlap_length, "bottom"

    return False, 0.0, "none"
