"""Silicon bridge generation and overlap checks for RL visualizations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

try:
    from .chiplet_model import Chiplet, LayoutProblem, get_adjacency_info, MIN_OVERLAP
except ImportError:
    from chiplet_model import Chiplet, LayoutProblem, get_adjacency_info, MIN_OVERLAP


SILICONBRIDGE_LENGTH = 1


class SiliconBridge:
    """Silicon bridge connecting two adjacent chiplets."""

    def __init__(
        self,
        chip1_id: str,
        chip2_id: str,
        chip1: Chiplet,
        chip2: Chiplet,
        bridge_width: Optional[float] = None,
    ):
        self.chip1_id = chip1_id
        self.chip2_id = chip2_id

        is_adj, overlap_len, direction = get_adjacency_info(chip1, chip2)
        if not is_adj:
            raise ValueError(f"Chiplets {chip1_id} and {chip2_id} are not adjacent")

        self.bridge_width = bridge_width if bridge_width is not None else MIN_OVERLAP
        if overlap_len < self.bridge_width:
            raise ValueError(
                f"Overlap length {overlap_len:.2f} is too short for silicon bridge width {self.bridge_width:.2f}"
            )

        self.direction = direction
        self.bridge_length = SILICONBRIDGE_LENGTH

        if direction in ["left", "right"]:
            self.overlap_start = max(chip1.y, chip2.y)
            self.overlap_end = min(chip1.y + chip1.height, chip2.y + chip2.height)
        else:
            self.overlap_start = max(chip1.x, chip2.x)
            self.overlap_end = min(chip1.x + chip1.width, chip2.x + chip2.width)

        self.bridge_center = (self.overlap_start + self.overlap_end) / 2.0
        self._compute_bounding_box(chip1)

    def _compute_bounding_box(self, chip1: Chiplet) -> None:
        bridge_half_length = self.bridge_length / 2.0
        half_width = self.bridge_width / 2.0

        if self.direction == "right":
            boundary = chip1.x + chip1.width
            self.x_min = boundary - bridge_half_length
            self.x_max = boundary + bridge_half_length
            self.y_min = self.bridge_center - half_width
            self.y_max = self.bridge_center + half_width
        elif self.direction == "left":
            boundary = chip1.x
            self.x_min = boundary - bridge_half_length
            self.x_max = boundary + bridge_half_length
            self.y_min = self.bridge_center - half_width
            self.y_max = self.bridge_center + half_width
        elif self.direction == "top":
            boundary = chip1.y + chip1.height
            self.y_min = boundary - bridge_half_length
            self.y_max = boundary + bridge_half_length
            self.x_min = self.bridge_center - half_width
            self.x_max = self.bridge_center + half_width
        elif self.direction == "bottom":
            boundary = chip1.y
            self.y_min = boundary - bridge_half_length
            self.y_max = boundary + bridge_half_length
            self.x_min = self.bridge_center - half_width
            self.x_max = self.bridge_center + half_width

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)

    def __repr__(self) -> str:
        return (
            f"SiliconBridge({self.chip1_id}-{self.chip2_id}, "
            f"bbox=({self.x_min:.1f},{self.y_min:.1f})-({self.x_max:.1f},{self.y_max:.1f}))"
        )


def rectangles_overlap(
    rect1: Tuple[float, float, float, float],
    rect2: Tuple[float, float, float, float],
) -> bool:
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2
    overlap_x = not (x1_max <= x2_min or x2_max <= x1_min)
    overlap_y = not (y1_max <= y2_min or y2_max <= y1_min)
    return overlap_x and overlap_y


def generate_silicon_bridges(layout: Dict[str, Chiplet], problem: LayoutProblem) -> List[SiliconBridge]:
    """Generate all silicon bridges from layout and connection requirements."""
    bridges = []
    for chip1_id, chip2_id in problem.connection_graph.edges():
        if chip1_id not in layout or chip2_id not in layout:
            continue
        chip1 = layout[chip1_id]
        chip2 = layout[chip2_id]
        is_adj, _, _ = get_adjacency_info(chip1, chip2)
        if not is_adj:
            continue
        try:
            bridges.append(SiliconBridge(chip1_id, chip2_id, chip1, chip2))
        except ValueError:
            continue
    return bridges


def SiliconBridge_is_legal(
    layout: Dict[str, Chiplet],
    problem: LayoutProblem,
    verbose: bool = False,
) -> bool:
    bridges = generate_silicon_bridges(layout, problem)
    all_legal = True
    conflict_count = 0

    for i in range(len(bridges)):
        for j in range(i + 1, len(bridges)):
            bridge1 = bridges[i]
            bridge2 = bridges[j]
            if rectangles_overlap(bridge1.get_bounding_box(), bridge2.get_bounding_box()):
                all_legal = False
                conflict_count += 1
                if verbose:
                    print(f"Bridge overlap: {bridge1.chip1_id}-{bridge1.chip2_id} vs {bridge2.chip1_id}-{bridge2.chip2_id}")

    if verbose:
        print("Legal: All bridges are non-overlapping" if all_legal else f"Illegal: {conflict_count} bridge overlap(s)")
    return all_legal
