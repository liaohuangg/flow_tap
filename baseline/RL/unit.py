"""Local metric and visualization helpers for the standalone RL package."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from .chiplet_model import Chiplet, LayoutProblem, get_adjacency_info
except ImportError:
    from chiplet_model import Chiplet, LayoutProblem, get_adjacency_info


def calculate_wirelength(layout: Dict[str, Chiplet], problem: LayoutProblem) -> float:
    """Return total Euclidean wirelength over connected chip pairs."""
    if not layout or not problem.connection_graph.edges():
        return 0.0

    total_wirelength = 0.0
    for chip1_id, chip2_id in problem.connection_graph.edges():
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        if chip1 is None or chip2 is None:
            continue

        center1_x = chip1.x + chip1.width / 2
        center1_y = chip1.y + chip1.height / 2
        center2_x = chip2.x + chip2.width / 2
        center2_y = chip2.y + chip2.height / 2
        total_wirelength += ((center2_x - center1_x) ** 2 + (center2_y - center1_y) ** 2) ** 0.5

    return total_wirelength


def calculate_manhattan_wirelength(layout: Dict[str, Chiplet], problem: LayoutProblem) -> float:
    """Return total center-based Manhattan wirelength over connected chip pairs."""
    if not layout or not problem.connection_graph.edges():
        return 0.0

    total_wirelength = 0.0
    for chip1_id, chip2_id in problem.connection_graph.edges():
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        if chip1 is None or chip2 is None:
            continue

        center1_x = chip1.x + chip1.width / 2
        center1_y = chip1.y + chip1.height / 2
        center2_x = chip2.x + chip2.width / 2
        center2_y = chip2.y + chip2.height / 2
        total_wirelength += abs(center2_x - center1_x) + abs(center2_y - center1_y)

    return total_wirelength


def get_bridge_center(chip1: Chiplet, chip2: Chiplet, direction: str) -> Tuple[float, float]:
    """Return the bridge center used by the ICCAD23 wirelength metric."""
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()

    if direction == "right":
        return x1_max, (max(y1_min, y2_min) + min(y1_max, y2_max)) / 2.0
    if direction == "left":
        return x1_min, (max(y1_min, y2_min) + min(y1_max, y2_max)) / 2.0
    if direction == "top":
        return (max(x1_min, x2_min) + min(x1_max, x2_max)) / 2.0, y1_max
    if direction == "bottom":
        return (max(x1_min, x2_min) + min(x1_max, x2_max)) / 2.0, y1_min

    center1_x = (x1_min + x1_max) / 2.0
    center1_y = (y1_min + y1_max) / 2.0
    center2_x = (x2_min + x2_max) / 2.0
    center2_y = (y2_min + y2_max) / 2.0
    return (center1_x + center2_x) / 2.0, (center1_y + center2_y) / 2.0


def get_grid_points(chip: Chiplet, grid_size: int = 16) -> List[Tuple[float, float]]:
    """Return the 16x16 chiplet grid points used by ICCAD23 EMIB wirelength."""
    x_min, y_min, x_max, y_max = chip.get_bounds()
    points: List[Tuple[float, float]] = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = x_min + (x_max - x_min) * (i + 0.5) / grid_size
            y = y_min + (y_max - y_min) * (j + 0.5) / grid_size
            points.append((x, y))
    return points


def calculate_emib_wirelength(chip1: Chiplet, chip2: Chiplet, wire_count: int) -> float:
    """ICCAD23 EMIB wirelength: grid points on both chips to the bridge center."""
    _, _, direction = get_adjacency_info(chip1, chip2)
    bridge_center_x, bridge_center_y = get_bridge_center(chip1, chip2, direction)
    grid_points_1 = get_grid_points(chip1, grid_size=16)
    grid_points_2 = get_grid_points(chip2, grid_size=16)

    total_wirelength = 0.0
    for wire_idx in range(wire_count):
        point_x, point_y = grid_points_1[wire_idx % len(grid_points_1)]
        total_wirelength += math.sqrt((point_x - bridge_center_x) ** 2 + (point_y - bridge_center_y) ** 2)

    for wire_idx in range(wire_count):
        point_x, point_y = grid_points_2[wire_idx % len(grid_points_2)]
        total_wirelength += math.sqrt((point_x - bridge_center_x) ** 2 + (point_y - bridge_center_y) ** 2)

    return total_wirelength


def calculate_normal_wirelength(chip1: Chiplet, chip2: Chiplet, wire_count: int) -> float:
    """ICCAD23 normal wirelength: center Manhattan distance times wire count."""
    x1_min, y1_min, x1_max, y1_max = chip1.get_bounds()
    x2_min, y2_min, x2_max, y2_max = chip2.get_bounds()

    center1_x = (x1_min + x1_max) / 2.0
    center1_y = (y1_min + y1_max) / 2.0
    center2_x = (x2_min + x2_max) / 2.0
    center2_y = (y2_min + y2_max) / 2.0

    return (abs(center2_x - center1_x) + abs(center2_y - center1_y)) * wire_count


def _iter_connection_records(problem: LayoutProblem) -> List[Tuple[str, str, int]]:
    records: List[Tuple[str, str, int]] = []
    all_conns = getattr(problem, "all_connections", []) or []

    if all_conns:
        for conn in all_conns:
            if not isinstance(conn, dict):
                continue
            chip1_id = conn.get("node1") or conn.get("source") or conn.get("from")
            chip2_id = conn.get("node2") or conn.get("target") or conn.get("to")
            if chip1_id is None or chip2_id is None:
                continue
            wire_count = int(float(conn.get("wireCount", conn.get("weight", 1))))
            records.append((chip1_id, chip2_id, wire_count))
        return records

    for chip1_id, chip2_id in problem.connection_graph.edges():
        edge_data = problem.connection_graph[chip1_id][chip2_id]
        wire_count = int(float(edge_data.get("wireCount", edge_data.get("weight", 1))))
        records.append((chip1_id, chip2_id, wire_count))

    return records


def calculate_iccad23_wirelength(layout: Dict[str, Chiplet], problem: LayoutProblem) -> Tuple[float, float, float, int]:
    """Return ICCAD23 total wirelength split into EMIB and normal parts.

    Connections with wireCount > 255 are treated as EMIB, matching
    baseline/ICCAD23/src/wirelength.py.
    """
    if not layout:
        return 0.0, 0.0, 0.0, 0

    emib_wirelength = 0.0
    normal_wirelength = 0.0
    total_wire_count = 0

    for chip1_id, chip2_id, wire_count in _iter_connection_records(problem):
        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        if chip1 is None or chip2 is None:
            continue

        total_wire_count += wire_count
        if wire_count > 255:
            emib_wirelength += calculate_emib_wirelength(chip1, chip2, wire_count)
        else:
            normal_wirelength += calculate_normal_wirelength(chip1, chip2, wire_count)

    return emib_wirelength + normal_wirelength, emib_wirelength, normal_wirelength, total_wire_count


def calculate_layout_utilization(layout: Dict[str, Chiplet]) -> Tuple[float, float, float, float, float]:
    """Return occupied-area utilization, including microbump halos."""
    if not layout:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    chiplets = list(layout.values())
    x_min = min(chip.x - float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    y_min = min(chip.y - float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    x_max = max(chip.x + chip.width + float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    y_max = max(chip.y + chip.height + float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    bbox_area = bbox_width * bbox_height
    chip_total_area = sum(
        (chip.width + 2.0 * float(getattr(chip, "hubump", 0.0) or 0.0))
        * (chip.height + 2.0 * float(getattr(chip, "hubump", 0.0) or 0.0))
        for chip in chiplets
    )
    utilization = (chip_total_area / bbox_area * 100) if bbox_area > 0 else 0.0

    return utilization, bbox_area, chip_total_area, bbox_width, bbox_height


def save_layout_to_json(layout: Dict[str, Chiplet], json_path: str) -> None:
    data = {
        "chiplets": [
            {
                "id": chip.id,
                "width": chip.width,
                "height": chip.height,
                "x": chip.x,
                "y": chip.y,
                "power": getattr(chip, "power", 0.0),
                "hubump": float(getattr(chip, "hubump", 0.0) or 0.0),
            }
            for chip in layout.values()
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_emib_types(emib_json_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load EMIB type definitions used by the ICCAD23 JSON exporter."""
    if emib_json_path is None:
        candidates = [
            Path(__file__).resolve().parent / "examples" / "EMIB.json",
            Path("examples") / "EMIB.json",
        ]
        emib_json_path = next((str(path) for path in candidates if path.exists()), None)

    defaults = {
        "interfaceA": {"LinearIODensity": 1200, "max_Reach_length": 1},
        "interfaceB": {"LinearIODensity": 300, "max_Reach_length": 5},
        "interfaceC": {"LinearIODensity": 40, "max_Reach_length": 100},
    }
    if emib_json_path is None:
        return defaults

    try:
        with open(emib_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = {}
        for item in data.get("EMIBTypes", []):
            name = item.get("name")
            if name:
                loaded[name] = item
        return loaded or defaults
    except Exception:
        return defaults


def _iter_export_connections(problem: LayoutProblem) -> List[Dict[str, Any]]:
    all_connections = getattr(problem, "all_connections", []) or []
    if all_connections:
        return [conn for conn in all_connections if isinstance(conn, dict)]

    connections = []
    for chip1_id, chip2_id in problem.connection_graph.edges():
        edge_data = problem.connection_graph[chip1_id][chip2_id]
        connections.append(
            {
                "node1": chip1_id,
                "node2": chip2_id,
                "wireCount": edge_data.get("wireCount", edge_data.get("weight", 1)),
                "EMIBType": edge_data.get("EMIBType", "interfaceB"),
                "EMIB_length": edge_data.get("EMIB_length", 0.8533),
                "EMIB_max_width": edge_data.get("EMIB_max_width", 3.0),
                "EMIB_bump_width": edge_data.get("EMIB_bump_width", 1.0),
            }
        )
    return connections


def _export_rotation(chip: Chiplet, problem: Optional[LayoutProblem]) -> int:
    rotation = getattr(chip, "rotation", None)
    if rotation is not None:
        return int(rotation)
    if problem is None or chip.id not in problem.chiplets:
        return 0

    original = problem.chiplets[chip.id]
    if abs(chip.width - original.height) < 1e-9 and abs(chip.height - original.width) < 1e-9:
        return 1
    return 0


def _build_emib_connection_exports(
    layout: Dict[str, Chiplet],
    problem: LayoutProblem,
    emib_json_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    connections = []
    emib_types = load_emib_types(emib_json_path)

    for conn in _iter_export_connections(problem):
        chip1_id = conn.get("node1") or conn.get("source") or conn.get("from")
        chip2_id = conn.get("node2") or conn.get("target") or conn.get("to")
        if chip1_id is None or chip2_id is None:
            continue

        wire_count = int(float(conn.get("wireCount", conn.get("weight", 1))))
        if wire_count <= 255:
            continue

        chip1 = layout.get(chip1_id)
        chip2 = layout.get(chip2_id)
        if chip1 is None or chip2 is None:
            continue

        is_adjacent, _, direction = get_adjacency_info(chip1, chip2)
        if not is_adjacent:
            continue  # Free placement does not imply a physical silicon bridge.
        bridge_center_x, bridge_center_y = get_bridge_center(chip1, chip2, direction)
        emib_rotation = 1 if direction in ("left", "right") else 0
        emib_type = conn.get("EMIBType", "interfaceB")
        emib_length = float(conn.get("EMIB_length", 0.8533))
        emib_bump_width = float(conn.get("EMIB_bump_width", 1.0))
        emib_width = emib_bump_width * 2.0

        if emib_rotation == 1:
            emib_x = bridge_center_x - emib_width / 2.0
            emib_y = bridge_center_y - emib_length / 2.0
        else:
            emib_x = bridge_center_x - emib_length / 2.0
            emib_y = bridge_center_y - emib_width / 2.0

        emib_info = emib_types.get(emib_type, emib_types.get("interfaceB", {}))
        connections.append(
            {
                "node1": chip1_id,
                "node2": chip2_id,
                "EMIBType": emib_type,
                "EMIB_length": emib_length,
                "EMIB_max_width": emib_info.get("max_Reach_length", conn.get("EMIB_max_width", 3.0)),
                "EMIB_width": emib_width,
                "EMIB_bump_width": emib_bump_width,
                "EMIB-x-position": float(emib_x),
                "EMIB-y-position": float(emib_y),
                "EMIB-rotation": emib_rotation,
            }
        )

    return connections


def export_layout_result_json(
    layout: Dict[str, Chiplet],
    problem: LayoutProblem,
    json_path: str,
    metrics: Optional[Dict[str, Any]] = None,
    emib_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export layout in the ICCAD23/BT-tree result schema with EMIB bridge positions."""
    data: Dict[str, Any] = {"chiplets": []}

    for chip_id, chip in layout.items():
        source_chip = problem.chiplets.get(chip_id) if problem is not None else None
        chip_data = {
                "name": chip.id,
                "x-position": float(chip.x),
                "y-position": float(chip.y),
                "width": float(chip.width),
                "height": float(chip.height),
                "rotation": _export_rotation(chip, problem),
                "power": float(getattr(chip, "power", getattr(source_chip, "power", 0.0) if source_chip else 0.0) or 0.0),
            }
        hubump = float(getattr(chip, "hubump", 0.0) or 0.0)
        if hubump > 0.0:
            chip_data.update({
                "hubump": hubump,
                "occupied-x-position": float(chip.x - hubump),
                "occupied-y-position": float(chip.y - hubump),
                "occupied-width": float(chip.width + 2.0 * hubump),
                "occupied-height": float(chip.height + 2.0 * hubump),
            })
        data["chiplets"].append(chip_data)

    data["connections"] = _build_emib_connection_exports(layout, problem, emib_json_path)

    if metrics:
        wirelength = metrics.get("rlplanner_total_wirelength", metrics.get("wirelength"))
        area = metrics.get("bbox_area", metrics.get("bounding_rect_area"))
        if wirelength is not None:
            data["wirelength"] = wirelength
        if area is not None:
            data["area"] = area
        bbox_height = metrics.get("bbox_height")
        if metrics.get("bbox_width") is not None and bbox_height not in (None, 0):
            data["aspect_ratio"] = round(float(metrics["bbox_width"]) / float(bbox_height), 2)
        elif layout:
            _, _, _, bbox_width, bbox_height = calculate_layout_utilization(layout)
            data["aspect_ratio"] = round(bbox_width / bbox_height, 2) if bbox_height else 1.0
    else:
        try:
            total_wirelength, _, _, _ = calculate_iccad23_wirelength(layout, problem)
            data["wirelength"] = total_wirelength
        except Exception:
            pass
        try:
            _, bbox_area, _, bbox_width, bbox_height = calculate_layout_utilization(layout)
            data["area"] = bbox_area
            data["aspect_ratio"] = round(bbox_width / bbox_height, 2) if bbox_height else 1.0
        except Exception:
            pass

    output_path = Path(json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def generate_color(chip_id: str):
    random.seed(hash(chip_id))
    return (
        random.uniform(0.3, 0.9),
        random.uniform(0.3, 0.9),
        random.uniform(0.3, 0.9),
    )


def visualize_layout_with_bridges(
    layout: Dict[str, Chiplet],
    problem: LayoutProblem,
    output_file: str = "layout_with_bridges.png",
    show_bridges: bool = True,
    show_coordinates: bool = True,
) -> None:
    """Visualize chiplets and directly adjacent EMIB bridge regions."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed; skip layout visualization")
        return

    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    if not layout:
        print("Error: no chiplet data")
        return

    chiplets = list(layout.values())
    x_min = min(chip.x - float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    y_min = min(chip.y - float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    x_max = max(chip.x + chip.width + float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    y_max = max(chip.y + chip.height + float(getattr(chip, "hubump", 0.0) or 0.0) for chip in chiplets)
    margin = max((x_max - x_min), (y_max - y_min), 1.0) * 0.1

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for chip_id, chip in layout.items():
        hubump = float(getattr(chip, "hubump", 0.0) or 0.0)
        if hubump > 0.0:
            bump_rect = Rectangle(
                (chip.x - hubump, chip.y - hubump),
                chip.width + 2.0 * hubump,
                chip.height + 2.0 * hubump,
                linewidth=1.5,
                edgecolor="darkorange",
                facecolor="orange",
                alpha=0.18,
                linestyle="--",
            )
            ax.add_patch(bump_rect)
        rect = Rectangle(
            (chip.x, chip.y),
            chip.width,
            chip.height,
            linewidth=2,
            edgecolor="black",
            facecolor=generate_color(chip_id),
            alpha=0.6,
            label=chip_id,
        )
        ax.add_patch(rect)

        center_x = chip.x + chip.width / 2
        center_y = chip.y + chip.height / 2
        ax.text(
            center_x,
            center_y,
            chip_id,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

        ax.text(center_x, chip.y - 1, f"{chip.width:g}x{chip.height:g}", ha="center", va="top", fontsize=9)
        if show_coordinates:
            ax.text(chip.x, chip.y + chip.height + 0.5, f"({chip.x:.1f}, {chip.y:.1f})", fontsize=8)

    bridge_count = 0
    if show_bridges:
        try:
            emib_connections = _build_emib_connection_exports(layout, problem)
            for conn in emib_connections:
                x_min_b = float(conn["EMIB-x-position"])
                y_min_b = float(conn["EMIB-y-position"])
                if int(conn["EMIB-rotation"]) == 1:
                    width_b = float(conn["EMIB_width"])
                    height_b = float(conn["EMIB_length"])
                else:
                    width_b = float(conn["EMIB_length"])
                    height_b = float(conn["EMIB_width"])

                ax.add_patch(
                    Rectangle(
                        (x_min_b, y_min_b),
                        width_b,
                        height_b,
                        linewidth=2,
                        edgecolor="red",
                        facecolor="yellow",
                        alpha=0.5,
                        linestyle="--",
                    )
                )

                ax.text(
                    x_min_b + width_b / 2,
                    y_min_b + height_b / 2,
                    f"{conn['node1']}-{conn['node2']}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="red",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="red"),
                )
                bridge_count += 1
        except Exception as exc:
            print(f"Warning: failed to render bridges - {exc}")

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    title = f"Chiplet Layout Visualization ({len(layout)} chiplets"
    if show_bridges:
        title += f", {problem.connection_graph.number_of_edges()} connections"
    title += ")"
    ax.set_title(title, fontsize=14, fontweight="bold")

    legend_text = "Legend:\n"
    legend_text += "• Black border = Chiplet boundary\n"
    legend_text += "• Semi-transparent fill = Chiplet area\n"
    legend_text += "• Orange dashed border = Microbump occupied boundary"
    if show_bridges:
        legend_text += "\n• Red dashed box = Silicon bridge area\n"
        legend_text += "• Yellow semi-transparent = Bridge occupancy"
    ax.text(
        0.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nLayout visualization saved to: {output_path}")
    print(f"  - Number of chiplets: {len(layout)}")
    print(f"  - Number of rendered bridges: {bridge_count}")
    print(f"  - Layout dimensions: {x_max - x_min:.1f} x {y_max - y_min:.1f}")
    plt.close()
