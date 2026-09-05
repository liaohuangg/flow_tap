#!/usr/bin/env python3
"""
Convert .cfg files to JSON format for chiplet placement.
"""

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_chiplets_json(
    json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the raw chiplet description JSON.

    Parameters
    ----------
    json_path:
        Path to the `chiplets.json` file. If None, uses relative path from src directory.

    Returns
    -------
    dict
        The parsed JSON object. Top-level keys are chiplet names.
    """
    if json_path is None:
        # 从 src 目录的相对路径: ../../benchmark/dummy-chiplet-input/chiplet_input/chiplets.json
        current_file = Path(__file__)
        src_dir = current_file.parent
        project_root = src_dir.parent
        json_path = project_root / "benchmark" / "dummy-chiplet-input" / "chiplet_input" / "chiplets.json"
        # 如果默认路径不存在，抛出错误提示用户提供路径
        if not json_path.exists():
            raise FileNotFoundError(
                f"默认的 chiplets.json 文件不存在: {json_path}\n"
                f"请使用 input_json_path 参数指定正确的 JSON 文件路径。"
            )

    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    return data


def build_chiplet_table(chiplets: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a simple table from the chiplet JSON.

    Each row in the table is a dict with the following keys:

    - ``name``: chiplet name (top-level key in the JSON)
    - ``dimensions``: the raw ``dimensions`` object for this chiplet
    - ``phys``: the raw ``phys`` list for this chiplet
    - ``power``: the ``power`` value for this chiplet

    Parameters
    ----------
    chiplets:
        Parsed JSON dict returned by :func:`load_chiplets_json`.

    Returns
    -------
    list of dict
        A list of rows; each row has keys ``name``, ``dimensions``, ``phys``,
        and ``power``.
    """

    table: List[Dict[str, Any]] = []

    for name, info in chiplets.items():
        row = {
            "name": name,
            "dimensions": info.get("dimensions", {}),
            "phys": info.get("phys", []),
            "power": info.get("power", None),
        }
        table.append(row)

    return table


def pretty_print_table(table: List[Dict[str, Any]]) -> None:
    """Pretty-print the chiplet table to the console.

    This is just for quick inspection / debug.
    """

    from pprint import pprint

    for row in table:
        pprint(row)

def parse_cfg_file(cfg_path):
    """
    Parse a .cfg file and extract chiplet information.
    
    Args:
        cfg_path: Path to the .cfg file
        
    Returns:
        dict with keys: widths, heights, powers, connections_matrix
    """
    with open(cfg_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract widths
    widths_match = re.search(r'widths\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not widths_match:
        raise ValueError(f"Could not find 'widths' in {cfg_path}")
    widths_str = widths_match.group(1).strip()
    widths = [float(x.strip()) for x in widths_str.split(',') if x.strip()]
    
    # Extract heights
    heights_match = re.search(r'heights\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not heights_match:
        raise ValueError(f"Could not find 'heights' in {cfg_path}")
    heights_str = heights_match.group(1).strip()
    heights = [float(x.strip()) for x in heights_str.split(',') if x.strip()]
    
    # Extract powers
    powers_match = re.search(r'powers\s*=\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if not powers_match:
        raise ValueError(f"Could not find 'powers' in {cfg_path}")
    powers_str = powers_match.group(1).strip()
    powers = [float(x.strip()) for x in powers_str.split(',') if x.strip()]
    
    # Extract connections matrix (may span multiple lines)
    # Find the connections section
    connections_match = re.search(r'connections\s*=\s*(.+?)(?=\n\n|\n\[|\n[a-z]+\s*=|$)', content, re.DOTALL)
    if not connections_match:
        raise ValueError(f"Could not find 'connections' in {cfg_path}")
    connections_str = connections_match.group(1).strip()
    
    # Parse the matrix: split by semicolon to get rows, then by comma to get values
    rows = []
    for row_str in connections_str.split(';'):
        row_str = row_str.strip()
        if not row_str:
            continue
        # Remove tabs and extra spaces, split by comma
        row = [int(x.strip()) for x in row_str.split(',') if x.strip()]
        if row:  # Only add non-empty rows
            rows.append(row)
    
    # Validate dimensions
    if len(widths) != len(heights) or len(widths) != len(powers):
        raise ValueError(f"Dimension mismatch: widths={len(widths)}, heights={len(heights)}, powers={len(powers)}")
    
    if len(rows) != len(widths):
        raise ValueError(f"Connections matrix rows ({len(rows)}) != chiplet count ({len(widths)})")
    
    for i, row in enumerate(rows):
        if len(row) != len(widths):
            raise ValueError(f"Connections matrix row {i} has {len(row)} columns, expected {len(widths)}")
    
    return {
        'widths': widths,
        'heights': heights,
        'powers': powers,
        'connections_matrix': rows
    }


def matrix_to_connections(connections_matrix, default_emib_type: str = "interfaceC"):
    """
    Convert adjacency matrix to list of connections.
    
    输出格式：每条边为无向边，包含 node1, node2, wireCount, EMIBType, EMIB_length, EMIB_max_width, EMIB_bump_width。
    wireCount 直接取自 cfg 的 connections 矩阵 connections_matrix[i][j]（chiplet i 与 j 之间的线数）。
    EMIBType 默认标注为 interfaceC，可用 update_connection.py 按范围重新标注。
    EMIB_length = wireCount / LinearIODensity，EMIB_max_width = max_Reach_length - 2*(wireCount/AreaIODensity)/EMIB_length，
    EMIB_bump_width = (wireCount / AreaIODensity) / EMIB_length。
    默认接口 interfaceC：LinearIODensity=40, max_Reach_length=100, AreaIODensity=80。
    
    Args:
        connections_matrix: 2D list representing adjacency matrix（对称矩阵，值为线数 wireCount）
        default_emib_type: 默认 EMIBType 标注（默认 "interfaceC"）
        
    Returns:
        List of dicts: [{"node1": "A", "node2": "B", "wireCount": 200, "EMIBType": "interfaceC", "EMIB_length": ..., "EMIB_max_width": ..., "EMIB_bump_width": ...}, ...]
    """
    linear_io = 40
    max_reach = 100.0
    area_io = 80.0

    connections = []
    num_chiplets = len(connections_matrix)
    
    def index_to_name(idx):
        return chr(ord('A') + idx)
    
    for i in range(num_chiplets):
        for j in range(i + 1, num_chiplets):
            wire_count = int(connections_matrix[i][j])
            if wire_count > 0:
                emib_length = wire_count / linear_io if linear_io > 0 else 2.5
                emib_max_width = max_reach - 2 * (wire_count / area_io) / emib_length if emib_length > 0 else max_reach
                emib_bump_width = (wire_count / area_io) / emib_length if emib_length > 0 and area_io > 0 else 0.0
                connections.append({
                    "node1": index_to_name(i),
                    "node2": index_to_name(j),
                    "wireCount": wire_count,
                    "EMIBType": default_emib_type,
                    "EMIB_length": round(emib_length, 4),
                    "EMIB_max_width": round(emib_max_width, 4),
                    "EMIB_bump_width": round(emib_bump_width, 4),
                })
    
    return connections


def cfg_to_json(cfg_path, output_dir, default_emib_type: str = "interfaceC"):
    """
    Convert a .cfg file to JSON format.
    
    Args:
        cfg_path: Path to input .cfg file
        output_dir: Directory to save the JSON file
        default_emib_type: 默认 EMIBType 标注（默认 "interfaceC"）
    """
    print(f"Processing: {cfg_path}")
    
    # Parse the .cfg file
    data = parse_cfg_file(cfg_path)
    
    # Create chiplets list
    chiplets = []
    for i in range(len(data['widths'])):
        chiplet_name = chr(ord('A') + i)  # 0->A, 1->B, 2->C, ...
        chiplets.append({
            'name': chiplet_name,
            'width': data['widths'][i],
            'height': data['heights'][i],
            'power': int(data['powers'][i])  # Power is typically an integer
        })
    
    # Convert connections matrix to list of {node1, node2, wireCount, EMIBType}
    connections = matrix_to_connections(data['connections_matrix'], default_emib_type=default_emib_type)
    
    # Create JSON structure
    json_data = {
        'chiplets': chiplets,
        'connections': connections
    }
    
    # Generate output filename
    cfg_name = Path(cfg_path).stem  # Get filename without extension
    output_path = os.path.join(output_dir, f"{cfg_name}.json")
    
    # Write JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  -> Saved to: {output_path}")
    print(f"  -> Chiplets: {len(chiplets)}, Connections: {len(connections)}")
    return output_path


def main():
    """Main function to process all .cfg files."""
    import argparse
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(
        description="Convert .cfg files to JSON. connections 输出为 {node1, node2, wireCount, EMIBType} 格式，EMIBType 默认 interfaceC。"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="../Dataset/config",
        help=".cfg 文件所在目录（相对本脚本所在目录，默认: ../Dataset/config）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="../Dataset/dataset/input_test",
        help="输出 JSON 的目录（相对本脚本所在目录，默认: ../Dataset/dataset/input_test）",
    )
    parser.add_argument(
        "--start-i",
        type=int,
        default=None,
        help="仅处理 system_{i}.cfg 中编号 >= start-i 的文件（含端点）。默认不过滤。",
    )
    parser.add_argument(
        "--end-j",
        type=int,
        default=None,
        help="仅处理 system_{j}.cfg 中编号 <= end-j 的文件（含端点）。默认不过滤。",
    )
    args = parser.parse_args()
    
    config_dir = (script_dir / args.config_dir).resolve()
    output_dir = (script_dir / args.output_dir).resolve()
    
    os.makedirs(output_dir, exist_ok=True)
    cfg_files = list(Path(config_dir).glob('*.cfg'))

    # Optional filtering by system_{k}.cfg index
    if args.start_i is not None or args.end_j is not None:
        pat = re.compile(r"system_(\d+)\.cfg$")
        filtered = []
        skipped_non_system = 0
        skipped_out_of_range = 0
        for p in cfg_files:
            m = pat.match(p.name)
            if not m:
                skipped_non_system += 1
                continue
            idx = int(m.group(1))
            if args.start_i is not None and idx < int(args.start_i):
                skipped_out_of_range += 1
                continue
            if args.end_j is not None and idx > int(args.end_j):
                skipped_out_of_range += 1
                continue
            filtered.append(p)

        cfg_files = filtered
        print(
            f"Filter enabled: start-i={args.start_i} end-j={args.end_j} -> {len(cfg_files)} file(s) selected "
            f"(skipped_non_system={skipped_non_system}, skipped_out_of_range={skipped_out_of_range})"
        )
    
    if not cfg_files:
        print(f"No .cfg files found in {config_dir}")
        return
    
    print(f"Found {len(cfg_files)} .cfg file(s) to process")
    print(f"EMIBType 默认: interfaceC\n")
    
    success_count = 0
    error_count = 0
    
    for cfg_path in sorted(cfg_files):
        try:
            cfg_to_json(str(cfg_path), output_dir, default_emib_type="interfaceC")
            success_count += 1
        except Exception as e:
            print(f"ERROR processing {cfg_path}: {e}")
            error_count += 1
        print()
    
    # Summary
    print("=" * 60)
    print(f"Processing complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

# python /root/placement/flow_GCN/Dataset/dataset/input_preprocess.py  --config-dir ../config --output-dir input_test --start-i 65999 --end-j 100000       