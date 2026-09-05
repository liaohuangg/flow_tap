#!/usr/bin/env python3
"""绘制 128×128 热图 / 功耗图, 用于检验 HotSpot 热仿真结果。

数据源(自动识别):
  - *.csv         : "idx,value" 逗号格式(1-based), 已经是显示单位(℃ / W),
                    且 row 0 = 底部(bottom-origin)。即本工程生成的
                    thermal_map/system_temp_{i}_{j}.csv 与 power_map/system_power_{i}_{j}.csv。
  - *.grid.steady : HotSpot 扁平单层输出 "<idx>\\t<temp_K>", 自动垂直翻转并 K -> ℃。

覆盖芯片布局: 从 --flp 读取 Chiplet_* / C* 矩形(米), 叠加黑色边框, 并据此确定物理范围(mm)。

用法:
  python draw_thermal_map.py --input thermal_map/system_temp_300001_0.csv \\
      --flp config/system_300001_config/system_300001L4_ChipLayer.flp --out t.png --unit C
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def _configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


_configure_matplotlib()

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patheffects import withStroke

GRID = 128


def _as_square(arr: np.ndarray, src: str) -> np.ndarray:
    n = int(round(arr.size ** 0.5))
    if n * n != arr.size:
        raise RuntimeError(f"{src}: {arr.size} 个值无法构成方阵")
    return arr.reshape(n, n)


def read_csv_grid(path: Path) -> np.ndarray:
    """读 idx,value CSV(1-based), 已是显示单位 + bottom-origin。"""
    vals: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) >= 2:
            vals.append(float(parts[1]))
    return _as_square(np.asarray(vals, dtype=np.float64), str(path))


def read_grid_steady(path: Path, to_celsius: bool) -> np.ndarray:
    """读 HotSpot 扁平 grid.steady (row 0 = 顶部), 垂直翻转为 bottom-origin。"""
    vals: list[float] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            vals.append(float(parts[1]))
    arr = _as_square(np.asarray(vals, dtype=np.float64), str(path))
    arr = np.flipud(arr)  # HotSpot row0=top -> row0=bottom
    if to_celsius:
        arr = arr - 273.15
    return arr


def _is_chiplet(name: str) -> bool:
    return name.startswith("Chiplet") or re.fullmatch(r"C\d+", name) is not None


def read_flp(path: Path) -> tuple[list[tuple[str, float, float, float, float]], tuple | None]:
    """读 FLP(米) -> (chiplets[(name,w,h,x,y)_mm], extent (minx,maxx,miny,maxy)_mm 覆盖所有块)。"""
    chiplets: list[tuple[str, float, float, float, float]] = []
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    seen = False
    if path.is_file():
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 5:
                continue
            name = parts[0]
            try:
                w, h, x, y = (float(parts[k]) * 1000.0 for k in (1, 2, 3, 4))
            except ValueError:
                continue
            seen = True
            if _is_chiplet(name):
                chiplets.append((name, w, h, x, y))
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
    if not seen or max_x <= min_x or max_y <= min_y:
        return chiplets, None
    return chiplets, (min_x, max_x, min_y, max_y)


def _unit_label(unit: str) -> str:
    u = (unit or "C").strip().upper()
    return {"C": "°C", "K": "K", "W": "W"}.get(u, u)


def plot(input_path: Path, flp_path: Path | None, output: Path, unit: str,
         title: str | None, show_names: bool) -> None:
    text = input_path.read_text(encoding="utf-8", errors="replace")
    first = next((l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith("#")), "")
    is_csv = "," in first

    want_c = _unit_label(unit) == "°C"
    if is_csv:
        data = read_csv_grid(input_path)
    else:
        data = read_grid_steady(input_path, to_celsius=want_c)

    chiplets, extent = ([], None)
    if flp_path is not None:
        chiplets, extent = read_flp(flp_path)
    if extent is None:
        n = data.shape[0]
        extent = (0.0, float(n), 0.0, float(n))
    min_x, max_x, min_y, max_y = extent

    cmap = "jet"  # blue=cold -> red=hot, 与 HotSpot 经典配色一致
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(data, extent=extent, origin="lower", aspect="equal", cmap=cmap)
    im.set_clim(float(np.min(data)), float(np.max(data)))

    for name, w, h, x, y in chiplets:
        ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=1.0,
                                       edgecolor="black", facecolor="none"))
        if show_names:
            ax.text(x + w / 2.0, y + h / 2.0, name, ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white",
                    path_effects=[withStroke(linewidth=2, foreground="black")])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"value ({_unit_label(unit)})", fontsize=14)

    info = f"Max={float(np.max(data)):.2f}, AVG={float(np.mean(data)):.2f}"
    ax.text(0.01, 0.99, info, transform=ax.transAxes, ha="left", va="top",
            fontsize=13, color="white", path_effects=[withStroke(linewidth=3, foreground="black")])

    ax.set_title(title or f"{input_path.name} ({_unit_label(unit)})", fontsize=14)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot] saved: {output}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "--steady", dest="input", required=True,
                   help="输入: *.csv (idx,value) 或 *.grid.steady (扁平, K)")
    p.add_argument("--flp", default=None,
                   help="FLP 文件(米), 用于叠加 chiplet 边框并确定范围; 推荐 L4_ChipLayer.flp")
    p.add_argument("--out", required=True, help="输出 PNG 路径")
    p.add_argument("--unit", default="C", help="显示单位: C / K / W (默认 C)")
    p.add_argument("--title", default=None, help="图标题")
    p.add_argument("--show-names", default="0", help="是否标注 chiplet 名字: 1/0")
    args = p.parse_args(argv)

    show_names = str(args.show_names).strip().lower() in {"1", "true", "yes", "y", "on"}
    plot(Path(args.input), Path(args.flp) if args.flp else None,
         Path(args.out), args.unit, args.title, show_names)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
