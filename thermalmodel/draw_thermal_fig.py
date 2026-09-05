import os
from pathlib import Path
from typing import Optional


def _configure_matplotlib_backend() -> None:
    # Avoid Qt backend crashes in headless env.
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except Exception:
        pass


_configure_matplotlib_backend()


def _project_root() -> str:
    # thermalmodel/draw_thermal_fig.py -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_chiplet_rects_mm(flp_path: str):
    """Return chiplet rectangles from HotSpot FLP in mm.

    FLP line: name width height x y (meters). We exclude TIM blocks (names starting with 'T').
    Returns list of (name, w_mm, h_mm, x_mm, y_mm).
    """
    import re

    rects = []
    p = Path(flp_path)
    if not p.is_file():
        return rects

    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = re.split(r"\s+", s)
        if len(parts) < 5:
            continue
        name = parts[0]
        if name.startswith("T"):
            continue
        try:
            w_m, h_m, x_m, y_m = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        rects.append((name, w_m * 1000.0, h_m * 1000.0, x_m * 1000.0, y_m * 1000.0))
    return rects


def _hotspot_cmap():
    from matplotlib.colors import ListedColormap

    palette_rgb = [
        (255, 0, 0),
        (255, 51, 0),
        (255, 102, 0),
        (255, 153, 0),
        (255, 204, 0),
        (255, 255, 0),
        (204, 255, 0),
        (153, 255, 0),
        (102, 255, 0),
        (51, 255, 0),
        (0, 255, 0),
        (0, 255, 51),
        (0, 255, 102),
        (0, 255, 153),
        (0, 255, 204),
        (0, 255, 255),
        (0, 204, 255),
        (0, 153, 255),
        (0, 102, 255),
        (0, 51, 255),
        (0, 0, 255),
    ]
    palette_rgb = list(reversed(palette_rgb))
    palette_norm = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in palette_rgb]
    return ListedColormap(palette_norm, name="hotspot_grid_palette")


def plot_thermal_grid_overlay(
    flp_path: str,
    grid: "object",
    output_image: str,
    *,
    title: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap_name: str = "hotspot",
) -> None:
    """Plot a 2D thermal grid with chiplet rectangle overlay.

    - Does NOT draw EMIB.
    - grid is expected shape (H,W) or (1,H,W); accepts numpy array or torch tensor.
    - Temperature unit is fixed to Celsius (°C).
    """
    import numpy as np
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.patheffects import withStroke

    # grid -> numpy (H,W)
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    grid = np.asarray(grid)
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]

    chiplets = _load_chiplet_rects_mm(flp_path)

    # Compute layout bounding box (mm). We draw in a square canvas whose side is the
    # longest edge of the layout bounding box, and pad equally on both sides so the
    # chiplet layout is centered (left/right and up/down).
    min_x = min_y = 0.0
    max_x = max_y = 0.0
    if chiplets:
        min_x = min(x for _n, _w, _h, x, _y in chiplets)
        min_y = min(y for _n, _w, _h, _x, y in chiplets)
        max_x = max(x + w for _n, w, _h, x, _y in chiplets)
        max_y = max(y + h for _n, _w, h, _x, y in chiplets)

    total_w = max_x - min_x
    total_h = max_y - min_y
    if total_w <= 0 or total_h <= 0:
        total_w = total_h = float(max(grid.shape[-1], grid.shape[-2]))
        min_x = min_y = 0.0
        max_x = total_w
        max_y = total_h

    side = max(total_w, total_h)
    pad_x = (side - total_w) / 2.0
    pad_y = (side - total_h) / 2.0

    x0, x1 = (min_x - pad_x), (max_x + pad_x)
    y0, y1 = (min_y - pad_y), (max_y + pad_y)

    if cmap_name == "hotspot":
        cmap = _hotspot_cmap()
    else:
        cmap = cmap_name

    fig, ax = plt.subplots(1, figsize=(10, 8))
    im = ax.imshow(
        np.flipud(grid),
        cmap=cmap,
        extent=(x0, x1, y0, y1),
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )



    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Temperature (°C)", fontsize=18)
    cbar.ax.tick_params(labelsize=14)

    if title:
        ax.set_title(title, fontsize=18)
    ax.set_xlabel("X (mm)", fontsize=18)
    ax.set_ylabel("Y (mm)", fontsize=18)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.tick_params(axis="both", labelsize=14)

    # annotate max/avg
    try:
        gmax = float(np.max(grid))
        gavg = float(np.mean(grid))
        info = f"Max={gmax:.2f} °C, AVG={gavg:.2f} °C"
        ax.text(
            0.01,
            0.99,
            info,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            color="white",
            path_effects=[withStroke(linewidth=3, foreground="black")],
        )
    except Exception:
        pass

    for name, w, h, x, y in chiplets:
        rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor="black", facecolor="none")
        ax.add_patch(rect)
        cx, cy = x + w / 2.0, y + h / 2.0
        ax.text(
            cx,
            cy,
            name,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    out = Path(output_image)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
