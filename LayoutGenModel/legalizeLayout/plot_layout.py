"""合法化前后的布局对比图。

画法沿用仓库既有约定 (``gen_dataset/draw_thermal_map.py``): Agg 后端, ``patches.Rectangle``
画矩形, chiplet 名字用带黑色描边的白字标在中心, 轴标 X/Y (mm), 保存走
``savefig(dpi=150, bbox_inches="tight", facecolor="white")``。

图内文字一律用**英文**: 这套环境里 matplotlib 找不到任何中文字体 (fc-list 里没有,
``fontManager.ttflist`` 里也没有), 中文会render成一排豆腐块 —— 而且
``draw_thermal_map.py`` 那个被本文件对齐的模板本身也只用英文。控制台输出仍然是中文,
那是终端在渲染, 不受影响。

两边**共享同一组坐标轴** —— 这是这个图唯一的要点: 分开 autoscale 会让"位移很小"
和"位移很大"看起来一样, 对比就失去意义。

图上标什么
----------
* 重叠的 chiplet **涂红**, 越深重叠面积越大; 合法的一侧不涂。
* 越出画布的 chiplet 用**紫色虚线**边框圈出来。
* 连线按 wireCount 画成灰线, 线长优化的效果直接看得见 (线越密越短)。
* 每侧标题带该侧的读数 (重叠对数 / 峰值温度 / 神经线长 / 外接框面积), 于是
  三条判据的净变化在图上就能读, 不必再去翻报告。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)


_configure_matplotlib()

import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

__all__ = ["plot_comparison"]

# 单侧宽度 (inch)。两侧拼起来约 16 inch, 12 个 case 的图并排看也够清楚。
_PANEL_W, _PANEL_H = 8.0, 7.0
_DPI = 150


def _overlap_severity(lower, sizes, geom_tol):
    """每个 chiplet 参与的最深重叠深度 (mm)。0 = 不参与任何重叠。

    用**最深**而不是总面积: 深 5mm 的窄条重叠比一片 0.05mm 的贴边严重得多,
    而总面积会把它平均掉。
    """
    V = lower.shape[0]
    upper = lower + sizes
    lo = np.maximum(lower[:, None, :], lower[None, :, :])
    hi = np.minimum(upper[:, None, :], upper[None, :, :])
    depth = np.where((hi - lo) > geom_tol, hi - lo, 0.0).min(axis=-1)  # (V,V) 逐对重叠的浅边
    np.fill_diagonal(depth, 0.0)
    return depth.max(axis=1)


def _draw_panel(ax, lower, sizes, names, edges, extent, title, subtitle=None,
                geom_tol=1e-6, canvas=None, hotspot=()):
    from matplotlib import colormaps

    severity = _overlap_severity(lower, sizes, geom_tol)
    worst = float(severity.max()) if severity.size else 0.0
    hotspot = set(int(i) for i in (hotspot or ()))

    # 先画连线, 再压矩形 —— 否则连线会盖在色块上面, 看起来很脏
    if edges:
        centers = lower + sizes * 0.5
        for src, dst, weight in edges:
            ax.plot(
                [centers[src, 0], centers[dst, 0]],
                [centers[src, 1], centers[dst, 1]],
                color="0.55", linewidth=0.6 + 0.5 * float(np.log1p(weight)), alpha=0.55, zorder=1,
            )

    cmap = colormaps["Reds"]
    limit = max(extent[2] - extent[0], extent[3] - extent[1])
    min_label = 0.022 * limit  # 小到这个尺寸以下的 chiplet 不再标名字, 免得糊成一团

    for i, name in enumerate(names):
        x, y = float(lower[i, 0]), float(lower[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])

        if worst > 0.0 and severity[i] > 0.0:
            # 深浅按该 chiplet 的最深重叠占全局最深的比例, 全局最深必为最深的红
            frac = min(1.0, float(severity[i]) / worst)
            face = cmap(0.30 + 0.65 * frac)
        else:
            face = (0.62, 0.79, 0.94, 0.85)  # 合法/无重叠: 淡蓝

        outside = False
        if canvas is not None:
            origin, side = canvas
            outside = bool(
                (lower[i] < origin - geom_tol).any()
                or (lower[i] + sizes[i] > origin + float(side) + geom_tol).any()
            )
        style = {"linestyle": (0, (4, 2)), "linewidth": 2.0, "edgecolor": "purple"} if outside else {
            "linewidth": 0.9, "edgecolor": "black"
        }
        if i in hotspot:
            # 热点: 橙色粗实线。用实线而不是虚线, 是要和"越出画布"的紫色虚线一眼区分开。
            style = {"linewidth": 3.0, "edgecolor": "darkorange", "linestyle": "-"}
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face, zorder=2, **style))

        if i in hotspot:
            ax.text(
                x + w / 2.0, y + h, "HOTSPOT", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="darkorange", zorder=4,
                path_effects=[withStroke(linewidth=2.5, foreground="white")],
            )

        if min(w, h) >= min_label or w * h >= (min_label * min_label):
            ax.text(
                x + w / 2.0, y + h / 2.0, str(name), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=3,
                path_effects=[withStroke(linewidth=2.0, foreground="black")],
            )

    if canvas is not None:
        origin, side = canvas
        ax.add_patch(
            patches.Rectangle(
                (float(origin[0]), float(origin[1])), float(side), float(side),
                facecolor="none", edgecolor="0.25", linewidth=1.4, linestyle=(0, (6, 3)), zorder=4,
            )
        )

    if subtitle:
        ax.text(0.015, 0.985, subtitle, transform=ax.transAxes, ha="left", va="top",
                fontsize=10, color="black", zorder=5,
                path_effects=[withStroke(linewidth=3, foreground="white")])

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.3, zorder=0)


def _fmt_delta(value, unit="", pct=False, nd=2):
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:+.{nd}f}%"
    return f"{value:+.{nd}f}{unit}"


def plot_comparison(
    out_path,
    names,
    lower_before,
    lower_after,
    sizes,
    edges=None,
    *,
    metrics_before=None,
    metrics_after=None,
    canvas=None,
    title=None,
    geom_tol=1e-6,
    hotspot=None,
):
    """画「输入 vs 合法化后」两联图并存成 PNG。返回输出路径。

    ``edges`` 是 ``[(src, dst, wireCount), ...]`` (下标索引); ``metrics_*`` 是含
    ``overlap_pairs`` / ``peak_temp_C`` / ``wirelength_neural_mm`` / ``bbox_area_mm2``
    的字典 (缺项就不标); ``canvas`` 是 ``(origin, side)`` 的 mm 画布。
    ``hotspot`` 是被冻结的热点 chiplet 下标 —— 橙色粗框标出, 两侧画的是同一组
    (热点是在合法化后的布局上定的), 这样"哪些被钉住了"在图上直接可读。
    """
    lower_before = np.asarray(lower_before, dtype=np.float64)
    lower_after = np.asarray(lower_after, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)

    lo = np.minimum(lower_before.min(axis=0), lower_after.min(axis=0))
    hi = np.maximum((lower_before + sizes).max(axis=0), (lower_after + sizes).max(axis=0))
    if canvas is not None:
        origin, side = canvas
        lo = np.minimum(lo, np.asarray(origin, dtype=np.float64))
        hi = np.maximum(hi, np.asarray(origin, dtype=np.float64) + float(side))
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.04 * float(max(span))
    extent = (lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad)

    fig, axes = plt.subplots(1, 2, figsize=(2 * _PANEL_W, _PANEL_H))

    def subtitle(node):
        if not node:
            return None
        parts = [
            f"overlaps: {node.get('overlap_pairs', '?')}",
            f"peak {node['peak_temp_C']:.1f}C" if node.get("peak_temp_C") is not None else None,
            f"neural WL {node['wirelength_neural_mm']:.0f}" if node.get("wirelength_neural_mm") is not None else None,
            f"bbox {node['bbox_area_mm2']:.0f} mm2" if node.get("bbox_area_mm2") is not None else None,
        ]
        return "   ".join(p for p in parts if p)

    _draw_panel(axes[0], lower_before, sizes, names, edges, extent,
                "Input (illegal)", subtitle(metrics_before), geom_tol, canvas, hotspot)
    _draw_panel(axes[1], lower_after, sizes, names, edges, extent,
                "Legalized", subtitle(metrics_after), geom_tol, canvas, hotspot)

    if metrics_before and metrics_after:
        dt = (
            metrics_after["peak_temp_C"] - metrics_before["peak_temp_C"]
            if metrics_after.get("peak_temp_C") is not None and metrics_before.get("peak_temp_C") is not None
            else None
        )
        dwl = (
            metrics_after["wirelength_neural_mm"] / metrics_before["wirelength_neural_mm"] - 1.0
            if metrics_after.get("wirelength_neural_mm") and metrics_before.get("wirelength_neural_mm")
            else None
        )
        dbbox = (
            metrics_after["bbox_area_mm2"] / metrics_before["bbox_area_mm2"] - 1.0
            if metrics_after.get("bbox_area_mm2") and metrics_before.get("bbox_area_mm2")
            else None
        )
        fig.suptitle(
            f"{title or Path(out_path).stem}      net change: "
            f"ΔT {_fmt_delta(dt, 'K')}   ΔWL {_fmt_delta(dwl, pct=True)}   "
            f"Δbbox {_fmt_delta(dbbox, pct=True)}",
            fontsize=13,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
