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

配色 (指定值, 见下面的常量)
---------------------------
* 普通 (合法) 芯粒      ``#B4DCFA``
* 参与重叠的芯粒        ``#F9B3A7``
* 热点                  **只用 ``#FF8021`` 粗边框**标出 (不写 "HOTSPOT" 字样, 字色也不变)
* 越出画布的芯粒        紫色虚线边框 (未指定, 沿用)

图上标什么
----------
* 重叠 / 不重叠**只分两类**, 不按重叠深度做渐变 —— 这张图要回答的是"哪几块有问题",
  一眼可读比色阶细腻重要; 重叠深度报告里逐对都给了, 不必压进配色。
* 芯粒用**数字**标 (1, 2, 3, ...), 不标 JSON 里的原名 —— 原名太长, 密版图里光是自己
  就比 chiplet 还宽。编号按**面积从小到大** (1 最小), 两联之间同一个数字就是同一个件。
* 连线按 wireCount 画成灰线, 线长优化的效果直接看得见 (线越密越短)。
* 净变化 (ΔT / ΔWL / Δbbox) 只在整图顶部一行; **每侧不挂 panel 标题、也不单独标读数** ——
  左右分工固定 (左=输入, 右=合法化后), 顶部那行已经说清了。
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

# 单侧高度 (inch)。宽度**不写死**, 按数据长宽比算 —— 见 ``_panel_width``。
_PANEL_H = 7.0
_PANEL_CHROME_W = 1.45   # 单侧宽度里留给 y 轴标签/刻度的余量 (inch)
_PANEL_CHROME_H = 0.95   # 高度里留给 x 轴标签/刻度的余量 (inch); 没有 panel 标题了, 比带标题时小
_DPI = 150

# 配色。三个是给定值, 改之前先确认。
_COLOR_CHIPLET = "#B4DCFA"   # 普通 (合法) 芯粒
_COLOR_OVERLAP = "#F9B3A7"   # 参与重叠的芯粒
_COLOR_HOTSPOT = "#FF8021"   # 热点: 只用这个色的粗边框标
_COLOR_OUTSIDE = "purple"    # 越出画布 (虚线, 未指定色值, 沿用)


def _panel_width(extent):
    """按数据长宽比算单侧该给多宽 (inch), 免得空白全挤在两联中间。

    为什么要算而不是写死: ``set_aspect("equal")`` 会把坐标盒缩成**数据的比例**, 而
    ``tight_layout`` 是按"等分图宽"分的 —— 图给宽了, 多出来的宽度不会变成坐标盒,
    而是整个变成两联之间的空白 (数据接近正方形时能空出近一半, 图上看着像断层)。
    所以宽度顺着数据的比例给: 坐标盒要多大, 图就只给多大, 剩下的交给标注余量。
    """
    aspect = (extent[2] - extent[0]) / max(extent[3] - extent[1], 1e-9)
    return max(4.2, aspect * (_PANEL_H - _PANEL_CHROME_H)) + _PANEL_CHROME_W


def _overlaps_mask(lower, sizes, geom_tol):
    """``(V,)`` bool: 该 chiplet 是否与别的 chiplet 有**实面积**重叠。

    口径与 ``verify.find_overlap_pairs`` 一致: 两个轴上的交叠宽度都要 > ``geom_tol``
    才算。只在一个轴上交叠 (贴边、擦角) 不算 —— 那在图上看起来是挨着, 不是压着。
    """
    upper = lower + sizes
    lo = np.maximum(lower[:, None, :], lower[None, :, :])
    hi = np.minimum(upper[:, None, :], upper[None, :, :])
    touch = ((hi - lo) > geom_tol).all(axis=-1)   # (V,V) 两轴都有正宽度 = 实面积重叠
    np.fill_diagonal(touch, False)
    return touch.any(axis=1)


def _text_size_data(ax, text, fontsize, weight="bold"):
    """字符串在当前数据坐标下占的 ``(宽, 高)`` (mm)。

    必须在 title/limits/aspect 全部定稿、并且 canvas 已经 draw 过一次之后调 ——
    "点数 -> 数据单位"的换算 (``transData``) 依赖坐标盒的**像素**尺寸, 在那之前量出来
    是错的; ``set_aspect("equal")`` 更是要等 draw 时才把坐标盒缩到数据比例上。
    """
    probe = ax.text(0.0, 0.0, text, fontsize=fontsize, fontweight=weight)
    bb = probe.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    probe.remove()
    (x0, y0), (x1, y1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    return abs(x1 - x0), abs(y1 - y0)


def _rects_hit(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _label_text(rank):
    """编号 -> 图上文字: **数字**, 从 1 开始 (1 = 面积最小的那个件)。

    用数字不用字母, 是因为字母到第 27 个就得进位成 AA/AB, 编号长度不齐、也容易和真正的
    名字混淆; 数字再多也只是多一位, 一直是"1, 2, 3, ..."这一种形式。
    """
    return str(int(rank) + 1)


def _place_labels(ax, texts, lower, sizes, drawable, fontsize):
    """贪心放标签, 尽量互不重叠。返回 ``{下标: (x, y)}``。

    **大件先放**: 它的名字放得下、也更该摆在正中, 让它先占位, 小件再在剩下的空间里挑
    (反过来会让小件把大件的正中心挤掉, 读起来最别扭)。

    每个标签只在**自己那一块**的框内和紧邻处取候选位置, 依次比较:
    与已放标签的碰撞数 -> 越出本块多远 -> 离本块中心多远。于是标签总是尽量贴着自己那一块,
    "这个名字属于哪块"不会看错; 实在避不开时才退而接受碰撞。
    """
    ax.set_autoscale_on(False)      # 后面加的 text 不要再去动坐标范围
    ax.figure.canvas.draw()         # 坐标盒定稿, 量出来的尺寸才算数

    size_of = {}
    for i in drawable:
        key = str(texts[i])
        if key not in size_of:
            size_of[key] = _text_size_data(ax, key, fontsize)

    placed, out = [], {}
    for i in sorted(drawable, key=lambda k: -float(sizes[k, 0] * sizes[k, 1])):
        lw, lh = size_of[str(texts[i])]
        x, y = float(lower[i, 0]), float(lower[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])
        cx, cy = x + w / 2.0, y + h / 2.0
        candidates = (
            (cx, cy),
            (cx, cy + 0.28 * h), (cx, cy - 0.28 * h),
            (cx, cy + 0.42 * h), (cx, cy - 0.42 * h),
            (cx, y + h + 0.58 * lh), (cx, y - 0.58 * lh),
            (x + w + 0.58 * lw, cy), (x - 0.58 * lw, cy),
        )
        best = None
        for px, py in candidates:
            rect = (px - lw / 2, py - lh / 2, px + lw / 2, py + lh / 2)
            hits = sum(1 for r in placed if _rects_hit(rect, r))
            spill = max(0.0, abs(px - cx) - w / 2) + max(0.0, abs(py - cy) - h / 2)
            key = (hits, spill, abs(px - cx) + abs(py - cy))
            if best is None or key < best[0]:
                best = (key, px, py, rect)
        _, px, py, rect = best
        placed.append(rect)
        out[int(i)] = (px, py)
    return out


def _draw_labels(ax, lower, sizes, extent, fontsize=11):
    """在坐标盒定稿之后画芯粒标记 (位置约束见 ``_place_labels``)。

    标的是**数字** (1, 2, 3, ...), 不是 JSON 里的原名 —— 原名 "Analog_1" 这种在密版图里
    光是自己就比 chiplet 还宽, 必然糊成一片。数字短, 既几乎不会互相压, 也让"同一个件在
    输入/合法化两联之间对号入座"变得一眼可做。

    编号按**面积从小到大**: 1 是最小的那个件。面积相同的按 JSON 数组序定先后 —— 必须有个
    确定的第二关键字, 否则同一份数据两次出图的编号可能不一样, 图就没法互相对照了。

    一律**黑字 + 白描边**, 热点件也一样 (热点只靠橙色**边框**标, 不靠字色)。底色
    ``#B4DCFA``/``#F9B3A7`` 很浅, 黑字对比已经够, 白描边是为了压在灰色连线上时不糊。

    小到这个尺寸以下的 chiplet 干脆不标: 标了只会盖住别人。
    """
    limit = max(extent[2] - extent[0], extent[3] - extent[1])
    min_label = 0.022 * limit
    drawable = [
        i for i in range(len(sizes))
        if min(sizes[i]) >= min_label or float(np.prod(sizes[i])) >= min_label ** 2
    ]
    if not drawable:
        return
    # 面积升序 -> 1 最小
    order = sorted(range(len(sizes)), key=lambda i: (float(np.prod(sizes[i])), i))
    texts = [""] * len(sizes)
    for rank, i in enumerate(order):
        texts[i] = _label_text(rank)
    for i, (px, py) in _place_labels(ax, texts, lower, sizes, drawable, fontsize).items():
        # 黑字 + 白描边: 底色 #B4DCFA / #F9B3A7 很浅, 黑字本身够清楚, 描边是为了压在灰
        # 连线上时不糊 (连线 zorder 更低, 是从字底下穿过去的)。
        ax.text(
            px, py, texts[i], ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="black", zorder=3,
            path_effects=[withStroke(linewidth=1.2, foreground="white")],
        )


def _draw_panel(ax, lower, sizes, names, edges, extent,
                geom_tol=1e-6, canvas=None, hotspot=()):
    overlapping = _overlaps_mask(lower, sizes, geom_tol)
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

    for i, name in enumerate(names):
        x, y = float(lower[i, 0]), float(lower[i, 1])
        w, h = float(sizes[i, 0]), float(sizes[i, 1])

        face = _COLOR_OVERLAP if overlapping[i] else _COLOR_CHIPLET

        outside = False
        if canvas is not None:
            origin, side = canvas
            outside = bool(
                (lower[i] < origin - geom_tol).any()
                or (lower[i] + sizes[i] > origin + float(side) + geom_tol).any()
            )
        style = {"linestyle": (0, (4, 2)), "linewidth": 2.0, "edgecolor": _COLOR_OUTSIDE} if outside else {
            "linewidth": 0.9, "edgecolor": "black"
        }
        if i in hotspot:
            # 热点: 橙色粗实线。用实线而不是虚线, 是要和"越出画布"的紫色虚线一眼区分开。
            style = {"linewidth": 3.0, "edgecolor": _COLOR_HOTSPOT, "linestyle": "-"}
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=face, zorder=2, **style))

    if canvas is not None:
        origin, side = canvas
        ax.add_patch(
            patches.Rectangle(
                (float(origin[0]), float(origin[1])), float(side), float(side),
                facecolor="none", edgecolor="0.25", linewidth=1.4, linestyle=(0, (6, 3)), zorder=4,
            )
        )

    # 不画 panel 标题: 左右分工是固定的 (左=输入, 右=合法化后), 顶部那行已经写明了,
    # 再各挂一个标题只是占地方。
    ax.set_xlabel("X (mm)", fontsize=15)
    ax.set_ylabel("Y (mm)", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
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

    # 宽度按数据比例给 (见 ``_panel_width``), 否则空白全部堆在两联中间。
    fig, axes = plt.subplots(1, 2, figsize=(2 * _panel_width(extent), _PANEL_H))

    _draw_panel(axes[0], lower_before, sizes, names, edges, extent,
                geom_tol, canvas, hotspot)
    _draw_panel(axes[1], lower_after, sizes, names, edges, extent,
                geom_tol, canvas, hotspot)

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
            fontsize=16,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # 标签最后放, 且必须等 tight_layout 定稿: 位置是按"文字在数据坐标下占多大"算的,
    # 而那个换算依赖坐标盒的最终像素尺寸 —— 布局一变就得重算, 提前算是白算。
    _draw_labels(axes[0], lower_before, sizes, extent)
    _draw_labels(axes[1], lower_after, sizes, extent)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path
