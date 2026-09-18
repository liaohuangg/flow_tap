#!/usr/bin/env python3
"""plot_pareto3d.py — 逐 case 画四方对比的三维帕累托前沿: 总线长 / 外接框面积 / 峰值温度。

三个目标全部越小越好, 所以 "支配" 定义为: p 三项都不比 q 差、且至少一项更好。
四个方法的解合在一起算一条全局前沿 (这批数据能达到的最优权衡), 谁落在前沿上
谁的图例里就记 "多少个 / 共多少个"。

四个方法 (图上用论文里的写法, 不用缩写):
    ATPlace2.5D   AT/50set.csv                            浅蓝 #5ECCF3
    RLPlanner     RL/result.csv                           绿   #81D31A
    MILP          ILP/result.csv                          靛蓝 #4E67C8
    TW-FM         FM/result_seed121_7case.csv             橙   #FF8021
                  (缺 case 时退到 FM/result_seed3_4case.csv)

数据已由 build_case_solution.py 按 case 归拢到 case_solution/<case>.csv, 本脚本只读它。

三轴默认 x=总线长, y=面积, z=峰值温度 (--axes 可循环轮换), 于是 "前沿" =
给定线长和面积预算下能拿到的最低峰温。前沿的弯曲形状随 case 变, 不是一律朝原点凹;
用 --elev/--azim/--box-aspect 换视角。

三个坐标轴用浅灰细线从墙角原点画出 (--axis-lines, 默认开): 墙角 = 左墙、右墙、
地板三块灰面交汇的那个内角 (不是盒子外棱! 外棱上那个角不是 "坐标原点" 的观感)。
matplotlib 3d 自带的轴线画在外棱上、位置随视角变, 所以这里关掉自带的,
显式从墙角沿三个轴各画到该轴另一端, 三条线一定交汇在同一个墙角。
墙角落在哪个数据点上随视角变, 由 room_corner() 按视线方向算。
三个立体背面仍是浅灰, 靠它撑立体感。

输出:
  fig/<case>_pareto3d.png                三维图
  parato_f/<case>_pareto_front.csv       落在全局前沿上的解 (表格视图, 便于核对数值)
用法:
  python plot_pareto3d.py                  # 画 case_solution 里全部 case
  python plot_pareto3d.py acend910         # 只画一个
  python plot_pareto3d.py hp11_m --elev 18 --azim -55
  python plot_pareto3d.py xerox8_m --lim front # 只按前沿取轴范围, 范围外的解不画
  python plot_pareto3d.py acend910 --dark      # 深色底版本
图里的文字一律英文 —— 本环境的 matplotlib 没有中文字体 (出图与中文控制台是两件事)。
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (注册 3d projection)

# 出图的字统一用 Calibri, 跟 PPT 里的正文一致。本机没装 Calibri, 但这是 WSL,
# 能直接读到 Windows 那份, 注册进 matplotlib 就行 —— 不把字体文件拷进仓库,
# 免得把 Windows 的字体散到别处去。找不到就退回 DejaVu Sans, 图照出, 只是字形不同。
CALIBRI_DIRS = [
    Path("/mnt/c/Windows/Fonts"),           # WSL: 宿主机的 Windows 字体
    Path("/usr/share/fonts/truetype/calibri"),
    Path("/usr/share/fonts/truetype/crosextra"),   # distro 的 Carlito 兜底
    Path.home() / ".local/share/fonts",
]
FONT_FALLBACK = "DejaVu Sans"


def setup_font() -> str:
    """注册 Calibri 并设成全局字体, 返回实际生效的字体名。"""
    from matplotlib import font_manager
    for d in CALIBRI_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("[Cc]alibri*.ttf")) + sorted(d.glob("[Cc]arlito*.ttf")):
            try:
                font_manager.fontManager.addfont(str(f))
            except Exception:
                pass        # 单个 ttf 坏了不影响其余的, 也不值得让出图失败
        if any(f.name in ("Calibri", "Carlito") for f in font_manager.fontManager.ttflist):
            name = next(f.name for f in font_manager.fontManager.ttflist
                        if f.name in ("Calibri", "Carlito"))
            matplotlib.rcParams["font.family"] = name
            return name
    matplotlib.rcParams["font.family"] = FONT_FALLBACK
    return FONT_FALLBACK


FONT = setup_font()

PF = Path("/root/placement/flow_tap/resultEval/pareto_front")
CASE_DIR = PF / "case_solution"
FIG_DIR = PF / "fig"
TAB_DIR = PF / "parato_f"

# 上下标直接写 Unicode 字符, 不用 mathtext: mathtext 走的是 matplotlib 自带那套
# 数学字体, 跟 Calibri 混在一行里字形和字重都对不上。
OBJECTIVES = [
    ("total_wirelength_mm", "Total Wirelength ({wl})"),
    ("bbox_area_mm2", "Bounding Box Area (mm²)"),
    ("max_temp_C", "Max Temperature (°C)"),
]
WL_SCALE = {"mm": 1.0, "m": 0.001}      # 线长在图上用 m 更顺眼; csv 表恒为 mm
AXIS_OF = {"wirelength": 0, "area": 1, "temp": 2}

# 四个方法按固定顺序取色, 不随筛选后的顺序变。
# MILP / RLPlanner 两色由用户指定; ATPlace2.5D / TW-FM 沿用原配色。
SERIES = {
    "ATPlace2.5D": "#5ECCF3",
    "RLPlanner": "#81D31A",
    "MILP": "#4E67C8",
    "TW-FM": "#FF8021",
}
ORDER = ["ATPlace2.5D", "RLPlanner", "MILP", "TW-FM"]

# 形状也带上方法身份。RLPlanner 的绿 #81D31A 和 TW-FM 的橙 #FF8021 在红绿色盲
# (deuteranopia) 下 OKLab 色差只剩 4.1, 光靠颜色分不开; 绿色是用户指定的不能动,
# 所以给 RLPlanner 换成三角形、MILP 换成方块, 色觉异常下靠形状也认得出来。
# (两个蓝色没问题: ATPlace2.5D vs MILP 各种色觉下色差都 >27)
MARKER = {"ATPlace2.5D": "o", "RLPlanner": "^", "MILP": "s", "TW-FM": "o"}

THEME = {
    # light 的画布 + 坐标区用纯白, 贴进白底论文不留灰边; 三个立体背面保持浅灰撑立体感。
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#dedcd4", "pane": "#f4f3ef", "axis": "#bfbdb5"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#3a3a37", "pane": "#232322", "axis": "#5e5d58"},
}
CTX_SIZE, CTX_ALPHA = 26.0, 0.55        # 被支配解: 小点淡显, 只做背景
FRONT_SIZE, FRONT_ALPHA = 78.0, 0.95    # 非支配解: 大点 + 表面色描边
STEM_ALPHA = 0.30                       # 垂线: 只做位置参照, 不能盖过点本身
AXIS_LW = 0.9                           # 墙角坐标轴: 细线, 只当参照, 别抢数据

# 画图顺序。computed_zorder=False 之后 zorder 就是唯一的次序依据, 全部写死在这里,
# 免得各处散着魔数、改一处忘了另一处。网格线和 pane 挂在轴 artist 上, matplotlib
# 默认给它们 1.5, 跟垂线一样高, 平局时按加入顺序画 —— 于是网格会横穿到点前面去,
# 所以轴整体压到 Z_AXES, 一定最先画。垂线要垫在点下面、网格上面。
Z_AXES, Z_STEM, Z_CTX, Z_FRONT, Z_CORNER = 0.0, 1.0, 2.0, 4.0, 1.2

# 字号与图例位置都从这里取, 单图和多图拼版共用一套代码, 只是传的值不同。
# legend_y 是图例在坐标区里的相对高度: 1.0 = 坐标区顶边, 3d 画出来的盒子占不满
# 坐标区, 顶边到盒子之间有一大块空白, 所以压到 0.9 附近图例才贴得住盒子。
FONTS = {
    # 磅数比 DejaVu 那套大 15% 左右: Calibri 的 x-height 小 (0.466 em vs DejaVu 的
    # 0.545), 同样磅数看着明显更小, 不补回来的话换字体等于把字又缩了一圈。
    "label": 17.0,          # 三个轴标题
    "tick": 16.0,           # 刻度数字
    "legend": 15.5,         # 图例
    "title": 18.5,          # case 标题
    "legend_y": 0.87,       # 图例下沿的坐标区相对高度 (越小越贴近图)
    "title_pad": 26.0,      # 标题离坐标区顶边的距离; 图例框从坐标区里往上长,
                            # 要留出图例那两行的位置, 不然标题会贴上去
    "title_y": None,        # 给了就按坐标区内的相对高度放标题 (拼版用), 否则走 pad
}
GRID_FONTS = {              # 拼版时每个 panel 小得多, 字号整体收一档
    "label": 11.0, "tick": 9.5, "legend": 12.0, "title": 13.0,
    "legend_y": 0.90, "title_pad": 0.0,
    # 3d 画出来的盒子只占坐标区中间七成, 标题按 pad 放在坐标区外面的话,
    # 盒子顶上会留一大片空白。pad 压成负的没用 —— matplotlib 把标题底边夹在
    # 坐标区顶边上, pad<=-20 之后再压就不再下移了; 改用 y 直接给坐标区内的位置。
    # 所有 panel 的盒子大小一致, 所以一个固定值就够。
    "title_y": 0.84,
}

# 论文拼版用的 8 个 case: 规模 6~13 chiplet, 覆盖 7 个系列, HV 余量从险胜到负结果都有。
PAPER8 = ["acend910", "cpu-dram", "hp6_m", "hp11_m",
          "multigpu", "syn4", "xerox6_m", "xerox7_m"]


def legend_handles():
    """图例用的代理图元 (颜色 + 形状); 拼版时全图共用一份图例。"""
    from matplotlib.lines import Line2D
    return [Line2D([], [], linestyle="", marker=MARKER[m], markersize=12,
                   markerfacecolor=SERIES[m], markeredgecolor=SERIES[m],
                   label=m) for m in ORDER]


def parse_axes(spec: str) -> list[int]:
    """--axes 的字符串 -> 三个原始列号 (依次是图的 x, y, z)。"""
    names = [s.strip().lower() for s in spec.split(",")]
    if sorted(names) != sorted(AXIS_OF):
        raise SystemExit(f"--axes 要把 wirelength / area / temp 各写一次, 收到 {spec!r}")
    return [AXIS_OF[n] for n in names]


def available_cases() -> list[str]:
    if not CASE_DIR.exists():
        raise SystemExit(f"找不到 {CASE_DIR}, 先跑 build_case_solution.py")
    return sorted(p.stem for p in CASE_DIR.glob("*.csv") if p.stem != "all")


def read_case(case: str) -> dict[str, list[dict]]:
    """读 case_solution/<case>.csv -> {方法名: [{label, seed, src, v}, ...]}"""
    path = CASE_DIR / f"{case}.csv"
    if not path.exists():
        raise SystemExit(f"找不到 {path}")
    data: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                v = [float(row[col]) for col, _ in OBJECTIVES]
            except (TypeError, ValueError):
                continue
            data.setdefault(row["method"], []).append(
                {"label": row["label"], "seed": row.get("seed", ""),
                 "src": row.get("src", ""), "v": v})
    return data


def pareto_mask(vals) -> list[bool]:
    """vals (n,3) -> 布尔列表, True = 非支配解 (三项都不劣于它、且至少一项更优的解不存在)。"""
    n = len(vals)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and all(vals[j][k] <= vals[i][k] for k in range(3)) \
                    and any(vals[j][k] < vals[i][k] for k in range(3)):
                keep[i] = False
                break
    return keep


def axis_lims(vals) -> list[tuple[float, float]]:
    """按给定点集算每个轴的显示范围 (留 6% 边距)。"""
    lims = []
    for k in range(len(vals[0])):
        col = [v[k] for v in vals]
        lo, hi = min(col), max(col)
        pad = (hi - lo) * 0.06 or max(abs(hi) * 0.01, 1e-6)
        lims.append((lo - pad * 0.5, hi + pad))
    return lims


def inside(vals, lims) -> list[bool]:
    """三个轴都落在显示范围内的点。范围外的点不画 —— 三维里画到框外会糊在图上。"""
    return [all(lims[k][0] <= v[k] <= lims[k][1] for k in range(3)) for v in vals]


def room_corner(lims, elev: float, azim: float) -> list[float]:
    """画出来的三块 pane 相交的那个墙角 —— 也就是房间里三面墙交的内角。

    不是盒子的任意一个棱角: 立体图里 "墙角" 是左墙、右墙、地板三块灰色面交汇的那个
    点, 它在数据坐标里的位置随视角变。matplotlib 把 pane 画在相机的反面, 所以每个轴
    取 min 还是 max, 由视线方向在该轴上的符号定: 视线在 +x 一侧时看到的是 x=xmin
    那块面, 墙角就落在 xmin 上。视线方向 = (cos e cos a, cos e sin a, sin e)。
    """
    e, a = math.radians(elev), math.radians(azim)
    ps = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
    return [lo if p > 0 else hi for p, (lo, hi) in zip(ps, lims)]


def draw_corner_axes(ax, lims, color: str, elev: float, azim: float) -> None:
    """从墙角原点把三条坐标轴画出来 (沿墙根和墙缝走)。

    matplotlib 3d 自带的 axis line 画在盒子的外棱上, 和墙角不相交。所以关掉自带的,
    显式从墙角出发沿三个轴各画到该轴的另一端, 三条线一定交汇在同一个墙角上。
    """
    p = room_corner(lims, elev, azim)
    for k, (axis_lim,) in enumerate(zip(lims)):
        end = list(p)
        # 沿第 k 个轴走到它的另一端 (其它两个坐标保持墙角的值)
        end[k] = axis_lim[1] if p[k] == axis_lim[0] else axis_lim[0]
        xs = [p[0], end[0]] if k == 0 else [p[0], p[0]]
        ys = [p[1], end[1]] if k == 1 else [p[1], p[1]]
        zs = [p[2], end[2]] if k == 2 else [p[2], p[2]]
        # zorder 压在散点下面: 这三条轴线要当参照, 不能横穿到数据点前面去
        ax.plot(xs, ys, zs, color=color, lw=AXIS_LW, linestyle="-",
                solid_capstyle="projecting", zorder=Z_CORNER)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color("none")         # 自带的轴线关掉, 免得多出三条重影


def draw_case(case: str, args, ax, th, fonts=None, legend=True,
              title=None) -> tuple[dict, dict, list[dict]]:
    """把一个 case 画到 ax 上, 返回 (范围外没画的个数, 各方法上前沿数, 前沿表行)。

    fonts 覆盖 FONTS 里的字号/图例位置; legend=False 时不画图例 (拼版时全图共用一份);
    title 给 None 用默认长标题, 拼版时传 case 名当 panel 标题。
    """
    fo = {**FONTS, **(fonts or {})}
    perm = parse_axes(args.axes)
    axis_labels = [OBJECTIVES[i][1].format(wl=args.wl_unit) for i in perm]
    wl_scale = WL_SCALE[args.wl_unit]

    data = read_case(case)
    if not data:
        raise SystemExit(f"{case}: case_solution 里没有解")
    pts, vals = {}, {}
    for name, rs in data.items():
        pts[name] = rs
        v = [list(r["v"]) for r in rs]
        for row in v:
            row[0] *= wl_scale               # 线长轴换单位, 复制的副本, 不动 pts
        vals[name] = [[row[i] for i in perm] for row in v]

    # 四个方法的解合成一个点集算一条全局前沿
    flat, owner = [], []
    for name in ORDER:
        if name not in vals:
            continue
        for v in vals[name]:
            flat.append(v)
            owner.append(name)
    gfront = pareto_mask(flat)
    gfv = [v for v, k in zip(flat, gfront) if k]
    lims = axis_lims(gfv if args.lim == "front" else flat)

    n_hidden, n_front = {}, {}
    for name in ORDER:
        if name not in vals:
            continue
        c = SERIES[name]
        mk = MARKER.get(name, "o")
        v = vals[name]
        shown = inside(v, lims)
        idx = [i for i, o in enumerate(owner) if o == name]
        front = [gfront[i] for i in idx]
        n_hidden[name] = sum(1 for s in shown if not s)
        n_front[name] = sum(1 for f in front if f)

        # 被支配解: 淡色小点, 颜色仍跟着方法走
        ctx = [i for i in range(len(v)) if shown[i] and not front[i]]
        if ctx:
            ax.scatter([v[i][0] for i in ctx], [v[i][1] for i in ctx],
                       [v[i][2] for i in ctx], s=CTX_SIZE, c=c, alpha=CTX_ALPHA,
                       marker=mk, depthshade=False, linewidths=0, zorder=Z_CTX)
        # 垂线: 三维里最容易看丢的就是点的前后深度, 一根竖线把点钉在地面
        if args.stems != "none":
            z_bot = lims[2][0]
            for i in range(len(v)):
                if shown[i] and v[i][2] > z_bot:
                    ax.plot([v[i][0], v[i][0]], [v[i][1], v[i][1]], [z_bot, v[i][2]],
                            color=c, lw=0.9, alpha=STEM_ALPHA, zorder=Z_STEM)
        # 落在全局前沿上的解: 大点 + 表面色描边环
        fv = [v[i] for i in range(len(v)) if front[i] and shown[i]]
        # 图例文案必须短: 两列并排以后整句 "N of M on the front" 会把图例框撑得
        # 比图还宽, 所以只留分数 (分子=进全局前沿的个数, 分母=该方法解总数),
        # "on the front" 这层含义交给图例标题交代。
        label = f"{name}  ({n_front[name]}/{len(v)})"
        if fv:
            ax.scatter([p[0] for p in fv], [p[1] for p in fv], [p[2] for p in fv],
                       s=FRONT_SIZE, c=c, alpha=FRONT_ALPHA, marker=mk,
                       depthshade=False, edgecolors=th["surface"], linewidths=1.6,
                       label=label, zorder=Z_FRONT)
        else:
            # 一个解都没进前沿时把图例挂到小点上, 免得这个方法在图例里消失
            ax.scatter([], [], [], s=CTX_SIZE, c=c, alpha=CTX_ALPHA, marker=mk,
                       label=label)

    ax.set_xlabel(axis_labels[0], color=th["ink2"], fontsize=fo["label"], labelpad=10)
    ax.set_ylabel(axis_labels[1], color=th["ink2"], fontsize=fo["label"], labelpad=10)
    # 温度轴的标题要跟着刻度一起往外让 (刻度 pad 提到 6 以后数字会顶到标题位置)
    ax.set_zlabel(axis_labels[2], color=th["ink2"], fontsize=fo["label"], labelpad=15)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    try:
        box = tuple(float(x) for x in str(args.box_aspect).split(","))
        if len(box) != 3:
            raise ValueError
    except ValueError:
        raise SystemExit(f"--box-aspect 要三个逗号分隔的数, 收到 {args.box_aspect!r}")
    ax.set_box_aspect(box)
    ax.view_init(elev=args.elev, azim=args.azim)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(th["pane"], 1.0))
        axis._axinfo["grid"].update(color=th["grid"], linewidth=0.6)
        # 网格线和 pane 都挂在轴 artist 上, 把轴整体压到最底下, 两者就都跑到点后面了
        axis.set_zorder(Z_AXES)
    if args.axis_lines:
        draw_corner_axes(ax, lims, th["axis"], args.elev, args.azim)
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color(th["axis"])
            axis.line.set_linewidth(AXIS_LW)
    ax.tick_params(colors=th["ink2"], labelsize=fo["tick"])
    # 3d 里刻度文字的偏移是 (pad + 8) * points, 同样的 pad 投影后观感差很多:
    # 竖轴的数贴着轴, 横轴的数飘着, 所以温度往外让、线长往里收, 面积轴不动。
    ax.zaxis.set_tick_params(pad=6)
    ax.xaxis.set_tick_params(pad=1)
    # 字号大了以后自动刻度会自己撞在一起, 三个轴都压到 6 个以内 (3d 里刻度是斜的)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    # 图例放到坐标区外面 (上方两行): 墙角坐标轴的竖线是穿过画面中央的, 图例放在
    # 左上角会被它划开, 而且那条线在画面里的位置随视角变, 放哪个角都不保险。
    head = title if title else f"{case} — 3D Pareto front (all three objectives minimised)"
    if not legend:
        if fo.get("title_y") is None:
            ax.set_title(head, color=th["ink"], fontsize=fo["title"],
                         pad=fo["title_pad"])
        else:
            ax.set_title(head, color=th["ink"], fontsize=fo["title"],
                         pad=0, y=fo["title_y"])
    else:
        ax.set_title(head, color=th["ink"], fontsize=fo["title"], pad=fo["title_pad"])
        leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, fo["legend_y"]),
                        ncol=2, fontsize=fo["legend"], frameon=True,
                        facecolor=th["pane"], edgecolor=th["grid"],
                        handletextpad=0.5, columnspacing=1.4, borderpad=0.7)
        for t in leg.get_texts():
            t.set_color(th["ink"])

    # 表格视图: 只放落在全局前沿上的解 (用原始 mm 值, 不受画图单位影响)
    rows = []
    for name in ORDER:
        if name not in vals:
            continue
        idx = [i for i, o in enumerate(owner) if o == name]
        for j, r in enumerate(pts[name]):
            if gfront[idx[j]]:
                rows.append({"method": name, "label": r["label"], "seed": r["seed"],
                             "src": r["src"],
                             **{k: f"{x:.6f}" for (k, _), x in zip(OBJECTIVES, r["v"])}})
    return n_hidden, n_front, rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="*", help="case 名; 不给就画 case_solution 里全部")
    ap.add_argument("--elev", type=float, default=15.0, help="俯仰角 (默认 15)")
    ap.add_argument("--azim", type=float, default=-70.0, help="方位角 (默认 -70)")
    ap.add_argument("--box-aspect", default="1,1,1",
                    help="三个轴的长度比 (默认 1,1,1)。matplotlib 3d 默认竖轴被压掉 "
                         "25%%, 想更强调凹面就把第三个数调大, 如 1,1,1.3")
    ap.add_argument("--axes", default="wirelength,area,temp",
                    help="三个目标依次放 x,y,z 轴 (默认 wirelength,area,temp)")
    ap.add_argument("--lim", choices=["all", "front"], default="all",
                    help="坐标轴范围按什么取: all=全部解 (默认, 每个解都画得出来), "
                         "front=只按全局前沿取 (前沿撑满画面, 但范围外的被支配解就不画了)")
    ap.add_argument("--stems", choices=["none", "floor"], default="floor",
                    help="给每个点画一根竖线当位置参照 (默认垂到画面底面)")
    ap.add_argument("--axis-lines", dest="axis_lines", action="store_true", default=True,
                    help="从墙角原点用浅灰细线画三条坐标轴 (默认开)")
    ap.add_argument("--no-axis-lines", dest="axis_lines", action="store_false",
                    help="关掉墙角坐标轴, 用 matplotlib 自带的轴线")
    ap.add_argument("--wl-unit", choices=["mm", "m"], default="m",
                    help="总线长轴的单位 (默认 m)。只影响画图, 写的 csv 恒为 mm")
    ap.add_argument("--grid", default="",
                    help="拼成一张多 panel 大图, 例 --grid 2x4; 全图共用一份图例")
    ap.add_argument("--grid-out", default="paper8_pareto3d.png",
                    help="拼版图的文件名 (写到 fig/ 下)")
    ap.add_argument("--dpi", type=int, default=200, help="位图 dpi (默认 200)")
    ap.add_argument("--paper8", action="store_true",
                    help=f"等价于在命令行里写出论文用的 8 个 case: {' '.join(PAPER8)}")
    ap.add_argument("--dark", action="store_true", help="深色底")
    args = ap.parse_args()

    if args.paper8:
        args.case = list(PAPER8) + list(args.case)
    th = THEME["dark" if args.dark else "light"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    cases = args.case or available_cases()

    def report(case, n_hidden, n_front, rows, out_png):
        out_csv = TAB_DIR / f"{case}_pareto_front.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "label", "seed", "src",
                                              *[k for k, _ in OBJECTIVES]])
            w.writeheader()
            w.writerows(rows)
        counts = "  ".join(f"{m}={n_front[m]}/{len(read_case(case).get(m, []))}"
                           for m in ORDER if m in n_front)
        print(f"[{case}] 全局前沿 {sum(n_front.values())} 个解   {counts}")
        if any(n_hidden.values()):
            print("   (范围外没画: " + ", ".join(f"{k}={v}" for k, v in n_hidden.items() if v)
                  + "; 想看全貌加 --lim all)")
        if out_png is not None:
            print(f"   -> {out_png}")
        print(f"   -> {out_csv}  (前沿 {len(rows)} 行)")

    def new_axes(fig, loc):
        ax = fig.add_subplot(*loc, projection="3d", facecolor=th["surface"])
        # 默认的自动深度排序会让散点穿插着画, 关掉, 改用显式 zorder
        ax.computed_zorder = False
        return ax

    if args.grid:
        nr, nc = (int(x) for x in args.grid.lower().split("x"))
        if len(cases) > nr * nc:
            raise SystemExit(f"--grid {args.grid} 只有 {nr * nc} 格, 放不下 {len(cases)} 个 case")
        # 每格给足宽度: 3d 的轴标题和刻度都画在坐标区外面, 而且 tight bbox 抓不到
        # 竖轴的标题, 所以这里不用 bbox_inches="tight", 改为自己排好边距留白,
        # 靠 wspace/hspace 把每个 panel 溢出的标签圈在自己的格子里。
        fig = plt.figure(figsize=(6.2 * nc, 5.2 * nr + 0.7), facecolor=th["surface"])
        for k, case in enumerate(cases):
            ax = new_axes(fig, (nr, nc, k + 1))
            n_hidden, n_front, rows = draw_case(case, args, ax, th, fonts=GRID_FONTS,
                                                legend=False, title=case)
            report(case, n_hidden, n_front, rows, None)
        # bottom 要给足: 每格右下角 "Bounding Box Area" 是斜着画到坐标区外面的,
        # 最后一行没有下一行给它让位, 留窄了就会像上一版那样被裁掉半行字。
        # hspace 要给上一行的 x 轴标题留位: 那行字画在坐标区外面, 行挨太近就会
        # 叠到下一行的 case 标题上。
        fig.subplots_adjust(left=0.012, right=0.988, top=0.88, bottom=0.07,
                            wspace=0.34, hspace=0.20)
        # 全图共用一份图例: 8 个 panel 各挂一份 4 条图例等于 32 行重复的墨,
        # 而且每个 panel 都被图例挤掉一块画图区。各 case 的前沿计数在正文表里。
        fig.legend(handles=legend_handles(), loc="upper center", ncol=len(ORDER),
                   fontsize=GRID_FONTS["legend"] + 2, frameon=False,
                   bbox_to_anchor=(0.5, 0.995), handletextpad=0.5, columnspacing=2.4)
        out_png = FIG_DIR / args.grid_out
        fig.savefig(out_png, dpi=args.dpi, facecolor=th["surface"])
        plt.close(fig)
        print(f"\n拼版图 {nr}x{nc} ({len(cases)} 个 case) -> {out_png}")
        return

    for case in cases:
        fig = plt.figure(figsize=(11.0, 8.4), facecolor=th["surface"])
        ax = new_axes(fig, (1, 1, 1))
        n_hidden, n_front, rows = draw_case(case, args, ax, th)

        out_png = FIG_DIR / f"{case}_pareto3d.png"
        # pad_inches 不能省: tight bbox 抓不到 3d 竖轴的轴标签, 留白会被裁掉
        fig.savefig(out_png, dpi=args.dpi, facecolor=th["surface"],
                    bbox_inches="tight", pad_inches=0.4)
        plt.close(fig)
        report(case, n_hidden, n_front, rows, out_png)


if __name__ == "__main__":
    main()
