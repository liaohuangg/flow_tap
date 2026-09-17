#!/usr/bin/env python3
"""plot_pareto3d.py — 画一个 case 的三维帕累托前沿: 总线长 / 外接框面积 / 峰值温度。

三个目标全部越小越好, 所以 "支配" 定义为: p 三项都不比 q 差、且至少一项更好。
每个方法各自算自己的帕累托前沿 (一个方法的解集能达到的最优权衡), 两个方法叠在
同一张图里对比。

三个目标的帕累托前沿一般是一张二维曲面而不是一条曲线, 所以除了画点, 还会铺一张曲面
(--surface): 三个目标里挑两个当底面、第三个当高度, 曲面 = 经验前沿面
min{ 高度轴 : 另外两轴都不超过 } 的柔化版 (见 smooth_envelope)。它的关键性质是
"所有点都落在曲面上或曲面之上, 而非支配解恰好贴着曲面" —— 于是每个解到曲面的竖直
距离就是它为另外两个目标付出的代价, 可以直接读。曲面由两家的点一起撑起来, 是共用的
参照, 不属于任何一个方法, 所以用中性灰而不是分类色。

三轴默认是 x=面积, y=峰温, z=总线长 (--axes 可换成另外两种循环轮换), 于是曲面 = 给定
面积和温度预算下能拿到的最短总线长。注意前沿的弯曲形状随 case 变, 不是一律朝原点凹:
实测 hp11_m 像碗, acend910 和 xerox8_m 更接近鞍面。用 --elev/--azim/--box-aspect 换视角。

数据源 (都是 eval_layout.py 的产物, 三列口径完全一致, 可以直接比):
  AT    AT_result/50set.csv
        case 列形如 acend910_wl00            —— wlsweep 的 50 个 wl_weight 点
  本方法 FM_result/pareto_weight_sweep_seed121_legal_only_results/result.csv
        case 列形如 acend910_thermal0p03_wirelength50
        —— thermal_weight × wirelength_weight 网格上的解

输出 (都写进本目录):
  <case>_pareto3d.png        三维图
  <case>_pareto_front.csv    图里高亮的非支配解, 用作表格视图 (便于核对具体数值)

用法:
  python plot_pareto3d.py acend910
  python plot_pareto3d.py hp11_m --elev 18 --azim -55
  python plot_pareto3d.py hp11_m --stems surface   # 竖线垂到曲面上, 看离前沿还差多少度
  python plot_pareto3d.py xerox8_m --lim all   # 全部解都进画面 (前沿会被离群点挤小)
  python plot_pareto3d.py acend910 --dark      # 深色底版本
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (注册 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

PROJECT = Path("/root/placement/flow_tap")
RESULT_EVAL = PROJECT / "resultEval"
OUT_DIR = RESULT_EVAL / "pareto_front"

# 三个目标: (csv 列名, 图上的轴标签)。全部是 "越小越好"。
OBJECTIVES = [
    ("total_wirelength_mm", "Total wirelength ({wl})"),
    ("bbox_area_mm2", "Footprint area (mm$^2$)"),
    ("max_temp_C", "Peak temperature ($^\\circ$C)"),
]
# 总线长在图上用 m 更顺眼 (几十万 mm 的刻度没人读)。只影响画图, csv 表恒为 mm。
WL_SCALE = {"mm": 1.0, "m": 0.001}

# 三个目标谁上哪个轴。三维里这是一次循环轮换 (x<-y<-z), 三种排法等价, 只是看的角度不同;
# 默认把线长放在竖轴 —— 线长是这里差别最大、最想比的一项, 放竖轴最容易读出高低。
# 曲面始终建在前两个轴上 (见 envelope_height), 所以换 --axes 曲面会跟着换底, 不用改别的。
AXIS_OF = {"wirelength": 0, "area": 1, "temp": 2}

# 两个方法: (图例名, csv 路径)。csv 里的 case 列 = <case> + 参数尾缀。
METHODS = [
    ("AT", RESULT_EVAL / "AT_result" / "50set.csv"),
    ("Ours", RESULT_EVAL / "FM_result" / "pareto_weight_sweep_seed121_legal_only_results" / "result.csv"),
]

# 分类色只按实体固定分配, 不随筛选后的顺序变: AT 恒为 slot1 (蓝), Ours 恒为 slot2 (橙)。
# 取自参考调色板, 已过 validate_palette.js (light/dark 两种 mode 的 all-pairs 检查全 PASS)。
THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#dedcd4", "pane": "#f4f3ef", "surf": "#6b6a66"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#3a3a37", "pane": "#232322", "surf": "#8a8985"},
}
SERIES = {
    "light": {"AT": "#2a78d6", "Ours": "#eb6834"},
    "dark": {"AT": "#3987e5", "Ours": "#d95926"},
}

# 图里的文字一律英文 —— 本环境的 matplotlib 没有中文字体 (出图与中文控制台是两件事)。
CTX_SIZE, CTX_ALPHA = 26.0, 0.30          # 被支配解: 小点淡显, 只做背景
FRONT_SIZE, FRONT_ALPHA = 78.0, 0.95      # 非支配解: 大点 + 2px 表面色描边
SURF_ALPHA = 0.42                         # 光滑前沿曲面: 半透明, 后面的点还看得见
FLAT_ALPHA, FLAT_EDGE_ALPHA = 0.16, 0.55  # 折面版 (tri/hull): 填充 + 网格线
STEM_ALPHA = 0.30                         # 垂线: 只做位置参照, 不能盖过点本身


def parse_axes(spec: str) -> list[int]:
    """--axes 的字符串 -> 三个原始列号 (依次是图的 x, y, z)。"""
    names = [s.strip().lower() for s in spec.split(",")]
    if sorted(names) != sorted(AXIS_OF):
        raise SystemExit(f"--axes 要把 wirelength / area / temp 各写一次, 收到 {spec!r}")
    return [AXIS_OF[n] for n in names]


def case_of(label: str) -> str:
    """两边 csv 的 case 列都带参数尾缀, 剥回 case 名。

    AT: <case>_wl<NN>                              -> <case>
    FM: <case>_thermal<X>_wirelength<Y>            -> <case>
    """
    return label.split("_thermal", 1)[0].split("_wl", 1)[0]


def read_points(path: Path, case: str) -> list[dict]:
    """读一个 csv 里属于 case 的有效行。error 非空的行直接丢掉。"""
    if not path.exists():
        raise SystemExit(f"找不到数据文件: {path}")
    pts = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("error") or "").strip():
                continue
            if case_of(row["case"]) != case:
                continue
            try:
                vals = [float(row[col]) for col, _ in OBJECTIVES]
            except (TypeError, ValueError):
                continue
            pts.append({"label": row["case"], "v": np.array(vals)})
    return pts


def pareto_mask(vals: np.ndarray) -> np.ndarray:
    """vals (n, 3) -> 布尔掩码, True = 非支配解 (三项都不劣于它、且至少一项更优的解不存在)。"""
    n = len(vals)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dominates = np.all(vals <= vals[i], axis=1) & np.any(vals < vals[i], axis=1)
        if dominates.any():
            keep[i] = False
    return keep


def envelope_height(vals: np.ndarray):
    """经验前沿面: z(u, v) = min{ 峰温_i : 线长_i <= u 且 面积_i <= v }。

    读作 "给这么多线长和面积预算, 已知能拿到的最好峰温"。这是采样点集能给出的前沿面的
    最自然的定义, 关键是它有两个插值曲面给不了的性质:

      * 每个采样点都落在曲面上或曲面之上 —— 点 j 在 (线长_j, 面积_j) 处自己就参与取 min;
      * "贴着曲面" 恰好等价于 "非支配" —— 被支配的解一定浮在曲面上面, 浮起的高度就是
        它为了线长/面积牺牲掉的峰温, 是个有量纲、可比较的数。

    之前用的 "过非支配解做 RBF 插值" 没这两个性质: 曲面从点云中间穿过去, 点一半在上
    一半在下 (实测 acend910 点面距离中位数只有 0.04C), 看着就是一张搭在点云上的布,
    不像前沿。曲面单调不增: 预算给得越多, 能拿到的最好峰温只会更低。
    """
    wl, ar, tp = vals[:, 0], vals[:, 1], vals[:, 2]

    def height_at(pu, pv) -> np.ndarray:
        pu = np.atleast_1d(np.asarray(pu, dtype=float))
        pv = np.atleast_1d(np.asarray(pv, dtype=float))
        ok = (wl[None, :] <= pu[:, None]) & (ar[None, :] <= pv[:, None])
        z = np.where(ok, tp[None, :], np.inf).min(axis=1)
        return np.where(ok.any(axis=1), z, np.nan)   # 该处一个点都不满足 -> 没定义

    return height_at


def smooth_envelope(vals: np.ndarray, n: int, lims: list[tuple[float, float]], blur: float):
    """包络的柔化版, 返回 (X, Y, Z, 求值函数)。求值函数是用来给垂线找落脚点的。

    做法: 先对包络做高斯模糊, 再逐点取 "模糊结果 与 原包络 的较小值"。取 min 这一步不能
    省 —— 模糊会把台阶顶上的值抹高, 抹高就意味着曲面跑到采样点上面去了, 前沿面不能这样。
    取 min 之后曲面处处 <= 包络, 而包络 <= 任何采样点, 所以 "所有点都在曲面上或之上"
    这条性质仍然成立, 同时竖墙变成了斜坡。

    代价是斜坡的上下端各留一处折角 (连续但不光滑), 比垂直墙面好得多。
    """
    U, V = np.meshgrid(np.linspace(*lims[0], n), np.linspace(*lims[1], n))
    Z0 = envelope_height(vals)(U.ravel(), V.ravel()).reshape(U.shape)
    unknown = np.isnan(Z0)
    if unknown.all():
        return U, V, Z0, None
    # 先填掉 "这里一个点都不满足" 的角再模糊, 否则 NaN 会被卷积抹开一大片, 最后再挖回 NaN。
    # 填最近的真实格子的值 (而不是最大或最小值): 填大的等于往曲面里灌假高值 —— 实测能把
    # 左边界的点顶到曲面上面 135m; 填小的 (如全局最小) 会把边界整片拽下去, 前沿点又浮起来
    # 85m。取最近邻两头都不偏, 剩下的误差交给下面那步局部下压去兜。
    fill = distance_transform_edt(unknown, return_distances=False, return_indices=True)
    Zf = Z0[tuple(fill)]
    Zs = gaussian_filter(Zf, blur * n) if blur > 0 else Zf
    Z = np.minimum(Zs, Zf)

    # 到这里曲面在网格点上 <= 包络, 但格子内部是线性插的, 而包络在格子里还会继续往下掉
    # (它是单调下降的), 于是插出来的面可能高过真实包络、甚至高过某个采样点 (实测不管的话
    # 有点能穿出曲面 60m)。所以在采样点上再验一遍, 把还冒出来的位置按冒出的深度局部压下去。
    #
    # 压的方式用 "各影响半径内取最大值", 不用解 RBF 方程组: 在中心处严格等于该点的深度,
    # 别处只会把曲面压得更低 (永不会抬高), 也不存在矩阵病态。压一次之后格子内部仍差一点点
    # (实测 0.2m 量级), 所以迭代几轮 —— 每轮都只往下压, 收敛的。
    sigma = max(blur, 0.02)                      # 影响半径, 单位同 su/sv (0~1)
    su = lambda a: (np.asarray(a, float) - U[0, 0]) / (U[0, -1] - U[0, 0])  # noqa: E731
    sv = lambda a: (np.asarray(a, float) - V[0, 0]) / (V[-1, 0] - V[0, 0])  # noqa: E731
    for _ in range(4):
        grid_at = RegularGridInterpolator((V[:, 0], U[0]), Z, bounds_error=False,
                                          fill_value=np.nan)
        viol = np.maximum(0.0, np.nan_to_num(grid_at(np.c_[vals[:, 1], vals[:, 0]]), nan=0.0)
                          - vals[:, 2])
        bad = viol > 1e-6
        if not bad.any():
            break
        gu, gv = su(U.ravel())[:, None], sv(V.ravel())[:, None]
        d2 = ((gu - su(vals[bad, 0])[None, :]) ** 2
              + (gv - sv(vals[bad, 1])[None, :]) ** 2) / sigma ** 2
        Z = Z - np.max(viol[bad][None, :] * np.exp(-d2 / 2.0), axis=1).reshape(Z.shape)

    Z = np.where(unknown, np.nan, Z)
    grid_at = RegularGridInterpolator((V[:, 0], U[0]), Z, bounds_error=False, fill_value=np.nan)

    def height_at(pu, pv) -> np.ndarray:
        return grid_at(np.c_[np.atleast_1d(np.asarray(pv, float)),
                             np.atleast_1d(np.asarray(pu, float))])

    return U, V, Z, height_at


def surface_hull(fv: np.ndarray) -> np.ndarray | None:
    """前沿点的凸包上朝向原点的那些面 —— 即 "凸帕累托前沿"。

    外法向三个分量都为负, 说明这个面朝的就是三个目标都更小的那一侧, 属于前沿面;
    其余的面朝着某一维更大的方向, 是包住的被支配区域, 不画。
    """
    if len(fv) < 4:
        return None
    try:
        hull = ConvexHull(fv)
    except Exception:  # noqa: BLE001  (共面/退化)
        return None
    tris = [fv[s] for eq, s in zip(hull.equations, hull.simplices)
            if np.all(eq[:3] < -1e-9)]
    return np.array(tris) if tris else None


def axis_lims(vals: np.ndarray) -> list[tuple[float, float]]:
    """按给定点集算每个轴的显示范围 (留 6% 边距)。"""
    lims = []
    for k in range(vals.shape[1]):
        col = vals[:, k]
        lo, hi = float(col.min()), float(col.max())
        pad = (hi - lo) * 0.06 or max(abs(hi) * 0.01, 1e-6)
        lims.append((lo - pad * 0.5, hi + pad))
    return lims


def inside(vals: np.ndarray, lims: list[tuple[float, float]]) -> np.ndarray:
    """三个轴都落在显示范围内的点。范围外的点不画 —— 三维里画到框外会糊在图上。"""
    m = np.ones(len(vals), dtype=bool)
    for k, (lo, hi) in enumerate(lims):
        m &= (vals[:, k] >= lo) & (vals[:, k] <= hi)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", help="case 名, 如 acend910 / hp11_m / xerox8_m")
    ap.add_argument("--out", default=None, help="输出 png 路径 (默认 <本目录>/<case>_pareto3d.png)")
    ap.add_argument("--elev", type=float, default=15.0, help="俯仰角 (默认 15, 压低一点更看得出曲面的弯)")
    ap.add_argument("--azim", type=float, default=-70.0, help="方位角 (默认 -70)")
    ap.add_argument("--box-aspect", default="1,1,1",
                    help="三个轴的长度比 (默认 1,1,1)。matplotlib 3d 默认是 1.19,1.19,0.89, "
                         "竖轴被压掉 25%%, 前沿朝原点凹进去的那点弧度会被压平; 想更强调凹面就"
                         "把第三个数调大, 如 1,1,1.3")
    ap.add_argument("--axes", default="area,temp,wirelength",
                    help="三个目标依次放 x,y,z 轴 (默认 area,temp,wirelength)。三者只能各出现"
                         "一次, 另外两种排法是 wirelength,area,temp (原版) 和 temp,wirelength,area。"
                         "曲面建在前两个轴上、第三个当高度, 所以换排法曲面会跟着换底")
    ap.add_argument("--lim", choices=["front", "all"], default="front",
                    help="坐标轴范围按什么取: front=两个方法的前沿并集 (默认, 前沿撑满画面), "
                         "all=全部解 (离群点会把轴拉长、前沿挤成一坨, 一般只在要展示全貌时用)")
    ap.add_argument("--surface", choices=["smooth", "envelope", "rbf", "hull", "none"],
                    default="smooth",
                    help="前沿曲面怎么画: smooth=柔化的经验前沿面 (默认, 见 smooth_envelope); "
                         "envelope=不柔化的经验前沿面 —— 就是 min{目标3: 目标1和目标2都不超}, "
                         "所有点都在它上面, 但采样点撑出来的阶梯会画成一堵堵竖墙; "
                         "hull=非支配点的凸包下表面 (凸帕累托前沿, 凹处被补平, 偏乐观); none=只画点")
    ap.add_argument("--surface-grid", type=int, default=200,
                    help="曲面的网格密度 (默认 200)。曲面是按网格采样的, 掩膜边界会被量化成\n                         锯齿; 网格越细锯齿越小, 代价是变慢")
    ap.add_argument("--surface-smooth", type=float, default=0.03,
                    help="仅 --surface smooth 用: 柔化半径, 单位是底面边长 (默认 0.03, "
                         "约等于把 3%% 见方的范围内的阶梯抹平)。调大更圆滑, 但离采样点更远; "
                         "0 = 不柔化, 等于 --surface envelope")
    ap.add_argument("--stems", choices=["none", "floor", "surface"], default="floor",
                    help="给每个点画一根竖线当位置参照 (三维图里前后深度最容易看丢): "
                         "floor=垂到画面底面 (默认); "
                         "surface=垂到前沿曲面 —— 线段的长度就是该解离前沿还差多少温度, "
                         "两个方法的线长分布能直接对比, 但只有 --surface smooth 时可用")
    ap.add_argument("--line", action="store_true",
                    help="把前沿点按总线长排序连成折线。默认不画 —— 三个目标的帕累托前沿"
                         "一般是一张曲面而不是一条曲线, 一维排序连出来的线会自己穿过自己, "
                         "容易看成比实际更规整的前沿。只在确定前沿退化成一维时才开")
    ap.add_argument("--wl-unit", choices=["mm", "m"], default="m",
                    help="总线长轴的单位 (默认 m)。只影响画图, 写出的 csv 恒为 mm")
    ap.add_argument("--dark", action="store_true", help="深色底")
    args = ap.parse_args()

    th = THEME["dark" if args.dark else "light"]
    series_color = SERIES["dark" if args.dark else "light"]

    wl_scale = WL_SCALE[args.wl_unit]
    perm = parse_axes(args.axes)
    axis_labels = [OBJECTIVES[i][1].format(wl=args.wl_unit) for i in perm]

    data = {}
    for name, path in METHODS:
        pts = read_points(path, args.case)
        if not pts:
            continue
        raw = np.vstack([p["v"] for p in pts])
        raw[:, 0] *= wl_scale         # vstack 已经是副本, 改它不会动 pts, csv 表仍是 mm
        # vals 是投到图上的坐标 (显示单位 + 按 --axes 换过序); raw 保持 csv 的列序, 控制台用
        data[name] = {"pts": pts, "raw": raw, "vals": raw[:, perm]}

    if not data:
        avail = set()
        for _, path in METHODS:
            with path.open(encoding="utf-8") as f:
                avail |= {case_of(r["case"]) for r in csv.DictReader(f)}
        raise SystemExit(f"case {args.case!r} 在两个 csv 里都没有数据。可用的 case: "
                         f"{', '.join(sorted(avail))}")

    # ---- 画图 ---------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 8.4), facecolor=th["surface"])
    ax = fig.add_subplot(111, projection="3d", facecolor=th["surface"])
    # 默认的自动深度排序会把散点和曲面穿插着画 (点被曲面盖住). 关掉, 改用显式 zorder:
    # 曲面 1 < 被支配点 2 < 前沿点 4。
    ax.computed_zorder = False

    # 两个方法的解合成一个点集算一条全局前沿 —— 曲面只画这一条。这是这张图的重点:
    # 每个方法各自的点落在同一张曲面的上方还是贴着它, 一眼就能看出谁的解更贴前沿
    # (曲面由最少的那批解撑起来, 所有点都在曲面上或曲面之上, 到曲面的垂距就是被支配的程度)。
    all_vals = np.vstack([d["vals"] for d in data.values()])
    gfront = pareto_mask(all_vals)
    gfv = all_vals[gfront]

    # 各方法在 all_vals 里的行区间, 用来把全局前沿的掩码切回每个方法
    span, off = {}, 0
    for name, d in data.items():
        span[name] = (off, off + len(d["pts"]))
        off += len(d["pts"])

    lims = axis_lims(gfv if args.lim == "front" else all_vals)

    # 曲面: 由两家的点一起撑起来, 是共用的参照, 不属于任何一个方法, 所以用中性灰而不是分类色
    height_at = None
    if args.surface in ("smooth", "envelope") and len(all_vals) >= 4:
        # 用全部点 (不只非支配解) 取包络 —— 被支配解永远不会成为某处的最小值, 结果一样,
        # 但直接对全量取 min 少一层 "先筛前沿" 的隐含假设
        X, Y, Z, height_at = smooth_envelope(
            all_vals, args.surface_grid, lims,
            args.surface_smooth if args.surface == "smooth" else 0.0)
        ax.plot_surface(X, Y, Z, color=th["surf"], alpha=SURF_ALPHA,
                        linewidth=0, antialiased=True, shade=True,
                        rcount=args.surface_grid, ccount=args.surface_grid, zorder=1)
    elif args.surface == "hull" and len(gfv) >= 4:
        tris = surface_hull(gfv)
        if tris is not None and len(tris):
            ax.add_collection3d(Poly3DCollection(
                tris, facecolor=to_rgba(th["surf"], FLAT_ALPHA),
                edgecolor=to_rgba(th["surf"], FLAT_EDGE_ALPHA), linewidths=0.7, zorder=1))

    n_hidden = {name: 0 for name in data}
    for name, d in data.items():
        c = series_color[name]
        v = d["vals"]
        # 范围外的点不画 (三维里画到框外会糊在图上), 但要数出来告诉用户
        shown = inside(v, lims)
        n_hidden[name] = int((~shown).sum())
        lo, hi = span[name]
        front = gfront[lo:hi] & shown           # 该方法里落在全局前沿上的解
        fv = v[front]

        # 被支配解: 淡色小点, 颜色仍跟着方法走 (不做按名次的重新配色)
        ctx = shown & ~gfront[lo:hi]
        ax.scatter(v[ctx, 0], v[ctx, 1], v[ctx, 2],
                   s=CTX_SIZE, c=c, alpha=CTX_ALPHA, depthshade=False,
                   linewidths=0, zorder=2)

        # 垂线: 三维里最容易看丢的就是点的前后深度, 一根竖线把点钉在地面/曲面上,
        # 空间关系立刻就清楚了。surface 模式画的是点到前沿的竖直差距 (温差), 底面
        # 落在凸包外的点没有对应曲面, 自然就没有线。
        if args.stems != "none":
            z_bot = (height_at(v[:, 0], v[:, 1]) if args.stems == "surface" and height_at is not None
                     else np.full(len(v), lims[2][0]))
            keep = shown & np.isfinite(z_bot) & (v[:, 2] > z_bot)
            n = int(keep.sum())
            if n:
                # 每条线两个端点, 第三行塞 NaN 把各条线断开 (mplot3d 只能一次画一条折线)
                seg = np.full((n, 3, 3), np.nan)
                seg[:, 0, :2] = seg[:, 1, :2] = v[keep, :2]
                seg[:, 0, 2] = z_bot[keep]
                seg[:, 1, 2] = v[keep, 2]
                ax.plot(seg[:, :, 0].ravel(), seg[:, :, 1].ravel(), seg[:, :, 2].ravel(),
                        color=c, lw=0.9, alpha=STEM_ALPHA, zorder=1.5)

        # 落在全局前沿上的解 (贴着曲面的那批): 大点 + 表面色描边环
        label = f"{name}  ({int(gfront[lo:hi].sum())} of {len(v)} on the front)"
        if len(fv):
            ax.scatter(fv[:, 0], fv[:, 1], fv[:, 2],
                       s=FRONT_SIZE, c=c, alpha=FRONT_ALPHA, depthshade=False,
                       edgecolors=th["surface"], linewidths=1.6,
                       label=label, zorder=4)
        else:
            # 该方法一个解都没进前沿时, 图例挂到小点上, 免得这个方法在图例里消失
            ax.scatter([], [], [], s=CTX_SIZE, c=c, alpha=CTX_ALPHA, label=label)

        # 前沿折线 (默认关): 按第 1 个目标 (总线长) 排序后连起来
        if args.line and len(fv) > 1:
            s = fv[np.argsort(fv[:, 0])]
            ax.plot(s[:, 0], s[:, 1], s[:, 2], color=c, lw=2.0, alpha=0.75, zorder=5)

    ax.set_xlabel(axis_labels[0], color=th["ink2"], fontsize=11, labelpad=10)
    ax.set_ylabel(axis_labels[1], color=th["ink2"], fontsize=11, labelpad=10)
    ax.set_zlabel(axis_labels[2], color=th["ink2"], fontsize=11, labelpad=10)
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
        axis.line.set_color(th["grid"])
    ax.tick_params(colors=th["ink2"], labelsize=9)
    ax.set_title(f"{args.case} — 3D Pareto front (all three objectives minimised)",
                 color=th["ink"], fontsize=13, pad=18)
    leg = ax.legend(loc="upper left", fontsize=10, frameon=True,
                    facecolor=th["pane"], edgecolor=th["grid"])
    for t in leg.get_texts():
        t.set_color(th["ink"])

    out_png = Path(args.out) if args.out else OUT_DIR / f"{args.case}_pareto3d.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # pad_inches 不能省: tight bbox 抓不到 3d 竖轴的轴标签, 留白会被裁掉
    fig.savefig(out_png, dpi=200, facecolor=th["surface"], bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)

    # ---- 表格视图 + 控制台小结 ---------------------------------------------
    # 表格视图跟图放同一个目录 —— --out 指到别处时不要把 csv 留在默认目录里
    out_csv = out_png.parent / f"{args.case}_pareto_front.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "label", "total_wirelength_mm", "bbox_area_mm2", "max_temp_C"])
        for name, d in data.items():
            lo, hi = span[name]
            for p, is_f in zip(d["pts"], gfront[lo:hi]):
                if is_f:
                    w.writerow([name, p["label"], *[f"{x:.6f}" for x in p["v"]]])

    print(f"[{args.case}] 数据源: " + ", ".join(f"{n}={len(d['pts'])}解" for n, d in data.items())
          + f"   坐标范围按 lim={args.lim}")
    print(f"  轴: x={args.axes.split(',')[0]}, y={args.axes.split(',')[1]}, "
          f"z={args.axes.split(',')[2]} (曲面底 = x×y, 高度 = z)")
    if any(n_hidden.values()):
        print(f"  (范围外的被支配解没画: "
              + ", ".join(f"{n}={c}" for n, c in n_hidden.items() if c)
              + "; 想看全貌加 --lim all)")
    print(f"  全局前沿 (两方法合起来算) {int(gfront.sum())} 个解:")
    for name, d in data.items():
        lo, hi = span[name]
        # 固定按 csv 的列序报 (线长/面积/温度), 跟图上的轴顺序无关, 免得看晕;
        # 线长在 raw 里已经是显示单位 (米制时被 wl_scale 缩过), 别硬写 mm
        r = d["raw"]
        print(f"  {name:<5s} 上前沿 {int(gfront[lo:hi].sum()):>2d}/{len(d['pts']):<3d}  "
              f"最优线长 {r[:, 0].min():>10.4f}{args.wl_unit}  "
              f"最小面积 {r[:, 1].min():>8.1f}mm2  "
              f"最低峰温 {r[:, 2].min():>6.2f}C")

    print(f"  -> {out_png}")
    print(f"  -> {out_csv}")


if __name__ == "__main__":
    main()
