#!/usr/bin/env python3
"""plot_pareto3d.py — 画一个 case 的三维帕累托前沿: 总线长 / 外接框面积 / 峰值温度。
三个目标全部越小越好, 所以 "支配" 定义为: p 三项都不比 q 差、且至少一项更好。
每个方法各自算自己的帕累托前沿 (一个方法的解集能达到的最优权衡), 两个方法叠在
同一张图里对比。
三个目标的帕累托前沿一般是一张二维曲面而不是一条曲线, 所以除了画点, 还会铺一张曲面
(--surface): 三个目标里挑两个当底面、第三个当高度。默认的 front (见 front_surface) 就是
"支配点构成的那块面" —— 先把非支配解在底面上连成一张直线网, 再把这张网抹平, 于是每个非支配
解都严格落在面上, 面也只在非支配解张成的那块地面上铺, 没有解的地方不编面。曲面由两家的点
一起撑起来, 是共用的参照, 不属于任何一个方法, 所以用中性灰而不是分类色。
另一条路是 smooth (见 smooth_envelope): 经验前沿面 min{ 高度轴 : 另外两轴都不超过 } 的
柔化版, 铺满整个底面, 每个点都落在面上或面之上, 于是点到面的竖直距离就是它为另外两个
目标付出的代价; 代价是包络本身是分片常数, 取 min 保性质时会把平台贴回面上, 看着有台阶。
三轴默认是 x=总线长, y=面积, z=峰值温度 (--axes 可换成另外两种循环轮换), 于是曲面 =
给定线长和面积预算下能拿到的最低峰温。注意前沿的弯曲形状随 case 变, 不是一律朝原点凹:
实测 hp11_m 像碗, acend910 和 xerox8_m 更接近鞍面。用 --elev/--azim/--box-aspect 换视角。
换轴序会同时换掉曲面建的底面, 出来的图差别很大: 底两轴如果是采样扫描得比较开的那两个
(线长 × 面积), 曲面处处撑得住; 一旦把峰温放到轴上, acend910 这种 "低温解全挤在大面积端"
的 case 会在底面里拉出一道近垂直的崖 —— 那是数据的形状, 不是画法的问题。
数据源 (都是 eval_layout.py 的产物, 三列口径完全一致, 可以直接比):
  AT    AT_result/50set.csv
        case 列形如 acend910_wl00              —— wlsweep 的 50 个 wl_weight 点
 本方法 FM_result/pareto_weight_sweep_seed121_legal_only_results/result.csv
        case 列形如 acend910_thermal0p03_wirelength50
        —— thermal_weight × wirelength_weight 网格上的解
输出 (都写进本目录):
  <case>_pareto3d.png     三维图
  <case>_pareto_front.csv  图里高亮的非支配解, 用作表格视图 (便于核对具体数值)
用法:
  python plot_pareto3d.py acend910
  python plot_pareto3d.py hp11_m --elev 18 --azim -55
  python plot_pareto3d.py hp11_m --stems floor   # 竖线垂到画面底面
  python plot_pareto3d.py xerox8_m --lim all   # 全部解都进画面 (前沿会被离群点挤小)
  python plot_pareto3d.py acend910 --dark      # 深色底版本
【修改说明】已移除全部灰色曲面绘制代码，--surface 参数不再生效，不再渲染任何3D曲面。
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
from matplotlib.ticker import MaxNLocator  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402  (注册 3d projection)
# ========== 删除曲面相关依赖：Poly3DCollection、Delaunay、ConvexHull、插值器等全部曲面函数保留但不调用 ==========
from scipy.interpolate import (LinearNDInterpolator,  # noqa: E402
                              RegularGridInterpolator)
from scipy.ndimage import distance_transform_edt, gaussian_filter, label  # noqa: E402
from scipy.spatial import ConvexHull, Delaunay, cKDTree  # noqa: E402
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
AXIS_OF = {"wirelength": 0, "area": 1, "temp": 2}
# 两个方法: (图例名, csv 路径)。csv 里的 case 列 = <case> + 参数尾缀。
METHODS = [
    ("AT", RESULT_EVAL / "AT_result" / "50set.csv"),
    # ours_50set.csv = 权重扫描 + select_ours_supplement.py 补进来的 seed 扫描解,
    # 让 acend910 / cpu-dram 也有 50 个解, 与 AT 的 50 个对等。
    ("Ours", RESULT_EVAL / "FM_result" / "ours_50set.csv"),
]
# 分类色只按实体固定分配, 不随筛选后的顺序变: AT 恒为 slot1 (蓝), Ours 恒为 slot2 (橙)。
# 取自参考调色板, 已过 validate_palette.js (light/dark 两种 mode 的 all-pairs 检查全 PASS)。
THEME = {
    # light 的大背景 (画布 + 坐标区, 也就是立体盒子外面那圈) 用纯白 #ffffff,
    # 贴进白底论文/文档里不留灰边; 三个立体背面仍保持原来的浅灰 #f4f3ef,
    # 靠它撑出立体感, 别一起刷白 —— 刷白了盒子就没边了。
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#dedcd4", "pane": "#f4f3ef", "surf": "#6b6a66"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7",
             "grid": "#3a3a37", "pane": "#232322", "surf": "#8a8985"},
}
SERIES = {"AT": "#5ECCF3", "Ours": "#FF8021"}   # 两个方法的点色 (按用户指定, 深浅底同一套)
# 哪些方法的解名在写 csv 时要换成 <case>_id<NN> 的中性编号 (见 anonymize_labels)
ANON_METHODS = {"Ours"}
# 图里的文字一律英文 —— 本环境的 matplotlib 没有中文字体 (出图与中文控制台是两件事)。
CTX_SIZE, CTX_ALPHA = 26.0, 0.55        # 被支配解: 小点淡显, 只做背景
FRONT_SIZE, FRONT_ALPHA = 78.0, 0.95    # 非支配解: 大点 + 2px 表面色描边
SURF_ALPHA = 0.42                       # 曲面参数保留（不再使用）
MESH_ALPHA, MESH_STRIDE = 0.30, 9       # 曲面网格参数保留（不再使用）
FLAT_ALPHA, FLAT_EDGE_ALPHA = 0.16, 0.55
STEM_ALPHA = 0.30                       # 垂线: 只做位置参照, 不能盖过点本身
KERNEL_CELLS = 1.5                      # 下压核半径（不再使用）
PUSH_ROUNDS = 12                        # 下压迭代轮数（不再使用）
PUSH_SPREAD_CELLS = 20.0
def parse_axes(spec: str) -> list[int]:
    """--axes 的字符串 -> 三个原始列号 (依次是图的 x, y, z)。"""
    names = [s.strip().lower() for s in spec.split(",")]
    if sorted(names) != sorted(AXIS_OF):
        raise SystemExit(f"--axes 要把 wirelength / area / temp 各写一次, 收到 {spec!r}")
    return [AXIS_OF[n] for n in names]
def case_of(label: str) -> str:
    """两边 csv 的 case 列都带参数尾缀, 剥回 case 名。
    AT: <case>_wl<NN>                                  -> <case>
    FM: <case>_thermal<X>_wirelength<Y>        -> <case>
    """
    return label.split("_thermal", 1)[0].split("_wl", 1)[0].split("_seed", 1)[0]
def read_points(path: Path, case: str, default_source: str) -> list[dict]:
    """读一个 csv 里属于 case 的有效行。error 非空的行直接丢掉。

    source 列 (有就带上) 记这一行是哪来的 —— 权重扫描还是补进来的 seed 扫描,
    写进 _pareto_front.csv 便于回溯。
    """
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
            pts.append({"label": row["case"], "v": np.array(vals),
                        "source": (row.get("source") or "").strip() or default_source})
    return pts
def anonymize_labels(case: str, pts: list[dict]) -> None:
    """把解的名字换成 <case>_id<NN>, 原地改 pts 里每个点的 label。

    只有 "我们的方法" 需要换: AT 那边 csv 里本来就是 <case>_wl<NN> 这种中性编号,
    而 ours_50set.csv 的 label 带着权重参数 (<case>_thermal0p3_wirelength2), 补进来的
    还带 seed —— 一眼能看出这个解是哪次扫描、哪组权重来的。对外给个编号就够了。

    编号按原 label 的字典序分配, 只取决于解集内容, 跟文件行序无关, 重跑映射不变;
    两张 csv (前沿 / 全解) 共用同一份编号, 同一个解在两边 id 一致。
    """
    for i, p in enumerate(sorted(pts, key=lambda p: p["label"])):
        p["label"] = f"{case}_id{i:02d}"


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
# ===================== 下面所有曲面相关函数全部保留定义，但主函数不再调用 =====================
def envelope_height(vals: np.ndarray):
    """经验前沿面: z(u, v) = min{ z_i : x_i <= u 且 y_i <= v } (x/y 是图的底两轴, z 是高度轴)。"""
    wl, ar, tp = vals[:, 0], vals[:, 1], vals[:, 2]
    def height_at(pu, pv) -> np.ndarray:
        pu = np.atleast_1d(np.asarray(pu, dtype=float))
        pv = np.atleast_1d(np.asarray(pv, dtype=float))
        ok = (wl[None, :] <= pu[:, None]) & (ar[None, :] <= pv[:, None])
        z = np.where(ok, tp[None, :], np.inf).min(axis=1)
        return np.where(ok.any(axis=1), z, np.nan)
    return height_at
def support_count(vals: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    flat = ((vals[None, :, 0] <= U.ravel()[:, None])
            & (vals[None, :, 1] <= V.ravel()[:, None]))
    return flat.sum(axis=1).reshape(U.shape)
def push_down(Z: np.ndarray, vals: np.ndarray, U: np.ndarray, V: np.ndarray,
              rounds: int = 4, spread: float = 0.0) -> np.ndarray:
    if Z is None or np.isnan(Z).all():
        return Z
    sigma = KERNEL_CELLS / max(len(U[0]) - 1, 1)
    su = lambda a: (np.asarray(a, float) - U[0, 0]) / (U[0, -1] - U[0, 0])  # noqa: E731
    sv = lambda a: (np.asarray(a, float) - V[0, 0]) / (V[-1, 0] - V[0, 0])  # noqa: E731
    for _ in range(rounds):
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
        D = np.max(viol[bad][None, :] * np.exp(-d2 / 2.0), axis=1).reshape(Z.shape)
        if spread > 0:
            D = gaussian_filter(D, spread)
        Z = Z - D
    return Z
def spanned_mask(fn: np.ndarray, q: np.ndarray, max_edge: float) -> np.ndarray:
    try:
        tri = Delaunay(fn)
    except Exception:  # noqa: BLE001  (共面/共线, 点太少)
        return np.ones(len(q), bool)
    idx = np.array([[0, 1, 1, 2, 2, 0]])
    edges = fn[tri.simplices[:, idx[0, [0, 2, 4]]]] - fn[tri.simplices[:, idx[0, [1, 3, 5]]]]
    keep = np.linalg.norm(edges, axis=2).max(axis=1) <= max_edge
    simp = tri.find_simplex(q)
    inside = simp >= 0
    return inside & keep[np.where(inside, simp, 0)]
NET_ROUNDS = 90        # 抹平的轮数
NET_SIGMA = 1.6        # 每轮模糊的半径 (格宽)
NET_TAIL = 14
MASK_ROUND_CELLS = 3.0
def footprint_mask(vn: np.ndarray, q: np.ndarray, max_edge: float) -> np.ndarray:
    try:
        tri = Delaunay(vn)
    except Exception:  # noqa: BLE001  (共面/共线, 点太少)
        return np.ones(len(q), bool)
    simp = tri.find_simplex(q)
    inside = simp >= 0
    if max_edge < 1.0:
        idx = np.array([[0, 1, 1, 2, 2, 0]])
        edges = vn[tri.simplices[:, idx[0, [0, 2, 4]]]] - vn[tri.simplices[:, idx[0, [1, 3, 5]]]]
        keep = np.linalg.norm(edges, axis=2).max(axis=1) <= max_edge
        inside = inside & keep[np.where(inside, simp, 0)]
    return inside | (cKDTree(vn).query(q)[0] <= 0.04)
def front_surface(vals: np.ndarray, reach: float):
    f = vals[pareto_mask(vals)]
    if len(f) < 4:
        return None
    lo, hi = vals[:, :2].min(axis=0), vals[:, :2].max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    fn = (f[:, :2] - lo) / span
    try:
        tri = Delaunay(fn)
    except Exception:  # noqa: BLE001  (共面/共线, 点太少)
        return None
    simp = tri.simplices
    if reach > 0:
        edge = np.linalg.norm(fn[simp] - fn[np.roll(simp, 1, axis=1)], axis=2)
        simp = simp[edge.max(axis=1) <= reach]
    if not len(simp):
        return None
    tris = f[simp]
    net_at = LinearNDInterpolator(tri, f[:, 2])
    def height_at(pu, pv) -> np.ndarray:
        pu = np.atleast_1d(np.asarray(pu, float))
        pv = np.atleast_1d(np.asarray(pv, float))
        return net_at(np.c_[pu, pv])
    adrift = int((np.bincount(simp.ravel(), minlength=len(f)) == 0).sum())
    print(f"  曲面: 支配点直线网, {len(simp)} 块面片, 用到 {len(f) - adrift}/{len(f)} 个支配点")
    if adrift:
        print(f"  注意: {adrift} 个支配点一块面片都没摊到 (--front-reach {reach} 把它的三角形"
              f"全砍了), 图上那就只剩一个孤零零的点")
    return tris, height_at
def surface_hull(fv: np.ndarray) -> np.ndarray | None:
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
    ap.add_argument("--axes", default="wirelength,area,temp",
                    help="三个目标依次放 x,y,z 轴 (默认 wirelength,area,temp)。三者只能各出现"
                         "一次, 另外两种排法是 area,temp,wirelength 和 temp,wirelength,area。")
    ap.add_argument("--lim", choices=["front", "all"], default="front",
                    help="坐标轴范围按什么取: front=两个方法的前沿并集 (默认, 前沿撑满画面), "
                         "all=全部解 (离群点会把轴拉长、前沿挤成一坨, 一般只在要展示全貌时用)")
    # ========= 保留参数，但不再做任何曲面渲染 =========
    ap.add_argument("--surface", choices=["front", "smooth", "envelope", "hull", "none"],
                    default="none",
                    help="[已禁用曲面渲染] 仅保留参数占位，所有选项都不会绘制曲面")
    ap.add_argument("--surface-grid", type=int, default=200, help="[无作用]")
    ap.add_argument("--surface-smooth", type=float, default=0.03, help="[无作用]")
    ap.add_argument("--surface-mesh", action="store_true", help="[无作用]")
    ap.add_argument("--front-reach", type=float, default=0.75, help="[无作用]")
    ap.add_argument("--surface-min-support", type=int, default=0, help="[无作用]")
    ap.add_argument("--stems", choices=["none", "floor", "surface"], default="floor",
                    help="给每个点画一根竖线当位置参照: floor=垂到画面底面 (默认); surface模式已失效，自动退化为floor")
    ap.add_argument("--line", action="store_true",
                    help="把前沿点按总线长排序连成折线。默认不画 —— 三个目标的帕累托前沿"
                         "一般是一张曲面而不是一条曲线, 一维排序连出来的线会自己穿过自己, "
                         "容易看成比实际更规整的前沿。只在确定前沿退化成一维时才开")
    ap.add_argument("--wl-unit", choices=["mm", "m"], default="m",
                    help="总线长轴的单位 (默认 m)。只影响画图, 写出的 csv 恒为 mm")
    ap.add_argument("--dark", action="store_true", help="深色底")
    args = ap.parse_args()
    th = THEME["dark" if args.dark else "light"]
    series_color = SERIES
    wl_scale = WL_SCALE[args.wl_unit]
    perm = parse_axes(args.axes)
    axis_labels = [OBJECTIVES[i][1].format(wl=args.wl_unit) for i in perm]
    data = {}
    for name, path in METHODS:
        pts = read_points(path, args.case, default_source=name.lower())
        if not pts:
            continue
        if name in ANON_METHODS:
            # 换掉带参数的 label, 只留 <case>_id<NN>; 要在算前沿之前换, 否则
            # 后面写 csv 时用的还是旧 label (前沿本身跟 label 无关, 不受影响)
            anonymize_labels(args.case, pts)
        raw = np.vstack([p["v"] for p in pts])
        raw[:, 0] *= wl_scale     # vstack 已经是副本, 改它不会动 pts, csv 表仍是 mm
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
    ax.computed_zorder = False
    # 两个方法的解合成一个点集算一条全局前沿
    all_vals = np.vstack([d["vals"] for d in data.values()])
    gfront = pareto_mask(all_vals)
    gfv = all_vals[gfront]
    # 各方法在 all_vals 里的行区间, 用来把全局前沿的掩码切回每个方法
    span, off = {}, 0
    for name, d in data.items():
        span[name] = (off, off + len(d["pts"]))
        off += len(d["pts"])
    lims = axis_lims(gfv if args.lim == "front" else all_vals)
    # ============ 【核心改动】完全移除曲面绘制逻辑，height_at置空 ============
    height_at = None
    # 不管 --surface 参数是什么，全部跳过曲面渲染
    n_hidden = {name: 0 for name in data}
    for name, d in data.items():
        c = series_color[name]
        v = d["vals"]
        # 范围外的点不画 (三维里画到框外会糊在图上), 但要数出来告诉用户
        shown = inside(v, lims)
        n_hidden[name] = int((~shown).sum())
        lo, hi = span[name]
        front = gfront[lo:hi] & shown         # 该方法里落在全局前沿上的解
        fv = v[front]
        # 被支配解: 淡色小点, 颜色仍跟着方法走 (不做按名次的重新配色)
        ctx = shown & ~gfront[lo:hi]
        ax.scatter(v[ctx, 0], v[ctx, 1], v[ctx, 2],
                   s=CTX_SIZE, c=c, alpha=CTX_ALPHA, depthshade=False,
                   linewidths=0, zorder=2)
        # 垂线: 三维里最容易看丢的就是点的前后深度, 一根竖线把点钉在地面
        if args.stems != "none":
            # surface模式失效，强制垂到底面
            z_bot = np.full(len(v), lims[2][0])
            keep = shown & np.isfinite(z_bot) & (v[:, 2] > z_bot)
            n = int(keep.sum())
            if n:
                # 每条线两个端点, 第三行塞 NaN 把各条线断开
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
    # 温度轴的轴标题要跟着刻度一起往外让 —— 刻度 pad 提到 14 以后, 数字正好压在
    # 标题原来的位置上, 不推开会叠字。3d 里标题偏移同样由 labelpad 控 (axis3d.py
    # 的 _draw_offset_text), 刻度让出多少, 标题就得让出差不多多少。
    ax.set_zlabel(axis_labels[2], color=th["ink2"], fontsize=11, labelpad=15)
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
    # 坐标刻度的数字: 三维里刻度是斜着看的, 原来 9 号偏小, 提到 12 (跟轴标题
    # 的 11 号同量级, 不再是一眼看不见的脚注大小)。
    ax.tick_params(colors=th["ink2"], labelsize=12)
    # 刻度和轴的距离逐轴微调。3d 里刻度文字的偏移量是 (tick.get_pad() + 8) * points
    # (mpl_toolkits/mplot3d/axis3d.py), 但同样的 pad 经投影后各轴观感差很多 ——
    # 竖轴 (温度) 的数是贴着轴的, 横轴 (线长) 的数离得远。默认 pad 都是 3.5,
    # 所以这里把温度轴往外推、线长轴往里收, 让两边的留白看起来一致。
    # 面积轴 (y) 的默认观感正好, 不动。
    ax.zaxis.set_tick_params(pad=6)     # 温度: 原来是贴轴的, 往外让一点点
    ax.xaxis.set_tick_params(pad=1)     # 线长: 原来是飘着的, 往里收一点点
    # 字号大了以后, 面积轴自动取的那 8 个刻度 (1300 步长 100) 会自己撞在一起,
    # 所以把三个轴的刻度数都压到 6 以内 —— 3d 里刻度是斜的, 刻度一密就叠字。
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        # steps 限定成 1/2/2.5/5/10 这几档, 免得 3d 自己挑出 150 这种不整的步长
        axis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
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
    # 两张 csv: 一张只放帕累托前沿, 一张放两个方法在该 case 的全部解
    # (100 行 = AT 50 + Ours 50), 两张都带 source / in_front / shown 便于回溯。
    def write_csv(path: Path, only_front: bool) -> int:
        n = 0
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "label", "source", "total_wirelength_mm",
                        "bbox_area_mm2", "max_temp_C", "in_front", "shown"])
            for name, d in data.items():
                lo, hi = span[name]
                shown = inside(d["vals"], lims)
                for p, is_f, vis in zip(d["pts"], gfront[lo:hi], shown):
                    if only_front and not is_f:
                        continue
                    w.writerow([name, p["label"], p["source"],
                                *[f"{x:.6f}" for x in p["v"]], int(is_f), int(vis)])
                    n += 1
        return n

    out_csv = out_png.parent / f"{args.case}_pareto_front.csv"
    out_all = out_png.parent / f"{args.case}_all_solutions.csv"
    n_front = write_csv(out_csv, only_front=True)
    n_all = write_csv(out_all, only_front=False)
    print(f"[{args.case}] 数据源: " + ", ".join(f"{n}={len(d['pts'])}解" for n, d in data.items())
          + f"   坐标范围按 lim={args.lim}")
    print(f"  轴: x={args.axes.split(',')[0]}, y={args.axes.split(',')[1]}, "
          f"z={args.axes.split(',')[2]} (曲面已关闭)")
    if any(n_hidden.values()):
        print(f"  (范围外的被支配解没画: "
              + ", ".join(f"{n}={c}" for n, c in n_hidden.items() if c)
              + "; 想看全貌加 --lim all)")
    print(f"  全局前沿 (两方法合起来算) {int(gfront.sum())} 个解:")
    for name, d in data.items():
        lo, hi = span[name]
        r = d["raw"]
        print(f"  {name:<5s} 上前沿 {int(gfront[lo:hi].sum()):>2d}/{len(d['pts']):<3d}  "
              f"最优线长 {r[:, 0].min():>10.4f}{args.wl_unit}  "
              f"最小面积 {r[:, 1].min():>8.1f}mm2  "
              f"最低峰温 {r[:, 2].min():>6.2f}C")
    print(f"  -> {out_png}")
    print(f"  -> {out_csv}   (前沿 {n_front} 行)")
    print(f"  -> {out_all}  (全部解 {n_all} 行)")
if __name__ == "__main__":
    main()
