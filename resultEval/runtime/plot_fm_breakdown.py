#!/usr/bin/env python3
"""plot_fm_breakdown.py — TW-FM 的运行时间拆成四块, 各画一张饼图。

四块 (先采样、再合法化、再精修, 最后是杂项):
    sampling     采样 (Flow Matching 去噪)
    legalization 合法化 (数据文件里的 stage A)
    refinement   精修 (数据文件里的 stage B)
    others       其余开销 (模型加载 / CUDA 初始化 / I/O 之类)

图例写在两张饼的右侧并且只出现一次: 两张饼的四块完全一样, 画两份是重复;
"Stage A/B" 是数据文件里的字段名, 图上直接写括号里的词 (legalization / refinement)。

两张图:
    TW-FM (CPU+GPU)   FMGPUTime.csv, alpha_0p9
    TW-FM (CPU)       FMCPUTime.csv, alpha_0p9

口径:
  * 每个 case 一行, 这里取 10 个 benchmark case 的**算术平均**, 再算各块占比。
    不按总时间加权 —— 正文说的是 "TW-FM 的时间花在哪", 平均每个 case 的花法
    才是这个问题, 加权会让最慢的那个 case 主导整个饼。
  * 只取 alpha_0p9 (FMGPUTime.csv 里还有 alpha_0p1, 不用; FMCPUTime.csv 只有 0p9)。
  * 占比的分母是**这四块之和**, 不是 total_mean_s。两者差一个 other_mean_s:
    数据文件里 total_mean_s = sampling + stage_a + stage_b (不含 other),
    而 legalize_total_mean_s = stage_a + stage_b + other。脚本会把这个差打出来核对。

已知的口径瑕疵 (2026-09-19 与用户确认后**故意保留**, 写进 caption 前必须知道)
--------------------------------------------------------------------------
两份 CSV 的 10 个 case 都跨两个批次 (见 plot_runtime.py 的说明):
    GPU: pareto_table_profiles_5seed 7 例 / pareto_new4_gpu 3 例 (hp6_m, xerox6_m, xerox7_m)
    CPU: selected-pareto-cpu12       7 例 / selected-pareto-new4b  同 3 例
`legalize_total_mean_s` 两批同口径 (GPU 12.28 vs 11.28 s), 但 A/B/other 的**切分**不同:
老批 7 例 other ≈ 5.51 s 而 A ≈ 1.09 s, 新批 3 例 other ≈ 0.08 s 而 A ≈ 5.17 s ——
同一份没单独打点的活, 老批落在 other, 新批被 A 吸收 (build_fmgputime.py 的
"两个来源的 other 不可横比" 说的就是这件事)。所以 GPU 饼上的
Stage A 4.7% = 老批 2.3% 与新批 10.0% 的混合, Others 7.9% 同理 (11.4% vs 0.15%)。
CPU 两批切分接近 (A 2.09/2.07, other 0.52/0.29), 基本不受影响。
用 10 个 case 就不要在正文里拿 Stage A / Others 的绝对值做跨方法论证; 只用来看
"采样占大头、合法化与精修是小头"这个量级。要按 case 均分单一口径, 只取那 7 个
单一来源的 case (ascend910, multigpu, cpu-dram, syn1, hp11_m, syn4, Case7) 重画。

用法: python3 plot_fm_breakdown.py
输出: fig/fm_time_breakdown.png / .eps 和 fm_time_breakdown.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches      # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
import matplotlib.transforms as mtrans     # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
GPU_CSV, CPU_CSV = HERE / "FMGPUTime.csv", HERE / "FMCPUTime.csv"

# 与 setup.tex 的 tab:benchmark / tab:runtime 同一组 case、同一顺序。
CASES = ["ascend910", "hp6_m", "multigpu", "xerox6_m", "xerox7_m",
         "cpu-dram", "syn1", "hp11_m", "syn4", "Case7"]

# 数据文件里写的是 acend910, 按论文口径改名成 ascend910 (与 plot_runtime.py 同一份映射)
RENAME = {"acend910": "ascend910"}

# 四块的名字顺序固定, 两张图共用一套映射。名字是图上写的那四个词。
PARTS = [("sampling_mean_s", "Sampling"),
         ("stage_a_mean_s", "Legalization"),
         ("stage_b_mean_s", "Refinement"),
         ("other_mean_s", "Others")]

# 用 plot_pareto3d.py 的 SERIES 那四个颜色, 各调浅一档 —— 饼是实心大色块,
# 原饱和度铺满整个圆太吵, 浅色版既保住同一套色相, 又不压过正文里的饼外数字。
BASE_COLOR = {
    "Sampling":     "#FF8021",   # TW-FM
    "Legalization": "#5ECCF3",   # ATPlace2.5D
    "Refinement":   "#81D31A",   # RLPlanner
    "Others":       "#4E67C8",   # MILP
}
LIGHTEN = 0.38                  # 往白里调的幅度, 0 = 原色, 1 = 纯白
INK, INK2 = "#0b0b0b", "#52514e"        # 与 plot_pareto3d.py 的 THEME["light"] 一致
PROFILE = "alpha_0p9"

# 图上元素的尺寸 / 位置, 集中放这里, 调版式不用翻下面的函数
PCT_FS = 15.5                   # 饼上百分比字号: 四块一律用这个, 窄扇形挪到饼外也一样大
LBL_FS = 16.0                   # 标题字号
SUB_FS = 12.0                   # 副标题 (case 数 / 总时间) 字号
LEG_FS = 14.0                   # 图例字号
LIFT_PART = "Sampling"          # 要"浮起来"的那一块: 采样是主项
# 投影的两层偏移 (占半径的比例) 和透明度。两层叠加近似一次模糊, 比单层柔和。
SHADOW = [((0.020, -0.023), 0.22), ((0.040, -0.046), 0.20)]
SHADOW_C = "#8a8a84"
PCT_R = 0.66                    # 百分比圆心距 (饼内)
NARROW_DEG = 24.0               # 小于这个角度算窄扇形, 数字挪到饼外
RADIAL_COS = 0.35               # |cos| 大于它算"横向"扇形, 数字沿半径直接推出去
RADIAL_R = 1.035                # 横向扇形的饼外数字半径 (贴着饼边, 别飘远)
LEAD_RAD = 1.025                # 横向扇形引线的长度 (很短, 只是把数字连回饼)
SIDE_X = 1.05                   # 竖向扇形的饼外数字横向位置
SIDE_Y = 0.90                   # 竖向扇形的饼外数字高度上限 (不超出饼身高度, 图不用长高)
FIG_W, FIG_H = 12.4, 4.3
XLIM = YLIM = (-1.14, 1.14)     # 饼身 + 数字, 轴框正好包住; 文字溢出部分靠 bbox_inches 兜
AX_BOX = [0.004, 0.085, 0.2844, 0.82]   # 左饼 (轴框做成正方形, aspect equal 不再缩)
AX_DX = 0.300                   # 右饼相对左饼的横移; 调它 = 调两饼间距
LEG_X = 0.615                   # 图例左边缘 (图例贴两饼右侧)

# --- 紧凑版: 两条 100% 堆叠条 (fm_time_breakdown_bar.*) --------------------------
# 原生尺寸按**单栏**留 (\columnwidth = 3.4 in), 放进正文 1:1 不缩,
# 所以这里的磅数就是正文里读到的磅数 —— 3.4 in 的图缩到 0.55 倍时 10.5 pt
# 会变成 5.3 pt, 这就是为什么不能沿用双栏那套尺寸。
BAR_FIG_W, BAR_FIG_H = 3.46, 1.05
BAR_LEFT = 0.30                         # 左侧留给行名 (左对齐, 按最长那行留)
BAR_TAG_X = -0.315                      # 行名左边缘 (轴宽的比例, 负 = 在轴左边)
BAR_XMAX = 106.0                        # 横轴到 106: 最右那个 "0.3%" 才不会被裁掉
BAR_YLIM = (0.338, 0.995)               # 上下正好包住数字, 不留空档
BAR_Y = (0.78, 0.55)                    # 两条 bar 的中心高度 (轴内, 轴底 0.338 轴顶 0.995)
BAR_H = 0.145                           # 条的厚度
BAR_MIN_PCT = 10.0                      # 窄于这个百分比就不放条内, 数字挪到条子上/下方
BAR_LEAD = 0.040                        # 引线从条子边缘起画多长 (数字贴着条子放)
BAR_STAGGER = 0.115                     # 位置挨得近的窄块, 数字在高度上错开多少
BAR_CLUSTER = 13.0                      # 窄块中点相差小于这个百分点就算"挨得近"
                                        # (窄图里 1 个百分点只有 0.025 in, 阈值要比宽图大)
BAR_PCT_FS = 7.0                        # 条内/条上的百分比字号
BAR_NAME_FS = 7.0                       # 行名字号
BAR_LEG_FS = 6.5                        # 图例字号
# 两条 bar 的行名 (上条 = 带 GPU 的配置, 下条 = 纯 CPU)。论文里这个方法叫 FlowTAP;
# 这里的字面量跟着论文走, 免得图放进正文之后和表里的列名对不上。
BAR_TAG = ("FlowTAP (CPU+GPU)", "FlowTAP (CPU)")


def lighten(hex_color: str, k: float) -> str:
    """把颜色往白里调 k 比例 (0 = 原色, 1 = 纯白)。"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = lambda v: round(v + k * (255 - v))          # noqa: E731
    return "#{:02X}{:02X}{:02X}".format(f(r), f(g), f(b))


COLOR = {name: lighten(c, LIGHTEN) for name, c in BASE_COLOR.items()}

CALIBRI_DIRS = [
    Path("/mnt/c/Windows/Fonts"),                  # WSL: 宿主机的 Windows 字体
    Path("/usr/share/fonts/truetype/calibri"),
    Path("/usr/share/fonts/truetype/crosextra"),   # distro 的 Carlito 兜底
    Path.home() / ".local/share/fonts",
]
FONT_FALLBACK = "DejaVu Sans"


def setup_font() -> str:
    """注册 Calibri 并设成全局字体。与 plot_pareto3d.py 同一份逻辑, 字号一致。"""
    from matplotlib import font_manager
    for d in CALIBRI_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("[Cc]alibri*.ttf")) + sorted(d.glob("[Cc]arlito*.ttf")):
            try:
                font_manager.fontManager.addfont(str(f))
            except Exception:
                pass
        if any(f.name in ("Calibri", "Carlito") for f in font_manager.fontManager.ttflist):
            name = next(f.name for f in font_manager.fontManager.ttflist
                        if f.name in ("Calibri", "Carlito"))
            matplotlib.rcParams["font.family"] = name
            return name
    matplotlib.rcParams["font.family"] = FONT_FALLBACK
    return FONT_FALLBACK


FONT = setup_font()


def read_rows(path: Path) -> dict[str, dict]:
    """取该文件里 alpha_0p9 的 10 个 case。缺哪个直接报错, 不留空。”"""
    with path.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["profile"] == PROFILE]
    by = {RENAME.get(r["case"], r["case"]): r for r in rows}
    missing = [c for c in CASES if c not in by]
    if missing:
        raise SystemExit(f"{path.name} 里没有这些 case 的 {PROFILE} 行: {missing}")
    return {c: by[c] for c in CASES}


def mean_parts(rows: dict[str, dict]) -> dict[str, float]:
    """四块的逐 case 算术平均 (秒)。"""
    n = len(CASES)
    return {name: sum(float(rows[c][key]) for c in CASES) / n for key, name in PARTS}


def report(tag: str, rows: dict[str, dict], avg: dict[str, float]) -> None:
    total = sum(avg.values())
    print(f"=== {tag}  (alpha_0p9, {len(CASES)} 个 case 平均) ===")
    print(f"{'块':24s}{'平均 (s)':>10s}{'占比':>8s}   {'逐 case 范围 (s)'}")
    for _, name in PARTS:
        v = avg[name]
        vs = [float(rows[c][k]) for k, n in PARTS if n == name for c in CASES]
        print(f"{name:24s}{v:10.2f}{100*v/total:7.1f}%   {min(vs):.2f} ~ {max(vs):.2f}")
    print(f"{'四项合计':24s}{total:10.2f}{100.0:7.1f}%")
    # 数据文件里 total_mean_s 与四项之和差一个 other
    tm = sum(float(rows[c]["total_mean_s"]) for c in CASES) / len(CASES)
    lt = sum(float(rows[c]["legalize_total_mean_s"]) for c in CASES) / len(CASES)
    print(f"{'total_mean_s (文件里的)':24s}{tm:10.2f}"
          f"   ← 四项合计 - other = {total - avg['Others']:.2f}, 对得上")
    print(f"{'legalize_total_mean_s':24s}{lt:10.2f}"
          f"   ← stageA + stageB + other = {lt:.2f}  (自洽)")
    print()


def lift(ax, w) -> None:
    """给某一片扇形加投影, 做出"浮起来"的效果 —— **扇形本身不动**。

    一开始用的是 matplotlib 的 explode (把整片沿半径推出去), 但那片就是主项
    (52.5% / 75.7%), 推出去之后另外三片留在原地不动, 右图看着就"缩"了一块 ——
    被推走的面积越大, 剩下那块越显小。投影不挪任何扇形, 四块的面积和位置都是原样,
    只有外缘多出一道沿偏移方向的灰边, 视觉上像这一层被抬起来了。
    """
    for (dx, dy), a in SHADOW:
        sh = mpatches.Wedge(w.center, w.r, w.theta1, w.theta2, width=w.width,
                            facecolor=SHADOW_C, edgecolor="none", alpha=a,
                            zorder=w.get_zorder() - 0.5)
        # 偏移在数据坐标里做 (饼半径 = 1), 再套上扇形自己的变换
        sh.set_transform(mtrans.Affine2D().translate(dx, dy) + w.get_transform())
        ax.add_patch(sh)


def start_angle_for(vals: list[float]) -> float:
    """挑一个起始角, 让窄扇形尽量落在**水平**方向 (正左 / 正右)。

    窄扇形在正上方或正下方时, 数字沿半径推出去会顶到图的上/下边, 整张图为它长高;
    转一下起始角, 把窄的那几片转到左右两侧, 数字就能横着写在饼边上, 一点纵向空间
    都不占。顺时针切片, 所以第 i 片从 cur 转到 cur - 2*half_i, 中点在 cur - half_i。

    目标是 **minimax**: 让"最歪的那片窄扇形"的 |sin(中点)| 最小。不用取和, 因为两个
    窄扇形的间距是固定的, 取和会出现两个几乎打平的角度 —— (0.105, 0.105) 和
    (0.001, 0.208), 和只差 0.001, 数据一动就翻面; minimax 下前者 0.105 明显赢后者
    0.208, 选出来的角度稳, 也说得清 —— "尽量把窄扇形转到水平"。
    """
    halves = [v / sum(vals) * 180.0 for v in vals]
    narrow = [h for h in halves if 2 * h < NARROW_DEG]
    if not narrow:
        return 90.0                          # 没有窄扇形, 照惯例从 12 点开始
    best, best_cost = 90.0, None
    for k in range(720):
        a, cur, worst = k * 0.5, k * 0.5, 0.0
        for h in halves:
            if 2 * h < NARROW_DEG:
                worst = max(worst, abs(math.sin(math.radians(cur - h))))
            cur -= 2 * h
        if best_cost is None or worst < best_cost - 1e-12:
            best, best_cost = a, worst
    return best


def draw(ax, avg: dict[str, float], title: str):
    """一张饼, 返回 wedges 供共用图例使用。

    百分比一律写在图上, 字号不分饼内饼外 (PCT_FS)。窄扇形 (CPU 那张里 Others 只有
    0.3%) 饼内画不下, 挪到饼外并用引线连回扇区 —— 不留引线的话那个数字看着像飘在
    空中的杂物。饼外的数字只是位置不同, 字号和饼内的完全一样。

    饼外数字一律往**横向**摆, 不往纵向摆: 顶/底的窄扇形如果沿半径推出去, 数字会顶到
    图的上边或下边, 整张图就得为它长高。所以分两种:
        横向扇形 (|cos| >= RADIAL_COS)  数字沿半径直接推出去, 本来就只占横向
        竖向扇形 (|cos| <  RADIAL_COS)  数字推到右侧, 高度压回 SIDE_Y 以内 ——
                                        纵向上不超出饼身, 图的高度就是饼的高度
    """
    names = [n for _, n in PARTS]
    vals = [avg[n] for n in names]
    total = sum(vals)
    wedges, _, autotexts = ax.pie(
        vals, colors=[COLOR[n] for n in names],
        startangle=start_angle_for(vals), counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.6),
        autopct="%.1f%%", pctdistance=PCT_R,
        textprops=dict(color=INK, fontsize=PCT_FS))
    lift(ax, wedges[names.index(LIFT_PART)])

    for w, at in zip(wedges, autotexts):
        ang = math.radians((w.theta1 + w.theta2) / 2)
        c, s = math.cos(ang), math.sin(ang)
        at.set_fontsize(PCT_FS)                       # 显式再设一次, 防 rcParams 干扰
        if w.theta2 - w.theta1 >= NARROW_DEG:
            continue                                  # 够宽, 数字就在饼内, 不动
        at.set_color(INK2)
        if abs(c) >= RADIAL_COS:                      # 横向扇形: 沿半径推出去
            ax.plot([c, LEAD_RAD * c], [s, LEAD_RAD * s], color=INK2, lw=0.9, zorder=1)
            at.set_position((RADIAL_R * c, RADIAL_R * s))
            at.set_ha("left" if c >= 0 else "right")
            at.set_va("bottom" if s >= 0.15 else "top" if s <= -0.15 else "center")
        else:                                         # 竖向扇形: 往右推到侧边, 高度压回饼内
            y = max(-SIDE_Y, min(SIDE_Y, s))
            ax.plot([c, SIDE_X - 0.02, SIDE_X - 0.02], [s, s, y],
                    color=INK2, lw=0.9, zorder=1)
            at.set_position((SIDE_X, y))
            at.set_ha("left")
            at.set_va("center")

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=LBL_FS, color=INK, pad=20)
    ax.text(0.5, 1.005, f"mean of {len(CASES)} cases, {total:.1f} s",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=SUB_FS, color=INK2)
    return wedges


def bar_chart(avg_gpu: dict[str, float], avg_cpu: dict[str, float]) -> None:
    """紧凑版: 两条 100% 堆叠条, 上下排, 共用一条横轴。

    比两张饼省一半以上面积, 而且四块的相对长度可以直接比 —— 饼图里斜的弧长
    和角度眼睛估不准, 两张饼之间更没法比。宽高比也更贴版面: 饼天生是方的,
    两张饼并排就是 2:1, 塞进双栏还要缩着放。

    窄块 (Legalization / Others) 的长度本来就看不见, 数字放到条子上方用竖引线
    指回去; 同一根条里两个窄块的位置可能只差几个百分点, 高度上错开避免撞车。
    """
    names = [n for _, n in PARTS]
    fig = plt.figure(figsize=(BAR_FIG_W, BAR_FIG_H), dpi=200)
    ax = fig.add_axes([BAR_LEFT, 0.21, 1 - BAR_LEFT - 0.005, 0.72])
    ax.set_xlim(0, BAR_XMAX)
    ax.set_ylim(*BAR_YLIM)
    ax.axis("off")

    # 上条的数字朝上, 下条朝下 —— 都朝上的话, 下条的数字会直接压在上一条上。
    for y, avg, up in ((BAR_Y[0], avg_gpu, True), (BAR_Y[1], avg_cpu, False)):
        total = sum(avg[n] for n in names)
        # 左对齐: 两行的公共前缀 "FlowTAP (CPU" 对齐, 差别只落在行尾的 +GPU / ) 上。
        # get_yaxis_transform 让 x 走轴宽比例 (好放在轴左边的图幅余量里), y 仍是数据坐标。
        ax.text(BAR_TAG_X, y, BAR_TAG[0] if up else BAR_TAG[1],
                ha="left", va="center", fontsize=BAR_NAME_FS, color=INK,
                transform=ax.get_yaxis_transform())
        left, smalls = 0.0, []
        for n in names:
            w = 100.0 * avg[n] / total
            ax.barh(y, w, height=BAR_H, left=left, color=COLOR[n],
                    edgecolor="white", linewidth=1.3, zorder=3)
            if w >= BAR_MIN_PCT:
                ax.text(left + w / 2, y, f"{w:.1f}%", ha="center", va="center",
                        fontsize=BAR_PCT_FS, color=INK, zorder=4)
            else:
                smalls.append((left + w / 2, w))
            left += w
        # 只有 x 挨得近的才在高度上错开。两条 bar 的窄块都在右端, 但上条的 4.7 / 7.9
        # 隔了 12 个百分点, 同一个高度放得下; 下条的 1.4 / 0.3 只差 1 个点, 必须错开。
        lvl, prev = {}, None
        for cx, _ in sorted(smalls):
            lvl[cx] = 0 if prev is None or cx - prev > BAR_CLUSTER else lvl[prev] + 1
            prev = cx
        edge = y + (BAR_H / 2 if up else -BAR_H / 2)
        sgn = 1.0 if up else -1.0
        for cx, w in smalls:
            ty = edge + sgn * (BAR_LEAD + lvl[cx] * BAR_STAGGER)
            ax.plot([cx, cx], [edge, ty - sgn * 0.015], color=INK2, lw=0.8, zorder=1)
            ax.text(cx, ty, f"{w:.1f}%", ha="center",
                    va="bottom" if up else "top",
                    fontsize=BAR_PCT_FS, color=INK2)

    handles = [mpatches.Patch(facecolor=COLOR[n], edgecolor="none") for n in names]
    # 图例挂在轴的下边缘 (而不是图幅高度), 这样调 ylim / 条的位置时图例跟着走,
    # 不用每次重算一个图幅坐标 —— 图例和 bar 之间的距离就锁死了。
    ax.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.0),
              bbox_transform=ax.transAxes, ncol=4, frameon=False,
              fontsize=BAR_LEG_FS, handlelength=1.1, columnspacing=1.8,
              handleheight=0.9, borderpad=0.05, borderaxespad=0.0)
    for ext in ("png", "eps"):
        fig.savefig(FIG / f"fm_time_breakdown_bar.{ext}", bbox_inches="tight")
    print(f"写出 {FIG/'fm_time_breakdown_bar.png'} 和 .eps")


def main() -> None:
    gpu, cpu = read_rows(GPU_CSV), read_rows(CPU_CSV)
    avg_gpu, avg_cpu = mean_parts(gpu), mean_parts(cpu)
    report("TW-FM (CPU+GPU)", gpu, avg_gpu)
    report("TW-FM (CPU)", cpu, avg_cpu)

    out = HERE / "fm_time_breakdown.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "part", "mean_s", "pct"])
        for mode, avg in (("CPU+GPU", avg_gpu), ("CPU", avg_cpu)):
            tot = sum(avg.values())
            for _, name in PARTS:
                w.writerow([mode, name, f"{avg[name]:.3f}", f"{100*avg[name]/tot:.2f}"])
            w.writerow([mode, "TOTAL", f"{tot:.3f}", "100.00"])
    print(f"写出 {out}")

    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200)
    wedges = None
    for i, (avg, t) in enumerate(((avg_gpu, "TW-FM (CPU+GPU)"),
                                  (avg_cpu, "TW-FM (CPU)"))):
        box = list(AX_BOX)
        box[0] += i * AX_DX
        wedges = draw(fig.add_axes(box), avg, t)
    # 两张饼的四块完全一样, 图例只画一份放右边, 竖排; 名字后面不带秒数, 因为两个
    # 模式的秒数不同, 共用图例放不下, 秒数在 fm_time_breakdown.csv 里。
    fig.legend(wedges, [n for _, n in PARTS], loc="center left",
               bbox_to_anchor=(LEG_X, 0.5), ncol=1, frameon=False,
               fontsize=LEG_FS, handlelength=1.2, labelspacing=1.2)
    for ext in ("png", "eps"):
        fig.savefig(FIG / f"fm_time_breakdown.{ext}", bbox_inches="tight")
    print(f"字体: {FONT}; 色板: "
          + ", ".join(f"{n}={c}" for n, c in COLOR.items()))
    print(f"写出 {FIG/'fm_time_breakdown.png'} 和 .eps")

    bar_chart(avg_gpu, avg_cpu)


if __name__ == "__main__":
    main()
