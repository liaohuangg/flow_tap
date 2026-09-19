"""汇总四种方法的运行时间, 出 CSV / LaTeX 表 / 对数刻度点名图。

数据源 (都在本目录下):
  AT.csv          ATPlace2.5D     -> total_time_mean_s
  ILP.csv         MILP            -> wall_s
  FMCPUTime.csv   TW-FM (CPU)     -> total_mean_s
  FMGPUTime.csv   TW-FM (CPU+GPU) -> sampling_mean_s + stage_a_mean_s + stage_b_mean_s
  RLPlanner       论文给的预算, 恒为 3600 s (没有逐 case 的实测值)

口径注意:
  * FM 两份文件都有 alpha 参数, 这里统一取 alpha_0p9 —— 因为 FMCPUTime.csv
    只有 alpha_0p9 一个 profile, CPU 和 CPU+GPU 必须同 alpha 才可比。
  * FMGPUTime.csv 里 n_seeds=5, FMCPUTime.csv 里 n_seeds=1, 前者是多次平均。
  * FMGPUTime.csv 有 alpha_0p1 / alpha_0p9 两个 profile, 只取 alpha_0p9。
    hp6_m / xerox6_m / xerox7_m 只有 alpha_0p9, 没有 alpha_0p1 —— 不影响, 本来也只用 0p9。
  * 数据文件里写的是 acend910, 按论文口径改名成 ascend910。

画点图而不是柱状图, 是因为时间跨了三个数量级 (MILP 最快 7 s, RL 预算 3600 s),
线性刻度下 MILP 会被压成一条线; 而对数刻度的柱子又不能从 0 起算, 会失真。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"

# case 顺序与 setup.tex 的 benchmark 表 (tab:benchmark) 严格一致: 按 die 数升序,
# 6/6/6/6/7/8/10/11/13/28, 同 die 数的按 ascend910, hp6_m, multigpu, xerox6_m 排。
# 两张表行序一致, 读者可以横向对照。
CASES = ["ascend910", "hp6_m", "multigpu", "xerox6_m", "xerox7_m",
         "cpu-dram", "syn1", "hp11_m", "syn4", "Case7"]

# 曾经 FMGPUTime.csv 缺 xerox6_m/hp6_m/xerox7_m 的 GPU 计时, 2026-09-18 已补齐。
# 现在要求四个来源对这 10 个 case 全部有数据, 任一缺口直接报错而不是留空 ——
# 留空的表格很容易被当成"跑得很快"读过去。
NO_GPU: set[str] = set()

# 数据文件里的旧拼写 -> 论文里的正式名
RENAME = {"acend910": "ascend910"}

# 每个 case 内五个点的**横向次序** = 由快到慢: MILP -> TW-FM (CPU+GPU) -> TW-FM (CPU)
# -> ATPlace2.5D -> RLPlanner。颜色不跟着位置走, 跟着方法走 (同一个方法在任何 case
# 里都是同一个颜色), 所以这里只是重排列表, 不动各方法自己的颜色。
SERIES = [
    ("MILP",            "#4E67C8"),
    ("TW-FM (CPU+GPU)", "#FF8021"),
    ("TW-FM (CPU)",     "#5DCEAF"),
    ("ATPlace2.5D",     "#5ECCF3"),
    ("RLPlanner",       "#81D31A"),
]

RL_TIME = 3600.0          # 论文给的 RLPlanner 预算
GPU_ALPHA = "alpha_0p9"   # 与 FMCPUTime.csv 对齐的 profile

CALIBRI_DIRS = [
    Path("/mnt/c/Windows/Fonts"),                  # WSL: 宿主机的 Windows 字体
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
                pass
        name = next((f.name for f in font_manager.fontManager.ttflist
                     if f.name in ("Calibri", "Carlito")), None)
        if name:
            matplotlib.rcParams["font.family"] = name
            return name
    matplotlib.rcParams["font.family"] = FONT_FALLBACK
    return FONT_FALLBACK


def read_csv(name: str) -> dict[str, dict]:
    with (HERE / name).open() as fh:
        return {RENAME.get(r["case"], r["case"]): r for r in csv.DictReader(fh)}


def collect() -> dict[str, list[float]]:
    at = read_csv("AT.csv")
    milp = read_csv("ILP.csv")
    cpu = read_csv("FMCPUTime.csv")
    gpu = read_csv("FMGPUTime.csv")

    missing = [(c, n) for c in CASES for n, src in
               (("AT.csv", at), ("ILP.csv", milp), ("FMCPUTime.csv", cpu),
                ("FMGPUTime.csv", gpu)) if c not in src]
    if missing:
        raise SystemExit("这些 case 在对应文件里没有数据: "
                         + ", ".join(f"{c}@{f}" for c, f in missing))

    out = {name: [] for name, _ in SERIES}
    for c in CASES:
        out["ATPlace2.5D"].append(float(at[c]["total_time_mean_s"]))
        out["RLPlanner"].append(RL_TIME)
        out["MILP"].append(float(milp[c]["wall_s"]))
        out["TW-FM (CPU)"].append(float(cpu[c]["total_mean_s"]))
        if c in NO_GPU:
            out["TW-FM (CPU+GPU)"].append(float("nan"))
        else:
            g = gpu[c]
            assert g["profile"] == GPU_ALPHA, f"{c}: GPU profile 是 {g['profile']}"
            out["TW-FM (CPU+GPU)"].append(
                float(g["sampling_mean_s"]) + float(g["stage_a_mean_s"])
                + float(g["stage_b_mean_s"]))
    return out


def emit_csv(data: dict[str, list[float]]) -> None:
    """汇总成 CSV: 各方法时间 + 相对 ATPlace2.5D 的加速比。

    列序与 tab:runtime / fig:runtime 的横向读法一致 (本方法两列 -> ATPlace2.5D ->
    MILP -> RLPlanner), 加速比跟在对应的时间列后面。
    """
    out = HERE / "runtime_summary.csv"
    fields = ["case", "TW-FM_CPU+GPU_s", "TW-FM_CPU_s", "ATPlace2.5D_s", "MILP_s",
              "RLPlanner_s", "speedup_CPU+GPU_x", "speedup_CPU_x"]
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i, c in enumerate(CASES):
            at = data["ATPlace2.5D"][i]
            gpu = data["TW-FM (CPU+GPU)"][i]
            cpu = data["TW-FM (CPU)"][i]
            fmt = lambda v: "" if np.isnan(v) else f"{v:.2f}"  # noqa: E731
            w.writerow([c, fmt(gpu), f"{cpu:.2f}", f"{at:.2f}",
                        f"{data['MILP'][i]:.2f}", f"{RL_TIME:.2f}",
                        fmt(at / gpu) if not np.isnan(gpu) else "", f"{at/cpu:.2f}"])
        # 末行给一个汇总, 免得正文里再算一遍
        at = np.array(data["ATPlace2.5D"])
        gpu = np.array(data["TW-FM (CPU+GPU)"])
        cpu = np.array(data["TW-FM (CPU)"])
        ok = ~np.isnan(gpu)
        w.writerow(["MEAN", f"{np.nanmean(gpu):.2f}", f"{cpu.mean():.2f}",
                    f"{at.mean():.2f}", f"{np.mean(data['MILP']):.2f}",
                    f"{RL_TIME:.2f}", f"{(at[ok]/gpu[ok]).mean():.2f}",
                    f"{(at/cpu).mean():.2f}"])
    print(f"写出 {out}")


def emit_table(data: dict[str, list[float]]) -> None:
    """同一份数字导成可直接 \\input 的 LaTeX 表格片段。"""
    lines = [
        r"\begin{table}[!t]",
        r"\caption{RUNTIME IN SECONDS OF THE FOUR METHODS ON EACH BENCHMARK SYSTEM, WITH "
        r"THE TW-FM SPEEDUP RELATIVE TO ATPLACE2.5D. RLPLANNER HAS NO PER-CASE "
        r"MEASUREMENT: IT IS GIVEN A FIXED 3600 s BUDGET ON EVERY SYSTEM, WHICH IS WHY "
        r"THAT COLUMN IS CONSTANT. SPEEDUP IS COMPUTED FROM THE TW-FM (CPU+GPU) COLUMN, "
        r"WHICH IS A FIVE-SEED AVERAGE; THE TW-FM (CPU) COLUMN IS A SINGLE RUN}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        # 列序: 本方法两列 -> ATPlace2.5D -> MILP -> RLPlanner。
        # 三个基线按 "有逐 case 实测的在前, 只有预算的在后" 排, RLPlanner 恒 3600 垫底。
        r"\textbf{Case} & \multicolumn{2}{c}{\textbf{TW-FM}}"
        r" & \textbf{ATPlace2.5D} & \textbf{MILP} & \textbf{RLPlanner} & \textbf{Spd.} \\",
        r" & \textbf{(CPU+GPU)} & \textbf{(CPU)} & & & & \\",
        r"\midrule",
    ]
    for i, c in enumerate(CASES):
        at = data["ATPlace2.5D"][i]
        gpu = data["TW-FM (CPU+GPU)"][i]
        g = "---" if np.isnan(gpu) else f"\\textbf{{{gpu:.0f}}}"
        spd = "---" if np.isnan(gpu) else f"{at/gpu:.1f}$\\times$"
        # case 名里的下划线要转义, 否则 \textit{xerox6_m} 会被当成数学模式的 _
        name = c.replace("_", r"\_")
        lines.append(f"\\textit{{{name}}} & {g} & "
                     f"{data['TW-FM (CPU)'][i]:.0f} & {at:.0f} & "
                     f"{data['MILP'][i]:.0f} & {data['RLPlanner'][i]:.0f} & {spd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\label{tab:runtime}",
              r"\end{table}", ""]
    out = HERE / "runtime_table.tex"
    out.write_text("\n".join(lines))
    print(f"写出 {out}")


def report(data: dict[str, list[float]]) -> None:
    print("运行时间 (s):")
    print(f"{'case':10s}" + "".join(f"{n:>18s}" for n, _ in SERIES))
    for i, c in enumerate(CASES):
        cells = "".join(f"{'---' if np.isnan(data[n][i]) else f'{data[n][i]:.2f}':>18s}"
                        for n, _ in SERIES)
        print(f"{c:10s}{cells}")


def plot(data: dict[str, list[float]], font: str) -> None:
    """横向点图: 竖轴是 case (按 die 数升序, 与 tab:benchmark 同序), 横轴是时间。

    case 放在竖轴是因为 10 个 case 名字横排要旋转 38 度才好读, 转过来之后 case 名
    直接正着写在左边; 时间轴在对数刻度上从左到右递增, 五个方法自然就按
    MILP -> TW-FM (CPU+GPU) -> TW-FM (CPU) -> ATPlace2.5D -> RLPlanner 从左读到右。
    """
    FIG.mkdir(exist_ok=True)
    y = np.arange(len(CASES))

    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#D9D9D9", lw=0.8, which="major")
    ax.grid(axis="x", color="#EFEFEF", lw=0.5, which="minor")

    # 同一 case 内把五个系列纵向错开: xerox6_m 上 MILP=178.74 和 TW-FM (CPU)=178.20
    # 在对数刻度上差不到 1 个像素, 不错开就是一个点; 每个方法拿到固定的行内偏移,
    # 所以同色点的相对高低在所有 case 里一致, 不会看串。
    dodge = np.linspace(0.30, -0.30, len(SERIES))
    for off, (name, color) in zip(dodge, SERIES):
        ax.plot(data[name], y + off, marker="o", ms=7.0, lw=0, color=color,
                markeredgecolor="white", markeredgewidth=1.0,
                label=name, zorder=3, clip_on=False)

    ax.set_xscale("log")
    ax.set_xlim(2.0, 9000)
    ax.set_ylim(len(CASES) - 0.5, -0.9)      # 反转: die 最少的 case 在最上面; 顶部留白放注记

    ax.set_yticks(y)
    ax.set_yticklabels(CASES, fontsize=12)
    ax.set_ylabel("Case", fontsize=13.5, labelpad=6)
    ax.set_xlabel("Runtime (s)", fontsize=13.5, labelpad=6)
    ax.tick_params(axis="x", labelsize=12)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BFBFBF")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
              fontsize=12, frameon=False, handletextpad=0.35, columnspacing=1.6)

    # RLPlanner 是常数, 用一条竖虚线把它标出来, 免得读者以为只是刚好都落在这
    ax.axvline(RL_TIME, color="#81D31A", ls="--", lw=0.9, alpha=0.6, zorder=1)
    ax.text(RL_TIME * 0.96, -0.62, "RLPlanner: 3600 s budget on every case",
            fontsize=10.5, color="#4F7A0B", ha="right", va="center")

    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(FIG / f"runtime_compare.{ext}", bbox_inches="tight")
    print(f"字体: {font}; 写出 {FIG/'runtime_compare.png'} 和 .eps")


def main():
    font = setup_font()
    data = collect()
    report(data)
    emit_csv(data)
    emit_table(data)
    plot(data, font)


if __name__ == "__main__":
    main()
