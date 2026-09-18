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
  * FMGPUTime.csv 只覆盖原始 12 个 benchmark 中的 8 个, 没有 xerox6_m / hp6_m /
    xerox7_m 的 GPU 计时 (整个目录下都找不到), 这三行的 CPU+GPU 留空。
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

# case 顺序与排列严格照搬 pareto.tex 的表 V / 图 5, 方便两张图表横向对照。
CASES = ["ascend910", "xerox6_m", "hp6_m", "multigpu",
         "xerox7_m", "cpu-dram", "hp11_m", "syn4"]

# FMGPUTime.csv 没覆盖的 case: 这三行的 CPU+GPU 一栏只能留空
NO_GPU = {"xerox6_m", "hp6_m", "xerox7_m"}

# 数据文件里的旧拼写 -> 论文里的正式名
RENAME = {"acend910": "ascend910"}

SERIES = [
    ("ATPlace2.5D",     "#5ECCF3"),
    ("RLPlanner",       "#81D31A"),
    ("MILP",            "#4E67C8"),
    ("TW-FM (CPU+GPU)", "#FF8021"),
    ("TW-FM (CPU)",     "#5DCEAF"),
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
    """汇总成 CSV: 各方法时间 + 相对 ATPlace2.5D 的加速比。"""
    out = HERE / "runtime_summary.csv"
    fields = ["case", "ATPlace2.5D_s", "RLPlanner_s", "MILP_s",
              "TW-FM_CPU_s", "TW-FM_CPU+GPU_s",
              "speedup_CPU_x", "speedup_CPU+GPU_x"]
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for i, c in enumerate(CASES):
            at = data["ATPlace2.5D"][i]
            gpu = data["TW-FM (CPU+GPU)"][i]
            cpu = data["TW-FM (CPU)"][i]
            fmt = lambda v: "" if np.isnan(v) else f"{v:.2f}"  # noqa: E731
            w.writerow([c, f"{at:.2f}", f"{RL_TIME:.2f}", f"{data['MILP'][i]:.2f}",
                        f"{cpu:.2f}", fmt(gpu), f"{at/cpu:.2f}",
                        fmt(at / gpu) if not np.isnan(gpu) else ""])
        # 末行给一个汇总, 免得正文里再算一遍
        at = np.array(data["ATPlace2.5D"])
        gpu = np.array(data["TW-FM (CPU+GPU)"])
        cpu = np.array(data["TW-FM (CPU)"])
        ok = ~np.isnan(gpu)
        w.writerow(["MEAN", f"{at.mean():.2f}", f"{RL_TIME:.2f}",
                    f"{np.mean(data['MILP']):.2f}", f"{cpu.mean():.2f}",
                    f"{np.nanmean(gpu):.2f}", f"{(at/cpu).mean():.2f}",
                    f"{(at[ok]/gpu[ok]).mean():.2f}"])
    print(f"写出 {out}")


def emit_table(data: dict[str, list[float]]) -> None:
    """同一份数字导成可直接 \\input 的 LaTeX 表格片段。"""
    lines = [
        r"\begin{table}[!t]",
        r"\caption{RUNTIME OF THE FOUR METHODS, WITH THE TW-FM SPEEDUP RELATIVE TO "
        r"ATPLACE2.5D. RLPLANNER IS ASSIGNED A FIXED 3600 s BUDGET AND HAS NO PER-CASE "
        r"MEASUREMENT. SPEEDUP IS COMPUTED FROM THE TW-FM (CPU+GPU) COLUMN, WHICH IS A "
        r"FIVE-SEED AVERAGE; THE TW-FM (CPU) COLUMN IS A SINGLE RUN. THE THREE SYSTEMS "
        r"MARKED --- HAVE NO GPU MEASUREMENT}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Case} & \textbf{ATPlace2.5D} & \textbf{RLPlanner} & \textbf{MILP}"
        r" & \multicolumn{2}{c}{\textbf{TW-FM}} & \textbf{Spd.} \\",
        r" & & & & \textbf{(CPU)} & \textbf{(CPU+GPU)} & \\",
        r"\midrule",
    ]
    for i, c in enumerate(CASES):
        at = data["ATPlace2.5D"][i]
        gpu = data["TW-FM (CPU+GPU)"][i]
        g = "---" if np.isnan(gpu) else f"\\textbf{{{gpu:.0f}}}"
        spd = "---" if np.isnan(gpu) else f"{at/gpu:.1f}$\\times$"
        lines.append(f"\\textit{{{c}}} & {at:.0f} & {data['RLPlanner'][i]:.0f} & "
                     f"{data['MILP'][i]:.0f} & {data['TW-FM (CPU)'][i]:.0f} & {g} & {spd} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\label{tab:runtime}", r"\end{table}", ""]
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
    FIG.mkdir(exist_ok=True)
    x = np.arange(len(CASES))

    fig, ax = plt.subplots(figsize=(9.2, 3.9), dpi=200)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", lw=0.8, which="major")
    ax.grid(axis="y", color="#EFEFEF", lw=0.5, which="minor")

    # 同一 case 内把五个系列横向错开, 否则 RLPlanner (恒 3600) 会和落在 3.3e3
    # 附近的 MILP 点完全重叠, 分不出是两个方法。
    dodge = np.linspace(-0.26, 0.26, len(SERIES))
    for off, (name, color) in zip(dodge, SERIES):
        ax.plot(x + off, data[name], marker="o", ms=7.0, lw=0, color=color,
                markeredgecolor="white", markeredgewidth=1.0,
                label=name, zorder=3, clip_on=False)

    ax.set_yscale("log")
    ax.set_xlim(-0.6, len(CASES) - 0.4)
    ax.set_ylim(2.0, 9000)

    ax.set_xticks(x)
    ax.set_xticklabels(CASES, rotation=38, ha="right", fontsize=12)
    ax.set_xlabel("Case", fontsize=13.5, labelpad=6)
    ax.set_ylabel("Runtime (s)", fontsize=13.5, labelpad=6)
    ax.tick_params(axis="y", labelsize=12)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BFBFBF")

    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=5,
              fontsize=12, frameon=False, handletextpad=0.35, columnspacing=1.6)

    # RLPlanner 是常数, 用一条虚线把它标出来, 免得读者以为只是刚好都落在这
    ax.axhline(RL_TIME, color="#81D31A", ls="--", lw=0.9, alpha=0.6, zorder=1)
    ax.text(0.15, RL_TIME * 1.55, "RLPlanner: 3600 s budget on every case",
            fontsize=10.5, color="#4F7A0B", ha="left", va="bottom")

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
