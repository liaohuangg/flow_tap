"""评价帕累托解集: 超体积 HV + 几个常用指标, 逐 case 算。

三个目标全部最小化。原始量纲差得太远 (线长 ~1e5 mm, 面积 ~1e3 mm^2, 温度 ~1e2 C),
直接算 HV 的话线长一项就把体积全占了、温度和面积等于没参与, 所以先归一化:
  每个 case 取该 case 所有方法的解 (AT/RL/MILP/FM 全部) 在每个目标上的 min/max,
  把该目标线性映射到 [0,1], 参考点取 (1.1, 1.1, 1.1)。
  —— 归一化框由该 case 的全部解定义, 不随被评价的子集变, 保证 AT 和 FM 用同一把尺。

HV 用精确算法 (Fonseca 2006 的三维做法), 不是采样估计:
  把参考点平移到原点, 每个解变成以原点为对角的一个盒子, HV = 这些盒子并集的体积。
  沿第一个目标切薄片: HV = a1*A(全部) + sum_k (a_{k+1}-a_k) * A(第 k+1..n 个点),
  其中 A(S) 是 S 在另两个目标上的二维并集面积 (扫描线求), n<=100 时几毫秒。

其他指标:
  |front|  该方法自己的解里非支配的个数 (ONVG, 能给出的选择有多少)
  C(A,B)   集合覆盖: B 里被 A 中某个解支配的比例 —— 两个方法正面对比, 最直观
  IGD+     到参考前沿的平均距离 (参考前沿 = 该 case 四个方法的解合起来的全局前沿)
  Spacing  自己前沿上相邻点间距的变异系数, 越小说明点分布越均匀
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PF = Path("/root/placement/flow_tap/resultEval/pareto_front")
M = ["total_wirelength_mm", "bbox_area_mm2", "max_temp_C"]
REF = np.array([1.1, 1.1, 1.1])


# ---------- 基础 ----------
def dominates(a, b) -> bool:
    """a 支配 b: 三项都不差, 且至少一项更好 (全部最小化)。"""
    return bool(np.all(a <= b) and np.any(a < b))


def front_mask(V: np.ndarray) -> np.ndarray:
    n = len(V)
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and dominates(V[j], V[i]):
                keep[i] = False
                break
    return keep


# ---------- 二维/三维并集体积 ----------
def area2d(P: np.ndarray) -> float:
    """点集 P (m,2) 的 "矩形 [0,b]x[0,c] 并集" 面积, 扫线求, 假定值都 >= 0。"""
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0], kind="mergesort")]
    b, c = P[:, 0], P[:, 1]
    suff = np.maximum.accumulate(c[::-1])[::-1]      # 后缀最大值
    return float(np.sum(np.diff(np.concatenate([[0.0], b])) * suff))


def hypervolume(V: np.ndarray, ref=REF) -> float:
    """点集 V (n,3) 相对于参考点 ref 的超体积, 全部最小化。"""
    U = ref - V                                     # 平移到 "盒子从原点长出去"
    U = U[np.all(U > 0, axis=1)]                    # 超出参考点的解不贡献体积
    if len(U) == 0:
        return 0.0
    # U 空间里盒子越大越好, 要留的是"极大"点集; front_mask 判的是极小,
    # 所以取反再判 (直接对 U 用 front_mask 会把该留的全滤掉)。
    U = U[front_mask(-U)]
    if len(U) == 0:
        return 0.0
    U = U[np.argsort(U[:, 0], kind="mergesort")]
    a = U[:, 0]
    hv = a[0] * area2d(U[:, 1:])                    # [0, a1) 这一段, 所有点都在
    for k in range(len(U) - 1):
        hv += (a[k + 1] - a[k]) * area2d(U[k + 1:, 1:])
    return float(hv)


def hypervolume_mc(V: np.ndarray, ref=REF, n: int = 400_000, seed: int = 0) -> float:
    """蒙特卡洛估 HV, 只用来校验精确实现 (真值应该略小, 是欠估计)。"""
    rng = np.random.default_rng(seed)
    U = ref - V
    U = U[np.all(U > 0, axis=1)]
    if len(U) == 0:
        return 0.0
    box = float(np.prod(ref))
    s = rng.random((n, 3)) * ref
    inside = np.any(np.all(s[:, None, :] <= U[None, :, :], axis=2), axis=1)
    return box * inside.mean()


# ---------- 集合指标 ----------
def c_metric(A: np.ndarray, B: np.ndarray) -> float:
    """C(A,B): B 中被 A 至少一个解支配的比例。"""
    if len(B) == 0:
        return float("nan")
    return sum(any(dominates(a, b) for a in A) for b in B) / len(B)


def igd_plus(A: np.ndarray, R: np.ndarray) -> float:
    """IGD+: 参考前沿 R 上每个点到 A 的最近距离的平均。R 为空返回 nan。

    距离用 IGD+ 的定义, 只惩罚 "A 比 r 差" 的那些分量 (A 比 r 好不罚),
    所以它不会因为 A 的点超出了参考前沿而虚高。
    """
    if len(A) == 0 or len(R) == 0:
        return float("nan")
    d = np.maximum(A[None, :, :] - R[:, None, :], 0.0)     # (|R|,|A|,3)
    return float(np.sqrt((d ** 2).sum(-1)).min(axis=1).mean())


def spacing(V: np.ndarray) -> float:
    """Schott 的 Spacing: 前沿上每个点到自身最近邻的距离的标准差, 0 = 完全均匀。

    公式 S = sqrt( 1/(n-1) * sum_i (d_i - d_bar)^2 ), d_i = min_{j != i} |f_i - f_j|。
    分母是 n-1 (样本标准差), 不是 n —— 论文引用 Schott 1995 时口径要一致。

    注意这个指标只度量 "点距等不等", 对 "有没有铺开" 完全不敏感: 把整个前沿朝质心
    收缩, Spacing 单调变小, 而覆盖范围同时塌掉。不能单独当主指标。
    单点 (n<2) 无定义, 返回 nan。
    """
    if len(V) < 2:
        return float("nan")
    d = np.sqrt(((V[:, None, :] - V[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).std(ddof=1))


# ---------- 主流程 ----------
def load_case(case: str) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader((PF / "case_solution" / f"{case}.csv").open()))
    out: dict[str, list[list[float]]] = {}
    for r in rows:
        out.setdefault(r["method"], []).append([float(r[k]) for k in M])
    return {k: np.array(v) for k, v in out.items()}


def normalize(raw: dict[str, np.ndarray], mode: str = "max"):
    """把该 case 全部解在每个目标上映射到归一化空间 (同一把尺给所有方法)。

    mode="max" (论文口径): 每个目标除以它在所有方法全部解上的最大值, 参考点取
        (1.1, 1.1, 1.1)。因为逐轴线性缩放会把体积乘上 prod(hi), 这与在原始单位下
        把参考点设在 1.1*[W_max, A_max, T_max] 只差一个常数因子, 排序完全一致。
    mode="minmax": 线性映射到 [0,1] 再配同样的参考点。数值不同, 不要混用。

    两种口径都必须按 "该 case 四个方法全部解" 定尺, 不能只按被评价的子集定 ——
    否则一个方法丢掉几个差解就能缩小自己的框, HV 凭空变大。
    返回 (归一化后的字典, lo, hi)。
    """
    allv = np.vstack(list(raw.values()))
    lo, hi = allv.min(0), allv.max(0)
    if mode == "max":
        scale = np.where(hi != 0, hi, 1.0)
        return {k: v / scale for k, v in raw.items()}, lo, hi
    if mode == "minmax":
        span = np.where(hi > lo, hi - lo, 1.0)
        return {k: (v - lo) / span for k, v in raw.items()}, lo, hi
    raise SystemExit(f"normalize: 不认识的 mode {mode!r} (只有 'max' 和 'minmax')")


def main():
    cases = ["Case7", "acend910", "cpu-dram", "hp11_m", "hp6_m", "multigpu",
             "syn1", "syn4", "xerox6_m", "xerox7_m", "xerox8_m"]
    names = ["ATPlace2.5D", "RLPlanner", "MILP", "TW-FM"]

    print("校验精确 HV 实现 (蒙特卡洛 vs 精确, 两者应几乎相等):")
    rng = np.random.default_rng(1)
    for trial in range(3):
        V = rng.random((40, 3)) * 0.9
        ex, mc = hypervolume(V), hypervolume_mc(V)
        print(f"   试 {trial+1}: 精确 {ex:.5f}   蒙特卡洛(40万点) {mc:.5f}   "
              f"相对差 {abs(mc-ex)/ex*100:.2f}%")

    rows_all = []
    for c in cases:
        raw = load_case(c)
        have = [n for n in names if n in raw]
        norm, lo, hi = normalize(raw)
        allpt = np.vstack(list(norm.values()))
        R = allpt[front_mask(allpt)]                   # 该 case 全局前沿当 IGD+ 参考
        hv_of = {n: hypervolume(norm[n]) for n in have}
        best = max(hv_of.values())
        print(f"\n{c}   (归一化框: 线长 {lo[0]:.0f}~{hi[0]:.0f} mm, "
              f"面积 {lo[1]:.0f}~{hi[1]:.0f} mm^2, 温度 {lo[2]:.1f}~{hi[2]:.1f} C)")
        print(f"   {'方法':11s} {'N':>3s} {'|front|':>7s} {'HV':>8s} {'HV占比':>7s} "
              f"{'IGD+':>7s} {'Spacing':>8s}")
        for n in have:
            V = norm[n]
            fr = V[front_mask(V)]
            hv = hv_of[n]
            print(f"   {n:11s} {len(V):>3d} {len(fr):>7d} {hv:>8.4f} {hv/best*100:>6.1f}% "
                  f"{igd_plus(V, R):>7.4f} {spacing(fr):>8.4f}")
            rows_all.append(dict(case=c, method=n, n=len(V), front=len(fr), hv=hv,
                                 hv_ratio=hv/best, igd=igd_plus(V, R),
                                 spacing=spacing(fr)))
        # 两两 C-metric: C(行, 列) = 列被行支配的比例
        print(f"   C-metric  行支配列的比例:")
        print(f"   {'':11s} " + " ".join(f"{n:>11s}" for n in have))
        for a in have:
            cells = []
            for b in have:
                cells.append("      --   " if a == b
                             else f"{c_metric(norm[a], norm[b]):>10.2f} ")
            print(f"   {a:11s} " + " ".join(cells))

    out = PF / "metrics" / "set_metrics.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "method", "n", "front", "hv",
                                           "hv_ratio", "igd", "spacing"])
        w.writeheader()
        w.writerows([{k: (f"{v:.6g}" if isinstance(v, float) else v)
                      for k, v in r.items()} for r in rows_all])

    print("\n=== HV 占该 case 最优的百分比 (100% = 该 case 最强) ===")
    print(f"{'case':10s} " + " ".join(f"{n:>11s}" for n in names))
    for c in cases:
        d = {r["method"]: r["hv_ratio"]*100 for r in rows_all if r["case"] == c}
        print(f"{c:10s} " + " ".join(f"{d.get(n, float('nan')):>10.1f}%" for n in names))
    n_win = {n: 0 for n in names}
    for c in cases:
        d = {r["method"]: r["hv"] for r in rows_all if r["case"] == c}
        n_win[max(d, key=d.get)] += 1
    print("\nHV 拿第一的次数: " + ", ".join(f"{k} {v}" for k, v in n_win.items()))
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
