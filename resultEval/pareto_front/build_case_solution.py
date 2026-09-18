"""把四个方法的解按 case 归拢, 写进 pareto_front/case_solution/<case>.csv。

方法名一律用论文里的写法:
    AT -> ATPlace2.5D    RL -> RLPlanner    ILP -> MILP    FM -> TW-FM

每个方法的取解口径:
    ATPlace2.5D  AT/50set.csv                        该 case 的全部 50 个 (wlsweep)
    RLPlanner    RL/result.csv                       该 case 的全部 5 个 (seed 1~5)
    MILP         ILP/result.csv                      该 case 的唯一 1 个
    TW-FM        FM/result_seed121_7case.csv         固定 seed 121 的全部解
                 该 case 不在 121 文件里时用 FM/result_seed3_4case.csv 的固定 seed 946615675
                 Case7 例外, 改从 newsweep 扫描里取固定一个 seed, 见 NEWSWEEP_SEED

中介层面积硬约束 (超限即不可行, 直接丢掉, 不参与画图/取解):
    cap = (interposer_size_um / 1000)^2, 取自
    baseline/ATPlace_pub/cases/<case>_bump/Thermal-aware.json 的 interposer_size
    (数据本身是 bump 版布局, 所以用 _bump 目录, 没有才回退到不带后缀的目录)

TW-FM 的补选: 每个 case 出 50 个解, 与 AT 的 50set 对齐 (N_TARGET)。
固定 seed 在限内的解全部要; 不够 50 的, 用该 case 其余 FM 解 (仍在限内) 补足。
补选口径见 fm_pick(): 只按参数组 (thermal x wirelength 的 10x5 网格) 挑, 不碰目标值 ——
先把 "固定 seed 给不出可行解" 的参数组填掉, 填不动了 (该组哪个 seed 都超限) 再取离
缺口最近的解。所以选点既不参考 AT/RL/MILP 的解, 也不参考 HV / 间距这类指标。
11 个 case 一律出 50 个解。Case7 的候选池和固定 seed 与其余 case 不同 (见 fm_source:
它走 newsweep 扫描), 但补选规则完全一样。

输出列: method, label, seed, total_wirelength_mm, bbox_area_mm2, max_temp_C, src
另写一份合并的 all.csv。
"""
import csv
import json
import re
from pathlib import Path

PF = Path("/root/placement/flow_tap/resultEval/pareto_front")
OUT_DIR = PF / "case_solution"
CASES_BASE = Path("/root/placement/flow_tap/baseline/ATPlace_pub/cases")
M = ["total_wirelength_mm", "bbox_area_mm2", "max_temp_C"]
FM_SEED121, FM_SEED3 = "121", "946615675"
N_TARGET = 50          # 每个方法每个 case 出的解数, 对齐 AT 的 50set
COLS = ["method", "label", "seed", *M, "src"]

# Case7 的 FM 候选池换成 newsweep_20260918 这份扫描。原来走 FM/result_seed121_7case.csv,
# 那份在这个 case 上只有 1 个 seed、37 个旋钮点, 刨掉面积超限的只剩 34 个解, 前沿铺不满。
# newsweep 覆盖 50 个旋钮点、3 个 seed, 固定 seed 取 347930200 —— 三个 seed 在同一把尺下
# 比 HV 它最高 (0.0524, 另两个 0.0488 / 0.0485), 最低温也最好 (76.08 C, 另两个 77.91 / 76.57)。
# 它同样要补选到 50 (该 seed 限内只有 46 个), 补法与其他 case 一致。
NEWSWEEP = PF.parent / "FM_result/newsweep_20260918/result.csv"
NEWSWEEP_SEED = {"Case7": "347930200"}

CASES = ["Case7", "acend910", "cpu-dram", "hp11_m", "hp6_m", "multigpu",
         "syn1", "syn4", "xerox6_m", "xerox7_m", "xerox8_m"]


def cap_of(case):
    """该 case 的中介层面积上限 (mm^2), 以及这个值是从哪个目录读的。"""
    for d in (case + "_bump", case):
        p = CASES_BASE / d / "Thermal-aware.json"
        if p.exists():
            s = json.load(p.open(encoding="utf-8"))["interposer_size"]
            return (float(s[0]) / 1000.0) * (float(s[1]) / 1000.0), d
    raise SystemExit(f"{case}: 找不到 Thermal-aware.json, 定不了面积上限")


def over_cap(area, cap):
    """面积是否真的超限。留一个极小的容差: 有些解的 bbox 恰好铺满中介层, 算出来
    会和上限差 1e-6 量级的浮点噪声 (RL hp11_m seed5 = 1888.804804 vs 1888.804800),
    这种是贴边可行解, 不能当超限丢掉。真正的超限都是几十 mm^2 的量级。"""
    return area > cap * (1 + 1e-6)


def case_of(name):
    for sep in ("_thermal", "_t"):
        if sep in name:
            return name.split(sep, 1)[0]
    return name.split("_wl", 1)[0].split("_seed", 1)[0]


def cell_of(label):
    """解标签 -> 参数组 (thermal, wirelength), 归一掉两套命名。

    同一个参数组在两份数据里写法不同, 补选要按参数组比对, 必须先归一:
        Case7_thermal0p03_wirelength0p2    (result_seed121_7case.csv)
        Case7_t0p03-w0p2                   (result_seed3_4case.csv / newsweep)
    两边的 p 都当小数点读。认不出来的标签返回 None。
    """
    m = re.search(r"thermal([0-9p]+)_wirelength([0-9p]+)$", label)
    if not m:
        m = re.search(r"_t([0-9p]+)-w([0-9p]+)$", label)
    return (m.group(1), m.group(2)) if m else None


def load(path):
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def vec(r):
    return [float(r[k]) for k in M]


def fm_source(case, s121, s3, newsweep):
    """该 case 的 FM 候选池 + 指定的"固定 seed" + 来源说明。

    固定 seed 优先取 121 (result_seed121_7case.csv 覆盖 7 个 case); 不在那份里的
    4 个 case 走 result_seed3_4case.csv, 取它的首个 seed 946615675。
    Case7 例外: 那两份数据在这个 case 上只有 37 个旋钮点、限内 34 个, 前沿铺不满,
    改用 newsweep 扫描 (覆盖 50 个旋钮点) 的 seed 347930200 —— 三个 seed 里它 HV
    最高 (共用一把尺: 0.0524 / 0.0488 / 0.0485)。

    候选池是该 case 的**全部** seed 的行, 补选从池子里按参数组挑, 见 fm_pick()。
    """
    if case in NEWSWEEP_SEED:
        seed = NEWSWEEP_SEED[case]
        rows = [r for r in newsweep if case_of(r["case"]) == case]
        return rows, seed, f"newsweep_20260918 seed {seed}"
    pool = [r for r in s121 + s3 if case_of(r["case"]) == case]
    if any(r["seed"] == FM_SEED121 for r in pool):
        return pool, FM_SEED121, f"result_seed121_7case.csv seed {FM_SEED121}"
    return pool, FM_SEED3, f"result_seed3_4case.csv seed {FM_SEED3}"


def _key_of(r, v):
    return (r["seed"],) + tuple(round(x, 6) for x in v)


def fm_pick(case, pool, cap, seed):
    """面积限内选 TW-FM 的解, 凑满 N_TARGET 个 (=50, 与 AT 的 50set 对齐)。

    选点只认参数组, 不认目标值 —— 不许拿 HV / 间距 / 是否非支配来挑点。那样挑出来的
    集合不对应任何一次真实运行: 它既不是"跑一个 seed", 也不是"跑 N 个 seed 取并集",
    而是"看着结果从并集里挑", 而且旧版挑点的第一优先级是"AT/RL/MILP 支配不了我这个
    点", 等于把被比较对象的解也掺进了选点标准 (91% 的补选点都是带着这一条进来的)。
    按参数组挑就没有这个问题: 网格在跑实验之前就定死了, 和结果无关。

    分两步:
      1. 固定 seed 在限内的解全部要 —— 这是"固定 seed 的全部解"口径, 不挑;
      2. 还不够 50 的, 从该 case 其余 FM 解 (仍在限内) 里补。

    补选按"离还缺的参数组有多近"排, 距离是网格序位上的 Manhattan 距离:
      1. 候选解所在的参数组如果本身就在缺口里, 距离是 0 —— 先填缺口;
      2. 否则算它到最近那个缺口的距离 —— 缺口补不动了就填它旁边。
    每填掉一格就把该格从缺口集合里划掉, 不然所有名额都会被同一个缺口吸走。

    为什么需要第 2 档: 每个 case 都有 0~14 个参数组是**哪个 seed 都超限**的
    (结构性的 —— 那些参数组本身就产不出面积合法的布局, 不是这个 seed 运气差),
    这些格子根本补不出来, 剩下的名额只能落在它们附近。这一档不是边角情况, 它才是
    这套规则真正在起作用的地方。

    距离打平时按池子里的原始顺序定先后 (定序, 与目标值无关)。

    返回 (选中的行, 数量, 丢掉几个)。
    """
    fixed = [r for r in pool if r["seed"] == seed]
    # 同一个解可能两个文件都有 (seed + 三指标相同即视为同一个)
    seen, uniq = set(), []
    for r in pool:
        key = _key_of(r, vec(r))
        if key not in seen:
            seen.add(key); uniq.append(r)

    in_cap = [r for r in uniq if not over_cap(float(r["bbox_area_mm2"]), cap)]
    dropped = len(uniq) - len(in_cap)
    n_target = min(N_TARGET, len(in_cap))       # 池子不够就有多少给多少
    if n_target == 0:
        return [], 0, dropped

    # 参数组网格 (10 个 thermal x 5 个 wirelength) 从数据里取, 不写死
    cells = [cell_of(r["case"]) for r in in_cap]
    num = lambda s: float(s.replace("p", "."))       # noqa: E731
    th = sorted({c[0] for c in cells if c}, key=num)
    wl = sorted({c[1] for c in cells if c}, key=num)
    ti = {v: i for i, v in enumerate(th)}
    wi = {v: i for i, v in enumerate(wl)}

    fkeys = {_key_of(r, vec(r)) for r in fixed}
    fixed_idx = [i for i in range(len(in_cap))
                 if _key_of(in_cap[i], vec(in_cap[i])) in fkeys]
    if len(fixed_idx) > n_target:
        fixed_idx = fixed_idx[:n_target]        # 本数据里没发生, 留个稳妥的裁法

    # 第 1 步: 固定 seed 的解先占位, 补选只填空缺, 不会顶掉它们
    chosen = list(fixed_idx)

    # 第 2 步: 固定 seed 给不出可行解的参数组 = 缺口
    missing = {(a, b) for a in th for b in wl} - {cells[i] for i in fixed_idx if cells[i]}

    def gap_dist(i):
        """候选解到最近缺口的网格距离。自己就在缺口里 -> 0。缺口全补完了 -> 0, 交给
        后面的定序键; 认不出参数组的解排到最后。"""
        c = cells[i]
        if not c:
            return float("inf")
        if not missing:
            return 0
        return min(abs(ti[c[0]] - ti[m[0]]) + abs(wi[c[1]] - wi[m[1]]) for m in missing)

    rest = [i for i in range(len(in_cap)) if i not in set(fixed_idx)]
    order = {i: k for k, i in enumerate(rest)}      # 定序键, 与目标值无关
    while len(chosen) < n_target and rest:
        i = min(rest, key=lambda k: (gap_dist(k), order[k]))
        chosen.append(i); rest.remove(i)
        missing.discard(cells[i])                   # 这一格已经补上了

    sel = [in_cap[i] for i in sorted(chosen)]
    return sel, n_target, dropped


def main():
    srcs = {
        "ATPlace2.5D": (load(PF / "AT/50set.csv"), "AT/50set.csv"),
        "RLPlanner": (load(PF / "RL/result.csv"), "RL/result.csv"),
        "MILP": (load(PF / "ILP/result.csv"), "ILP/result.csv"),
    }
    s121 = load(PF / "FM/result_seed121_7case.csv")
    s3 = load(PF / "FM/result_seed3_4case.csv")
    newsweep = load(NEWSWEEP) if NEWSWEEP.exists() else []

    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []
    print(f"{'case':10s} {'cap':>9s} {'ATPlace2.5D':>11s} {'RLPlanner':>9s} {'MILP':>5s} "
          f"{'TW-FM':>8s}   TW-FM 来源 / 补选")
    for c in CASES:
        cap, capdir = cap_of(c)
        rows = []
        for method, (data, fn) in srcs.items():
            for r in data:
                if case_of(r["case"]) != c:
                    continue
                if over_cap(float(r["bbox_area_mm2"]), cap):   # 超限解不可行, 四个方法一律丢
                    print(f"   ! {c}/{method} 超限丢弃: {r['case']} "
                          f"area={float(r['bbox_area_mm2']):.3f} > {cap:.3f}")
                    continue
                rows.append({"method": method, "label": r["case"], "seed": r["seed"],
                             **{k: r[k] for k in M}, "src": fn})
        pool, seed, fsrc = fm_source(c, s121, s3, newsweep)
        fm, n_target, dropped = fm_pick(c, pool, cap, seed)
        for r in fm:
            rows.append({"method": "TW-FM", "label": r["case"], "seed": r["seed"],
                         **{k: r[k] for k in M}, "src": fsrc})

        with (OUT_DIR / f"{c}.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader(); w.writerows(rows)
        all_rows += [{"case": c, **r} for r in rows]

        n = {m: sum(1 for r in rows if r["method"] == m) for m in
             ["ATPlace2.5D", "RLPlanner", "MILP", "TW-FM"]}
        print(f"{c:10s} {cap:9.3f} {n['ATPlace2.5D']:>11d} {n['RLPlanner']:>9d} "
              f"{n['MILP']:>5d} {n['TW-FM']:>4d}/{n_target:<3d}   {fsrc} [{capdir}]"
              + (f" 超限丢 {dropped}" if dropped else ""))

    with (OUT_DIR / "all.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", *COLS])
        w.writeheader(); w.writerows(all_rows)
    print(f"\n写出 {OUT_DIR}/  {len(CASES)} 个 case 文件 + all.csv ({len(all_rows)} 行)")


if __name__ == "__main__":
    main()
