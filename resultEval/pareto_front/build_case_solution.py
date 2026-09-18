"""把四个方法的解按 case 归拢, 写进 pareto_front/case_solution/<case>.csv。

方法名一律用论文里的写法:
    AT -> ATPlace2.5D    RL -> RLPlanner    ILP -> MILP    FM -> TW-FM

每个方法的取解口径:
    ATPlace2.5D  AT/50set.csv                        该 case 的全部 50 个 (wlsweep)
    RLPlanner    RL/result.csv                       该 case 的全部 5 个 (seed 1~5)
    MILP         ILP/result.csv                      该 case 的唯一 1 个
    TW-FM        FM/result_seed121_7case.csv         固定 seed 121 的全部解
                 该 case 不在 121 文件里时用 FM/result_seed3_4case.csv 的固定 seed 946615675

中介层面积硬约束 (超限即不可行, 直接丢掉, 不参与画图/取解):
    cap = (interposer_size_um / 1000)^2, 取自
    baseline/ATPlace_pub/cases/<case>_bump/Thermal-aware.json 的 interposer_size
    (数据本身是 bump 版布局, 所以用 _bump 目录, 没有才回退到不带后缀的目录)

TW-FM 的补选: 每个 case 出 50 个解, 与 AT 的 50set 对齐 (N_TARGET)。
固定 seed 在限内的解全部要; 不够 50 的, 用该 case 其余 FM 解 (仍在限内) 补足。
补选口径见 fm_pick(): 固定 seed 的解先占位, 补选时 "第 1 层优先" 保证贴着前沿,
"层内贪心最远点" 保证补进去的点在前沿上铺得开, 不往固定 seed 旁边挤。
Case7 的 FM 解总共只有 34 个在限内, 凑不满 50, 该 case 就 34 个。

输出列: method, label, seed, total_wirelength_mm, bbox_area_mm2, max_temp_C, src
另写一份合并的 all.csv。
"""
import csv
import json
import math
from pathlib import Path

PF = Path("/root/placement/flow_tap/resultEval/pareto_front")
OUT_DIR = PF / "case_solution"
CASES_BASE = Path("/root/placement/flow_tap/baseline/ATPlace_pub/cases")
M = ["total_wirelength_mm", "bbox_area_mm2", "max_temp_C"]
FM_SEED121, FM_SEED3 = "121", "946615675"
N_TARGET = 50          # 每个方法每个 case 出的解数, 对齐 AT 的 50set
COLS = ["method", "label", "seed", *M, "src"]

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


def load(path):
    with path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if not (r.get("error") or "").strip()]


def vec(r):
    return [float(r[k]) for k in M]


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def layers(V):
    """NSGA-II fast non-dominated sort -> [[下标...], ...], 第 1 层是非支配层。"""
    n = len(V)
    dom, cnt = [[] for _ in range(n)], [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(V[i], V[j]):
                dom[i].append(j); cnt[j] += 1
            elif dominates(V[j], V[i]):
                dom[j].append(i); cnt[i] += 1
    cur = [i for i in range(n) if cnt[i] == 0]
    out = []
    while cur:
        out.append(cur)
        nxt = []
        for i in cur:
            for j in dom[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        cur = nxt
    return out


def crowding(V, idx):
    """NSGA-II 拥挤度, 只算层内 idx, 边界点给 inf (同层里优先留边界, 撑开前沿)。"""
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: float("inf") for i in idx}
    for m in range(3):
        s = sorted(idx, key=lambda i: V[i][m])
        d[s[0]] = d[s[-1]] = float("inf")
        lo, hi = V[s[0]][m], V[s[-1]][m]
        if hi > lo:
            for k in range(1, len(s) - 1):
                d[s[k]] += (V[s[k + 1]][m] - V[s[k - 1]][m]) / (hi - lo)
    return d


def fm_fixed(case, s121, s3):
    """该 case 的"固定 seed"解集: 121 优先, 没有就用 seed3 文件的 946615675。"""
    rs = [r for r in s121 if case_of(r["case"]) == case and r["seed"] == FM_SEED121]
    if rs:
        return rs, "result_seed121_7case.csv", f"seed {FM_SEED121}"
    rs = [r for r in s3 if case_of(r["case"]) == case and r["seed"] == FM_SEED3]
    if rs:
        return rs, "result_seed3_4case.csv", f"seed {FM_SEED3}"
    return [], "", "无数据"


def _key_of(r, v):
    return (r["seed"],) + tuple(round(x, 6) for x in v)


def fm_pick(case, s121, s3, cap, others):
    """面积限内选 TW-FM 的解, 凑满 N_TARGET 个 (=50, 与 AT 的 50set 对齐)。

    others 是同一 case 里 ATPlace2.5D / RLPlanner / MILP 的解 (目标向量列表),
    只用来判断某个 FM 点有没有被别的方法支配, 不参与选点本身。

    分两步:
      1. 固定 seed 在限内的解全部要 —— 这是"固定 seed 的全部解"口径, 不挑;
      2. 还不够 50 的, 从该 case 其余 FM 解 (仍在限内) 里补。

    补选同时管三件事, 层内按这个优先级排:
      1. 别的方法支配不了的先要 —— 只有 FM 自己第 1 层的点有可能站在全局前沿上
         (第 2 层往后的点一定被自己第 1 层的某个点支配, 所以必然掉出全局前沿),
         而第 1 层里也只有一部分没被 AT 支配。补选名额有限, 先紧着这些填。
      2. 铺得开 —— 取 "离已选集合最远" 的那个 (贪心最远点采样)。
         必须显式做这件事: NSGA-II 的拥挤度只在层内算, 不知道固定 seed 已经占了哪些
         位置, 照着它补出来的点会挤在固定 seed 旁边, 前沿上留下空档。
      3. 距离打平时按拥挤度定先后 (边界点优先, 撑开前沿)。
    距离在归一化后的目标空间里算 —— 三个目标量纲差几个数量级, 不归一化的话
    "远近" 全由线长一项决定, 面积和温度等于没参与。

    Case7 的 FM 解总共只有 34 个在限内, 补不满 50, 有多少给多少。

    返回 (选中的行, 数量, 来源说明, 丢掉几个)。
    """
    fixed, fsrc, fnote = fm_fixed(case, s121, s3)
    pool = [r for r in s121 + s3 if case_of(r["case"]) == case]
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
        return [], 0, fsrc or "无数据", dropped

    V = [vec(r) for r in in_cap]
    # 归一化到 [0,1], 用来量 "点与点之间隔着多远"
    lo = [min(v[k] for v in V) for k in range(3)]
    span = [(max(v[k] for v in V) - lo[k]) or 1.0 for k in range(3)]
    N = [[(v[k] - lo[k]) / span[k] for k in range(3)] for v in V]

    fkeys = {_key_of(r, vec(r)) for r in fixed}
    fixed_idx = [i for i in range(len(in_cap)) if _key_of(in_cap[i], V[i]) in fkeys]
    if len(fixed_idx) > n_target:
        # 固定 seed 本身超过 50 (本数据里没发生, 留个稳妥的裁法): 按层序留靠前的
        rank = {}
        for li, L in enumerate(layers(V)):
            for i in L:
                rank[i] = li
        fixed_idx.sort(key=lambda i: (rank.get(i, len(in_cap)), i))

    # 第 1 步: 固定 seed 的解先占位, 后面的补选只填空缺, 不会顶掉它们
    chosen, room = set(fixed_idx[:n_target]), n_target - len(fixed_idx[:n_target])

    def far(i):
        """到已选集合的最近距离; 一个都没选时给 inf, 于是退化成按拥挤度挑边界点。"""
        return min((math.dist(N[i], N[j]) for j in chosen), default=float("inf"))

    # 第 2 步: 按层补, 层内先要"别的方法支配不了的", 再要离已选集合最远的
    if room > 0:
        free = {i for i in range(len(V))
                if not any(dominates(o, V[i]) for o in others)}   # 全局前沿上的候选
        for L in layers(V):                     # 层从好到差
            cd = crowding(V, L)                 # 距离打平时用它定先后
            while room > 0:
                cand = [i for i in L if i not in chosen]
                if not cand:
                    break
                i = max(cand, key=lambda k: (k in free, far(k), cd[k], -k))
                chosen.add(i); room -= 1
            if room == 0:
                break

    sel = [in_cap[i] for i in sorted(chosen)]
    return sel, n_target, fsrc, dropped


def main():
    srcs = {
        "ATPlace2.5D": (load(PF / "AT/50set.csv"), "AT/50set.csv"),
        "RLPlanner": (load(PF / "RL/result.csv"), "RL/result.csv"),
        "MILP": (load(PF / "ILP/result.csv"), "ILP/result.csv"),
    }
    s121 = load(PF / "FM/result_seed121_7case.csv")
    s3 = load(PF / "FM/result_seed3_4case.csv")

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
        fm, n_target, fsrc, dropped = fm_pick(
            c, s121, s3, cap, [[float(r[k]) for k in M] for r in rows])
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
