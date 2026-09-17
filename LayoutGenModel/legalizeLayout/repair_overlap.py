"""Stage A: 最小重叠修复。只推动重叠连通分量内的 chiplet, 其余一动不动。

算法 —— 迭代扫描 + 逐对最小穿透轴分离
--------------------------------------
每一轮扫描, 遍历当前所有重叠对, 各自沿分离代价较小的那个轴把这对推开。所有对的
位移累加后一起施加, 再夹回画布内。重复到没有重叠为止。

三个必要的细节, 每一个都是实测踩出来的
--------------------------------------

**1. 按剩余空间分配位移, 而不是各推一半。**
画布边界的墙会让位移凭空消失: 贴墙的 chiplet 收到向外的推力后被 clip 回原位, 它的
邻居却按"各推一半"只走了半步。这里改成按**各自还能往那个方向走多远**按比例分配 ——
贴墙的一方空间为 0, 于是另一方吃掉全部位移。

**2. 逐对死锁检测, 卡住就换轴。**
"最小穿透轴"在这里常常是**被塞满的那个轴**。实测 syn1: 链 I|C|G|D|A 两端贴墙, 这
五个 chiplet 的宽度和是 41mm, 而画布只有 40.10mm —— x 方向**根本无解**, 但每对在
x 上的穿透都只有 0.15~0.3mm, 远小于 y 上的 5~6.7mm, 所以最小穿透轴永远选中 x, 每轮
G 被左侧的 C 推向右、又被右侧的 D 推向左, 位移互相抵消, 重叠面积卡在 4.9 一动不动。

修法是记住每一对的穿透深度: 连续 ``switch_after`` 轮没有变小, 就永久改用另一个轴。
syn1 于是把 D 沿 y 上移约 9mm, x 链从 41mm 缩到 30mm, 立刻松开。

**3. 不要求全局重叠面积严格下降。**
试过"只接受让总面积严格下降的移动", 在链式重叠上直接死锁 —— 把 A 推离 B 恰好让 B
压到 C 上, 总面积不降, 于是一个候选都不能接受。扫描法不做这种全局筛选, 只要每一对
都在往分开的方向走, 面积自然收敛。

阻尼与收敛
----------
扫描本身很便宜 (纯 numpy, 几百轮不到 0.1 秒), 所以不做阻尼回退这种需要调参的机制,
直接跑满 ``max_sweeps``, 没收敛就在报告里标 ``converged=false``。合法性是硬要求,
不静默放过 —— 由最终 ``is_legal`` 断言把关。

热守卫 —— 在候选之间选, 不在循环里卡
-----------------------------------
**在扫描循环里做守卫是不可行的**, 实测过两种都失败:

* 折半线搜索: 神经前向对 0.26mm 的移动就给出 +0.001K 抖动, ``peak <= cur_peak``
  几乎每轮都不成立, frac 一路折半到 1/16, 位移被压到 0.264mm —— 而重叠深度是
  1.045mm, 合法化直接失败。
* 把守卫编进目标函数 (梯度下降那版): 目标里"去重叠"和"降温"在链式重叠上方向相反,
  实测直接把结果推开 —— Case7 重叠 3->11 对, 外接框缩 22%。

所以守卫放在**候选选择**这一层: 有 2^P 个分轴分配, 其中往往有**多个**都能跑到合法。
神经模型只给这几个终局各打一次分 (实测每 case 十几次前向, 总耗时 <0.1s), 按字典序选:

    合法 > 温度在容忍带内 > 线长小 > 面积小 > 位移小

**温度用容忍带而不是卡零**。理由是代理模型自身的分辨力: 组内 MAE 0.50°C, 且实测
**单芯片挪 0.05mm (远小于任何真实重排) 读数就跳 +0.14K**, 挪 0.2mm 跳到 +0.66K ——
那是 64x64 光栅化换格, 不是物理。卡 0.0 等于把光栅噪声当真约束: 实测 acend910 的
16 组静态分配 + 5 组动态策略**无一例外**升温 (最少 +0.27K), 而它比模型自己的 MAE
还小。带宽内不再区分温度高低, 直接比线长。

线长和面积在合法化阶段允许上升: 去重叠本身就要把 chiplet 摊开, 实测 hp11_m 八组
分轴策略无一例外让线长涨 3%~57%。这笔账由 Stage B 去还。

若**没有任何**候选落在容忍带内 (几何上就没这条路), 取升温最少的那个并置
``legality_overrode_thermal_guard`` + 记下抬升量, 由报告留痕 —— 绝不静默违反。
"""
from __future__ import annotations

import numpy as np

from _bootstrap import setup as _setup

_setup()  # 把 legalizeLayout/ 挂上 sys.path, 下面才能 import 同目录模块

from layout_io import bbox_area_mm2  # noqa: E402
from verify import DEFAULT_GEOM_TOL_MM, find_overlap_pairs  # noqa: E402

__all__ = ["repair_overlap", "overlap_components"]


def overlap_components(pairs, V: int) -> np.ndarray:
    """由重叠对构建无向图, 返回每个节点的分量标签 (不在任何对里的节点为 -1)。"""
    parent = list(range(V))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j, *_ in pairs:
        union(int(i), int(j))

    labels = np.full(V, -1, dtype=np.int64)
    seen = {}
    for node in range(V):
        root = find(node)
        if root not in seen:
            seen[root] = len(seen)
        labels[node] = seen[root]
    return labels


def _room(cur, sizes, node, axis, direction, origin, side):
    """``node`` 沿 ``axis`` 还能往 ``direction`` 走多远 (mm) 才碰到画布。"""
    if direction > 0.0:
        return max(0.0, origin[axis] + float(side) - sizes[node, axis] - cur[node, axis])
    return max(0.0, cur[node, axis] - origin[axis])


def _bbox_keeps(cur, sizes, node, axis, delta, bbox):
    """把 ``node`` 沿 ``axis`` 挪 ``delta`` 后, 是否仍在外接框内。"""
    lo = cur[node, axis] + delta
    hi = lo + sizes[node, axis]
    return lo >= bbox[axis] - 1e-9 and hi <= bbox[axis + 2] + 1e-9


def _pick_sign(cur, sizes, i, j, axis, step, bbox):
    """沿 ``axis`` 分离 i/j 有两个方向, 优先选**不让外接框变大**的那个。

    换轴 (死锁检测) 之后方向若随便取, 很容易把 chiplet 推到布局轮廓之外 —— 实测
    acend910 因此外接框涨 17.76%。两个方向在几何上等价 (都能分开), 所以让"不外扩"
    来定方向是免费的收益。

    两个方向都会外扩时退回按左右顺序定方向, 交给调用方的阻尼/舍入处理。
    """
    natural = 1.0 if cur[i, axis] <= cur[j, axis] else -1.0
    for sign in (natural, -natural):
        if _bbox_keeps(cur, sizes, i, axis, -step * sign, bbox) and _bbox_keeps(
            cur, sizes, j, axis, +step * sign, bbox
        ):
            return sign
    return natural


def _sweep_displacement(cur, sizes, pairs, movable, clearance, origin, side, forced_axis, axis_bias=None):
    """把当前所有重叠对转成一个累加的位移场 (V,2)。

    ``forced_axis`` 是死锁检测的产物: {(i,j): 0|1} —— 这些对改用指定轴分离。
    ``axis_bias`` 指定**未列出**的对 (含扫描中才新出现的对) 走哪根轴, None = 最小穿透轴。
    """
    disp = np.zeros_like(cur)
    cap = np.zeros_like(cur)   # 每个 chiplet 在每根轴上的单轮位移上限
    min_axis = {}
    penetration = {}
    bbox = (
        float(cur[:, 0].min()),
        float(cur[:, 1].min()),
        float((cur[:, 0] + sizes[:, 0]).max()),
        float((cur[:, 1] + sizes[:, 1]).max()),
    )
    for i, j, dx, dy, _area in pairs:
        key = (int(i), int(j))
        natural = 0 if dx <= dy else 1
        axis = forced_axis.get(key, natural if axis_bias is None else int(axis_bias))
        min_axis[key] = natural
        penetration[key] = dx if natural == 0 else dy

        depth = dx if axis == 0 else dy
        step = depth + clearance
        sign = _pick_sign(cur, sizes, i, j, axis, step, bbox)
        room_i = _room(cur, sizes, i, axis, -sign, origin, side)
        room_j = _room(cur, sizes, j, axis, +sign, origin, side)
        share_i = 1.0 if step <= 0.0 else min(1.0, room_i / step)
        share_j = 1.0 if step <= 0.0 else min(1.0, room_j / step)
        total = share_i + share_j
        if total <= 0.0:
            continue
        cap[i, axis] = max(cap[i, axis], depth)
        cap[j, axis] = max(cap[j, axis], depth)
        if movable[i]:
            disp[i, axis] -= step * (share_i / total) * sign
        if movable[j]:
            disp[j, axis] += step * (share_j / total) * sign

    # 位移封顶: 一个 chiplet 可能同时被好几对推, 全额累加会把位移叠成穿透深度的好几倍,
    # 一轮就撞出新重叠 —— 实测 acend910 重叠面积 36.7 -> 84.0, hp11_m 11.0 -> 110.0,
    # 越推越乱。封顶到"它自己要解决的最深那对穿透"就不会过冲。
    np.clip(disp, -np.maximum(cap, 0.0), np.maximum(cap, 0.0), out=disp)
    return disp, min_axis, penetration


def _clamp(cur, sizes, origin, side):
    """夹回画布。画布边长 >= 输入 bbox, 正常情况下不会有 chiplet 被夹到变形。"""
    np.clip(cur[:, 0], origin[0], origin[0] + float(side) - sizes[:, 0], out=cur[:, 0])
    np.clip(cur[:, 1], origin[1], origin[1] + float(side) - sizes[:, 1], out=cur[:, 1])
    return cur


def _sweep_to_legal(lower, sizes, origin, side, movable, cfg, forced_axis, adaptive, budget=None):
    """按给定的分轴分配跑到收敛或跑满预算, 返回 ``(cur, info)``。

    ``forced_axis``: {(i,j): 0|1}, 指定这些对沿哪根轴分离 (未列出的用最小穿透轴)。
    ``adaptive``: 是否启用逐对死锁检测换轴。

    这里**完全不碰神经模型** —— 纯几何。守卫不放在循环里, 是因为每轮扫描都过一遍
    线搜索会把 2^P 次枚举的成本乘以几十倍, 而且实测会把位移压到解不开重叠 (见模块
    docstring)。守卫改由 ``repair_overlap`` 在候选之间选择来承担: 若干个分轴分配都能
    跑到合法, 神经模型只给这几个终局打分。
    """
    geom_tol = float(cfg.get("geom_tol_mm", DEFAULT_GEOM_TOL_MM))
    clearance = float(cfg.get("clearance_mm", 0.0))
    max_sweeps = int(budget if budget is not None else cfg.get("max_sweeps", 500))
    damping = float(cfg.get("damping", 1.0))
    switch_after = int(cfg.get("switch_axis_after", 8))
    axis_bias = cfg.get("axis_bias", None)

    cur = lower.copy()
    info = {"sweeps": 0, "converged": False, "axis_switches": 0}

    forced_axis = dict(forced_axis)

    stalled: dict = {}
    best_pen: dict = {}
    while info["sweeps"] < max_sweeps:
        pairs = find_overlap_pairs(cur, sizes, geom_tol)
        if not pairs:
            info["converged"] = True
            break
        info["sweeps"] += 1

        disp, min_axis, penetration = _sweep_displacement(
            cur, sizes, pairs, movable, clearance, origin, side, forced_axis, axis_bias
        )

        if adaptive:
            # 死锁检测: 这一对的穿透连续多轮没变小, 就永久改用另一个轴
            for key, pen in penetration.items():
                if key in best_pen and pen >= best_pen[key] - geom_tol:
                    stalled[key] = stalled.get(key, 0) + 1
                    if stalled[key] >= switch_after and key not in forced_axis:
                        forced_axis[key] = 1 - min_axis[key]
                        info["axis_switches"] += 1
                else:
                    best_pen[key] = pen
                    stalled[key] = 0

        cur = _clamp(cur + (damping * disp), sizes, origin, side)
        cur[~movable] = lower[~movable]   # 不在重叠分量里的 chiplet 位移恒为 0

    return cur, info


def _geometric_score(cur, lower, sizes, geom_tol):
    """越小越好: (重叠对数, 重叠总面积, 总位移)。前两项是硬目标, 位移是 tie-break。"""
    pairs = find_overlap_pairs(cur, sizes, geom_tol)
    return (
        len(pairs),
        round(float(sum(p[4] for p in pairs)), 9),
        round(float(np.linalg.norm(cur - lower, axis=1).sum()), 9),
    )


def _exact_pair_pass(cur, sizes, movable, origin, side, clearance, geom_tol, max_rounds=200):
    """收尾: 对残留的重叠对穷举"把某一个推开到刚好分离"的候选。

    扫描法靠力推, 会过冲也会留残差; 这一步是**位置式**的 —— 直接把某个 chiplet 挪到
    刚好与对手分开的位置, 位移是解析解, 不会过冲。八个候选 (两轴 x 两方向 x 两方)
    逐个试, 只在

        * 挪完仍在画布内,
        * **不新增任何重叠对**,
        * 重叠对数和总面积严格下降

    时才接受。三个条件一起构成单调性, 所以这一轮必然终止, 也不会把布局搞乱。
    """
    def score_of(state):
        pairs = find_overlap_pairs(state, sizes, geom_tol)
        return len(pairs), round(float(sum(p[4] for p in pairs)), 9)

    cur = cur.copy()
    for _ in range(max_rounds):
        pairs = find_overlap_pairs(cur, sizes, geom_tol)
        if not pairs:
            break
        cur_score = score_of(cur)
        accepted = False
        for i, j, dx, dy, _area in sorted(pairs, key=lambda p: -p[4]):
            for axis in (0, 1):
                if not movable[i] and not movable[j]:
                    continue
                for mover, other, to_high in ((i, j, True), (i, j, False), (j, i, True), (j, i, False)):
                    if not movable[mover]:
                        continue
                    if to_high:
                        delta = (cur[other, axis] + sizes[other, axis] + clearance) - cur[mover, axis]
                    else:
                        delta = (cur[other, axis] - clearance) - (cur[mover, axis] + sizes[mover, axis])
                    candidate = cur.copy()
                    candidate[mover, axis] += delta
                    if not _inside_canvas(candidate[mover], sizes[mover], origin, side, geom_tol):
                        continue
                    if score_of(candidate) < cur_score:
                        cur, cur_score, accepted = candidate, score_of(candidate), True
                        break
                if accepted:
                    break
            if accepted:
                break
        if not accepted:
            break
    return cur


def _inside_canvas(lower, size, origin, side, geom_tol):
    return bool(
        (lower >= origin - geom_tol).all() and (lower + size <= origin + float(side) + geom_tol).all()
    )


def _read_of(lower_single, evaluate):
    """一次前向同时拿 (峰值温度, 神经线长) —— 两个读数共用同一个 cond, 不要跑两次。"""
    peaks, wls = evaluate(np.asarray(lower_single, dtype=np.float64)[None])
    return float(peaks[0]), float(wls[0])


def _excess_of(peak, wl, base_peak, base_wl, thermal_tol, wl_tol):
    """超出一条 (或两条) 容忍带的归一化幅度, 0 = 全部在带内。

    每一项都除以**它自己的**带宽, 所以 "温度超 2 倍带宽" 与 "线长超 2 倍带宽" 同权,
    不会因为 K 和 % 量纲不同而让某一项吃掉另一项。rank 的排序和收尾的准入都用它,
    两处判据必须是同一个 —— 否则收尾能靠"换一种口径"把自己放进来。
    """
    thermal_excess = max(0.0, (peak - base_peak) - thermal_tol) / max(thermal_tol, 1e-9)
    wl_excess = max(0.0, (wl / base_wl - 1.0) - wl_tol) / max(wl_tol, 1e-9)
    return thermal_excess + wl_excess


def repair_overlap(
    lower,
    sizes,
    origin,
    side,
    cfg,
    evaluate=None,
    cond=None,
    device=None,
    verbose=False,
):
    """返回 ``(new_lower, stats)``。

    ``evaluate`` 是 ``(C,V,2) -> (peaks, wls)`` (mm 左下角坐标), 用于热/线长守卫;
    传 None 跳过守卫。``cond`` / ``device`` 保留在签名里只为接口统一, 当前实现不需要。

    为什么要在几种分轴策略之间选
    ----------------------------
    分轴 (每一对沿 x 还是 y 分离) 没有一种启发式对所有布局都对。实测 12 个 case:
      * ``natural`` (最小穿透轴) 解不开 syn1 和 Case7 —— 那两个的重叠穿透在 x 上只有
        0.15~0.3mm, 但 x 方向已经被墙和邻居塞满, 反而要换到穿透 5~6mm 的 y 才解得开。
      * ``adaptive`` 解开了上面两个, 却把 hp11_m 从 3 对恶化到 6 对。
      * ``flip`` (全部换轴) 又是另一种取舍。
    这三个策略都只跑 0.1 秒, 所以直接都跑一遍, 按几何结果取最好的。重叠对很少
    (实测每个 case 0~4 对), 这个"多策略"开销可以忽略, 而且不需要任何调参。

    绝不劣化
    --------
    若所有策略的结果都不如输入, 就**原样返回输入**并置 ``gave_up=true``。宁可交回一个
    未合法的布局并明确报告, 也不交回一个"更乱"的布局。
    """
    geom_tol = float(cfg.get("geom_tol_mm", DEFAULT_GEOM_TOL_MM))
    lower = np.asarray(lower, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    V = lower.shape[0]

    initial_pairs = find_overlap_pairs(lower, sizes, geom_tol)
    stats = {
        "enabled": True,
        "overlap_pairs_start": len(initial_pairs),
        "movable_nodes": 0,
        "gave_up": False,
        "policy": None,
        "policies_tried": [],
        "legality_overrode_thermal_guard": False,
        "wl_rise_tolerated": False,
    }
    if not initial_pairs:
        stats.update(_finish(lower, lower, sizes, geom_tol, movable=None))
        stats.update({"sweeps": 0, "converged": True, "axis_switches": 0})
        return lower, stats

    labels = overlap_components(initial_pairs, V)
    movable = labels >= 0
    stats["movable_nodes"] = int(movable.sum())
    stats["components"] = int(len(set(labels[movable].tolist())))

    input_score = _geometric_score(lower, lower, sizes, geom_tol)
    best = (input_score, lower.copy(), None, None)

    keys = sorted((int(i), int(j)) for i, j, *_ in initial_pairs)
    max_enum = int(cfg.get("enumerate_max_pairs", 8))

    if len(keys) <= max_enum:
        # 穷举分轴分配。为什么需要穷举: 可行的那组分配常常是**混合**的, 静态启发式
        # 命中不了 —— 实测 acend910 只有 (A-B→x, A-C→y, C-F→x, E-F→y) 可行
        # (A、B 在 y 上要占 47.4mm > 画布 44.93mm, 所以 A-B 不能走 y), 而
        # natural 解不开、adaptive 和 flip 恰好都把 A-B 推到 y 上。
        # 重叠对实测只有 0~4 对, 2^4 次扫描 (每次约 30ms) 完全付得起。
        runs = []
        for mask in range(1 << len(keys)):
            forced = {k: (mask >> b) & 1 for b, k in enumerate(keys)}
            runs.append((f"enum:{mask:0{len(keys)}b}", forced, False, {}))
    else:
        runs = []

    # 穷举之外**始终**再跑 adaptive。它不表达成静态分配: 每对各自在自己卡住的轮次
    # 换轴, 走出的是一条**顺序相关**的混合路径, 因此不等于任何一个 mask。
    # 换轴的早晚由 switch_axis_after 决定, 而这个早晚会走到不同的终局 —— 实测
    # acend910 在早换/晚换下的最终温度和线长差得出来, 而"温度不升"是硬约束,
    # 多几个终局就多几次挑到干净解的机会。三个阈值一共只多花 ~0.1 秒。
    base_switch = int(cfg.get("switch_axis_after", 8))
    for tag, switch in (
        ("adaptive_fast", max(2, base_switch // 4)),
        ("adaptive", base_switch),
        ("adaptive_slow", base_switch * 4),
    ):
        runs.append((tag, {}, True, {"switch_axis_after": switch}))

    # axis_bias: 让**扫描中才新出现**的对 (不在初始重叠图里, 因此不被任何 mask 覆盖)
    # 一律走同一根轴。初始对的分配由 mask 决定, 但去重叠过程中会撞出新的重叠对,
    # 它们此前都按最小穿透轴走 —— 那是启发式, 不是唯一选择。
    for tag, bias in (("bias_x", 0), ("bias_y", 1)):
        runs.append((tag, {}, False, {"axis_bias": bias}))

    if len(keys) > max_enum:
        runs.append(("natural", {}, False, {}))
        flip = {
            k: 1 - (0 if dx <= dy else 1)
            for i, j, dx, dy, _a in initial_pairs
            for k in [(int(i), int(j))]
        }
        runs.append(("flip", flip, False, {}))

    # 先小预算粗筛, 再给有希望的跑满预算 —— 免得 2^P 个候选各跑 500 轮
    screen = int(cfg.get("enumerate_screen_sweeps", 80))
    scored = []
    for name, forced, adaptive, patch in runs:
        run_cfg = {**cfg, **patch}
        candidate, info = _sweep_to_legal(
            lower, sizes, origin, side, movable, run_cfg, forced, adaptive, budget=screen
        )
        score = _geometric_score(candidate, lower, sizes, geom_tol)
        scored.append((score, name, forced, adaptive, patch, candidate, info))

    scored.sort(key=lambda item: item[0])
    legal_runs = [item for item in scored if item[0][0] == 0]
    refine_runs = (legal_runs or scored)[: int(cfg.get("enumerate_refine_top", 16))]

    guard_cfg = cfg.get("guard", {}) or {}
    thermal_tol = float(guard_cfg.get("thermal_tol_k", 0.0))
    wl_tol = float(guard_cfg.get("wl_tol", 0.0))
    bbox_init = float(bbox_area_mm2(lower, sizes))
    base_peak = base_wl = None
    if evaluate is not None:
        base_peak, base_wl = (float(v[0]) for v in evaluate(lower[None]))

    candidates = []
    for _geo, name, forced, adaptive, patch, screened, info in refine_runs:
        if info["converged"]:
            candidate = screened      # 粗筛内就收敛了, 跑满预算也是同一个结果
        else:
            candidate, info = _sweep_to_legal(
                lower, sizes, origin, side, movable, {**cfg, **patch}, forced, adaptive
            )
        entry = {"name": name, "lower": candidate, "info": info, "geo": _geometric_score(candidate, lower, sizes, geom_tol)}
        if evaluate is not None:
            peak, wl = (float(v[0]) for v in evaluate(candidate[None]))
            entry.update(
                peak=peak,
                wl=wl,
                thermal_delta=peak - base_peak,
                wl_delta_rel=(wl / base_wl - 1.0) if base_wl else 0.0,
                bbox_delta_rel=bbox_area_mm2(candidate, sizes) / bbox_init - 1.0 if bbox_init else 0.0,
            )
        candidates.append(entry)

    def rank(entry):
        """选择次序 (用户口径, 逐条都短):

            1. 残留重叠对数                    硬要求: 合法性 > 守卫
            2. 残留重叠面积                    同上, 差多少也要看
            3. 温度落在容忍带内                出带 = 劣化
            4. 线长落在容忍带内                出带 = 劣化
            5. 都出带时, 超出各自的**带宽**最少 (按带宽归一, 无量纲可比)
            6. 线长                            带内比小
            7. 外接框                          更软: 尽量小
            8. 位移                            最小干预

        第 1、2 项**必须在守卫之前**: 候选全不合法时 (几何上就没这条路), 排序不能退化成
        "在一堆坏布局里挑线长最小的那个"。实测 Case6_candidate35: `natural` 那组自己已经
        跑到只剩 4 对重叠, 但线长比 `adaptive_fast` 多涨 0.34 个百分点, 于是守卫把**还剩
        12 对**的 `adaptive_fast` 选中了, 只能靠解析收尾硬拖到 1 对 —— 合法性是硬要求,
        不能因为"两个都出带、它出得少一点"就把它排到后面。原先的写法只有 ``illegal`` 一个
        0/1 旗标, 不区分"差多少", 所以七组全非法时第一关键字全部并列, 直接比到第 6 项。

        两条带都是**模型自己的容错**, 不是拍的数:
            温度 0.50 K  —— 热代理的组内 MAE (实测单芯片挪 0.05mm 读数就跳 +0.14K,
                            挪 0.2mm 跳 +0.66K, 那是 64x64 光栅化换格, 不是物理)
            线长 1.20 %  —— 线长 GNN 的测试集 MAPE (mae_log 0.0120)
        低于带宽的升降, 模型分辨不出来, 卡 0.0 只是在拟合噪声。带内一律视为"没变",
        由第 6 项 (线长) 接手做区分 —— 于是 "两条指标不劣化, 且线长尽量小" 就是这个
        元组的自然读法。

        第 5 项用**各自的带宽**做单位, 所以"温度超 2 倍带宽"和"线长超 2 倍带宽"同权,
        不会因为 K 和 % 量纲不同而让某一项吃掉另一项。

        若一个候选都没进带, 第 5 项给出最不坏的那个, 由
        ``legality_overrode_thermal_guard`` / ``legality_overrode_wl_guard`` 留痕
        —— 绝不静默违反。

        注意 ``rank`` 只排序**不筛除**: 出带的候选照样能赢, 只要没有比它更合法的。
        """
        if "peak" not in entry:
            return (entry["geo"][0], entry["geo"][1], 0, 0, 0.0, 0.0, 0.0, entry["geo"][2])
        return (
            entry["geo"][0],
            entry["geo"][1],
            0 if entry["thermal_delta"] <= thermal_tol else 1,
            0 if entry["wl_delta_rel"] <= wl_tol else 1,
            _excess_of(entry["peak"], entry["wl"], base_peak, base_wl, thermal_tol, wl_tol),
            entry["wl_delta_rel"],
            entry["bbox_delta_rel"],
            entry["geo"][2],
        )

    candidates.sort(key=rank)
    chosen = candidates[0] if candidates else None

    for entry in candidates:
        stats["policies_tried"].append(
            {
                "policy": entry["name"],
                "overlap_pairs": entry["geo"][0],
                "overlap_area_mm2": entry["geo"][1],
                "displacement_total_mm": entry["geo"][2],
                "converged": entry["info"]["converged"],
                "peak_temp_C": entry.get("peak"),
                "thermal_delta_C": entry.get("thermal_delta"),
                "wl_delta_rel": entry.get("wl_delta_rel"),
                "bbox_delta_rel": entry.get("bbox_delta_rel"),
                "thermal_guarded": (
                    None
                    if "peak" not in entry
                    else bool(entry["thermal_delta"] <= thermal_tol)
                ),
                "wl_guarded": (
                    None
                    if "peak" not in entry
                    else bool(entry["wl_delta_rel"] <= wl_tol)
                ),
            }
        )

    if chosen is None:
        cur, info, policy = lower.copy(), {"sweeps": 0, "converged": False, "axis_switches": 0}, None
    else:
        cur, info, policy = chosen["lower"], chosen["info"], chosen["name"]
    stats["policy"] = policy

    # ---- 位置式收尾 ----
    # 放在"原样返回"判定**之前**: 扫描够不着的时候, 收尾往往是唯一的出路。实测
    # hp11_m, 八组静态分配 + adaptive 跑满 500 轮没有一组能合法 (最好的一组仍剩 1 对),
    # 只有"把某个 chiplet 挪到刚好分离"能收掉它。收尾只做解析解的最小位移, 不过冲。
    if bool(cfg.get("exact_polish", True)):
        cur_score = _geometric_score(cur, lower, sizes, geom_tol)
        polished = _exact_pair_pass(
            cur, sizes, movable, origin, side, float(cfg.get("clearance_mm", 0.0)), geom_tol
        )
        if _geometric_score(polished, lower, sizes, geom_tol) < cur_score:
            # 收尾**不看守卫**。它只做"把某一对推开到刚好分离"的解析解位移, 且只在
            # (残留对数, 残留面积) 严格下降、不新增重叠、不出画布时才走 —— 每一步都在
            # 朝合法走。拿 T/WL 去否决它, 等于为了保线长而**保留一个不合法的布局**,
            # 与第一条优先级 (合法性 > 守卫) 直接冲突。
            #
            # 实测 Case6_candidate35: 选中 natural (还剩 4 对) 之后收尾被守卫拦下, 最终
            # 交出 4 对重叠的布局。之所以拦得住, 恰恰是因为那个局面线长**本来就出带**
            # (+2.16% > 1.20%) —— 归一化超出量已经在带外, 再涨一点点就被判成"变坏"。
            #
            # 代价不隐藏: 收尾前后各测一次, 差值记进 polish_guard_cost, 与
            # legality_overrode_thermal_guard / _wl_guard 一同留痕 —— 放宽不等于不记账。
            if evaluate is not None and base_wl:
                before_peak, before_wl = _read_of(cur, evaluate)
                cur = polished
                after_peak, after_wl = _read_of(cur, evaluate)
                stats["polish_guard_cost"] = {
                    "thermal_delta_k": float(after_peak - before_peak),
                    "wl_delta_rel": float(after_wl / before_wl - 1.0) if before_wl else 0.0,
                }
            else:
                cur = polished

    if _geometric_score(cur, lower, sizes, geom_tol) >= input_score:
        stats["gave_up"] = True
        cur = lower.copy()
        stats.update(_finish(lower, cur, sizes, geom_tol, movable))
        stats.update({"sweeps": 0, "converged": False, "axis_switches": 0})
        stats["thermal_delta_k"] = 0.0
        stats["wl_delta_rel"] = 0.0
        stats["bbox_delta_rel"] = 0.0
        if verbose:
            print("    [repair_overlap] 所有策略都不如输入, 原样返回", flush=True)
        return cur, stats

    stats["sweeps"] = info["sweeps"]
    stats["converged"] = info["converged"]
    stats["axis_switches"] = info["axis_switches"]
    stats.update(_finish(lower, cur, sizes, geom_tol, movable))

    if evaluate is not None:
        # 守卫数值按**实际返回的那个布局**重算 —— 收尾可能改了它。
        # 不沿用候选表里那份: 表是粗筛终局, 收尾会把布局再动一点。
        peak, wl = (float(v[0]) for v in evaluate(cur[None]))
        stats["thermal_delta_k"] = peak - base_peak
        stats["wl_delta_rel"] = float(wl / base_wl - 1.0) if base_wl else 0.0
        stats["bbox_delta_rel"] = (
            float(bbox_area_mm2(cur, sizes) / bbox_init - 1.0) if bbox_init else 0.0
        )
        # 出容忍带才叫"劣化", 需要留痕; 带内的升降是模型分辨不出来的波动。
        stats["legality_overrode_thermal_guard"] = bool(stats["thermal_delta_k"] > thermal_tol)
        stats["legality_overrode_wl_guard"] = bool(stats["wl_delta_rel"] > wl_tol)
        stats["wl_rise_tolerated"] = bool(stats["wl_delta_rel"] > 0.0)

    if verbose:
        print(
            f"    [repair_overlap] policy={policy} sweeps={stats['sweeps']} "
            f"overlap {stats['overlap_pairs_start']}->{stats['overlap_pairs_end']} "
            f"converged={stats['converged']} max_disp={stats['max_displacement_mm']:.3f}mm",
            flush=True,
        )
    return cur, stats


def _finish(lower, cur, sizes, geom_tol, movable):
    """位移统计。``non_movable_max_displacement_mm`` 必须是 0 —— 这是"只微调重叠部分"
    的可验证证据, 由报告断言。"""
    displacement = np.linalg.norm(cur - lower, axis=1)
    return {
        "overlap_pairs_end": len(find_overlap_pairs(cur, sizes, geom_tol)),
        "max_displacement_mm": float(displacement.max()) if displacement.size else 0.0,
        "displacement_total_mm": float(displacement.sum()),
        "non_movable_max_displacement_mm": (
            float(displacement[~movable].max())
            if movable is not None and (~movable).any()
            else 0.0
        ),
        "displacement_mm": displacement.tolist(),
        "bbox_area_mm2": float(bbox_area_mm2(cur, sizes)),
    }


def format_stats(stats) -> str:
    if not stats.get("enabled"):
        return "disabled"
    return (
        f"overlap {stats.get('overlap_pairs_start', 0)}->{stats.get('overlap_pairs_end', 0)} "
        f"sweeps={stats.get('sweeps', 0)} converged={stats.get('converged')} "
        f"axis_switches={stats.get('axis_switches', 0)} "
        f"max_disp={stats.get('max_displacement_mm', 0.0):.3f}mm"
    )
