"""Stage B: 在双硬守卫下做线长优化。

守卫 (用户口径)
--------------
    峰值温度  ΔT <= 0    硬守卫, 一点不能升
    神经线长  ΔWL <= 0   硬守卫, 一点不能升
    外接框    尽量不要增大  软偏好 (进目标函数, 另有 bbox_tol 过滤)

为什么这个实现是"便宜"的
------------------------
朴素做法是每步对全部 ``candidates`` 个候选各跑一次热代理 + 一次神经线长。
按实测吞吐 (4.2-4.8 ms/布局 热 + 1.8-3.4 ms/布局 线长), 32 候选 x 400 步
= 12.8 万次评估, 单 case 要 80 秒以上, 12 个 case 就是十几分钟。

这里换成三段漏斗, 把神经前向的候选数从 ``candidates`` 降到 ``top_k``:

    1. 几何过滤   纯 numpy 向量化, 不碰 GPU   (candidates -> admissible)
    2. 解析 HPWL  纯 numpy, 用于排序           (admissible -> top_k)
    3. 热 + 线长  各一次批量前向, k 个一起算    (top_k -> 接受/拒绝)

单步成本 = 一次 batch-k 热前向 + 一次 batch-k 线长前向。k=4 时约 25 ms,
200 步约 5 秒/case。用 HPWL 排序而不是直接上神经网络, 是因为 HPWL 与神经线长
同向 (thermal_refine.py:194-202 记录了 11/12 个 case 符号一致), 排序用便宜的就够了,
神经模型只在最后做**判定**。

为什么不做真正的双模型联合搜索: 双硬守卫叠加已经把可行域压得很窄, 任何更复杂的
搜索 (模拟退火 / 多节点联合移动) 都无法把"能同时降低 T 和 WL 的移动"变多, 只会
增加常数开销。收益在候选质量, 不在搜索策略。

热点冻结 + 牵引候选 (用户口径)
------------------------------
    "热点在布局的哪个位置, 合法化之后固定该位置的 chiplet, 然后让其他有链接关系
     的 chiplet 互相靠拢以减少线长 (不劣化热点: 劣化 = 升 0.5K 以上)"

三件事:

1. **冻结**: ``locate_hotspot`` 用最热的 top-5% 格子定出热点区域, 盖住它的
   chiplet 一律不动 (``frozen`` 掩码, 轮转里直接不选它们)。热点件自己不动,
   它的温度只会被邻居影响, 所以再叠一条**热点区守卫**: 候选的热点区最高温不得
   超过 ``热点区基线 + thermal_tol``, 这才是"不劣化热点"的直接实现 —— 全局峰值
   守卫管的是全局峰值, 管不到"峰值换了个地方但热点件自己升了"。

2. **牵引候选**: 原来的候选是各向同性的均匀方盒, 靠 200 步随机撞出线长收益。
   这里改成**沿牵引方向**采样 —— 每个节点算一次它所有连边对端的 wireCount 加权
   重心, 候选就沿着"指向这个重心"的方向按多档步长取, 再加一点横向抖动。方向和
   线长要的方向本来就是同一个, 于是候选预算全部花在有效方向上。

3. **各向同性兜底**: 牵引方向常常被几何挡住 (目标位置压着别的 chiplet), 所以保留
   一部分均匀候选做探索, 由 ``pull_share`` 分配比例。

牵引目标每步**重算**: 对端自己也在动, 一次算死会让两个互相靠拢的节点停在半路。

坐标约定: 几何全在 mm 左下角坐标上算 (与 verify.py 同源), 只在调神经模型时
转成归一化中心坐标。``radius`` 系列参数的单位是 **mm**(不是归一化坐标), 与 ``cur``
同量纲 —— 所以牵引步长 ``scale * radius`` 和牵引距离 ``|delta|`` 可以直接比较。
"""
from __future__ import annotations

import numpy as np

from _bootstrap import setup as _setup

_setup()  # 把 legalizeLayout/ 挂上 sys.path

from layout_io import bbox_area_mm2, to_norm_centers  # noqa: E402
from verify import DEFAULT_GEOM_TOL_MM  # noqa: E402

__all__ = ["refine_wirelength"]

# 半径自适应系数。与 thermal_refine.py 保持同一套实测结论:
#   * 卡住时**缩小**半径 —— 布局已把画布填满, 合法空隙是贴着当前位置的一条窄缝,
#     把方盒开大只会让候选散落到已占区域上, 全被几何过滤掉 (thermal_refine.py:85-93
#     记录了 12 个 case 的实测对比: 缩小 vs 放大, ΔK -6.34 vs -5.66, 接受 73 vs 53)。
#   * 接受后略放大, 保持探索步长。
_STUCK_SCALE = 0.97
_REJECT_SCALE = 0.98
_ACCEPT_SCALE = 1.05

# 接受判据的容差。(热, 线长) 都是浮点量, 用相对容差避免量纲差 3 个数量级时的误判。
_ACCEPT_REL_TOL = 1e-9


def _legal_mask(lower_batch, sizes, origin, side, geom_tol):
    """(C,V,2) -> (C,) bool, True 表示该候选整体合法 (无重叠且不出画布)。

    全向量化。公开的 ``verify.find_overlap_pairs`` 是逐对 Python 循环, 每步对
    几十个候选调用它会把单步耗时推高两个数量级, 所以这里单独写一份; 两者判据
    完全一致 (盒判交 + 容差), 最终结论仍由 ``verify`` 独立复核。

    归约必须把 **(V,V) 两根轴都消掉**。少消一根会得到 (C,V) 的逐节点掩码, 拿它去
    索引 (C,V,2) 会把批次维展平成 (n,2) —— 第 81 个 True 撞上 (1,V,2) 才炸, 而
    n 恰好是 V 的整数倍时**静默**把不合格的候选放进来。所以这里写成 .any(-1).any(-1),
    不写成先 or 再 any。
    """
    upper = lower_batch + sizes[None]
    lo = np.maximum(lower_batch[:, :, None, :], lower_batch[:, None, :, :])
    hi = np.minimum(upper[:, :, None, :], upper[:, None, :, :])
    overlap = ((hi - lo) > geom_tol).all(axis=-1)  # (C,V,V) 逐对判交
    diag = np.arange(sizes.shape[0])
    overlap[:, diag, diag] = False                 # 自己和自己不算重叠
    limit = origin[None, None, :] + float(side)
    outside = (lower_batch < origin[None, None, :] - geom_tol) | (upper > limit + geom_tol)
    return ~(overlap.any(axis=-1).any(axis=-1) | outside.any(axis=-1).any(axis=-1))


def _hpwl_proxy(lower_batch, sizes, edge_src, edge_dst, edge_weight):
    """(C,V,2) -> (C,) wireCount 加权的中心曼哈顿距离和, 单位 mm·wire。纯 numpy。"""
    if edge_src.size == 0:
        return np.zeros(lower_batch.shape[0], dtype=np.float64)
    center = lower_batch + sizes[None] * 0.5
    delta = np.abs(center[:, edge_src, :] - center[:, edge_dst, :]).sum(axis=-1)
    return delta @ edge_weight


def _permutation(v, seed, rng):
    """轮转顺序。用轮转而不是随机取点: 步数预算固定时 (200 步 / V 个 chiplet),
    随机取点会让一部分 chiplet 一次都没被访问过, 轮转保证覆盖均匀。"""
    order = list(range(v))
    rng.shuffle(order)
    return order


def _ray_reach(node, unit, cur, sizes, origin, side):
    """从 ``cur[node]`` 沿 ``unit`` 方向最多能走多远 (mm) 而**不撞任何东西、不出画布**。

    闭式解, 不采样。做法: 对每个其他 chiplet j, 两个轴各自解出"与 j 的重叠区间"
    ``[lo, hi]``, 交集就是"会撞上 j"的那段 t; 该区间若整段为负则 j 永远挡不住这条
    射线 (两轴不重叠)。全部取最小即得第一个障碍物。

    为什么需要它: 采样式的多档步长撞不到"几何允许的最近位置" —— 而那个位置恰恰是
    线长最优解。实测 acend910 只有 6 个 chiplet、A 是 5 个小片唯一的对端, 最优解就是
    "5 个小片全部贴到 A 的边上"; 200 步随机采样只把线长从 +11.7% 拉回 +2.2%, 而闭式
    解一步就能给出那个贴边位置。

    平行轴的退化情形 (方向恰与某轴垂直) 用 ``eps`` 顶替分母处理, 不要特判: 此时
    ``(-hh-d0)/eps`` 与 ``(hh-d0)/eps`` 会自动给出 (a) 两轴已重叠 -> 一整条无约束区间,
    或 (b) 该轴已分离 -> 整段在负半轴, 由 ``t_enter < t_exit`` 判定为"挡不住"。
    """
    V = cur.shape[0]
    p = cur[node]
    s = sizes[node]
    origin = np.asarray(origin, dtype=np.float64)

    reach = np.inf
    for axis in (0, 1):
        v = float(unit[axis])
        v = v if abs(v) > 1e-9 else 1e-9
        t_lo = (origin[axis] - p[axis]) / v
        t_hi = (origin[axis] + float(side) - s[axis] - p[axis]) / v
        reach = min(reach, max(t_lo, t_hi))

    half = (sizes + s) / 2.0                      # (V,2) 与各节点的 Minkowski 半宽
    d0 = p[None, :] - cur                         # (V,2) 当前左下角之差
    with np.errstate(divide="ignore", invalid="ignore"):
        t_a = (-half - d0) / unit                 # (V,2)
        t_b = (half - d0) / unit
    lo_axis = np.minimum(t_a, t_b)                # (V,2) 该轴重叠区间的两端
    hi_axis = np.maximum(t_a, t_b)
    t_enter = lo_axis.max(axis=1)                 # (V,) 两轴区间取交
    t_exit = hi_axis.min(axis=1)
    blocking = (t_enter > 0.0) & (t_enter < t_exit)
    blocking[node] = False
    if blocking.any():
        reach = min(reach, float(t_enter[blocking].min()))
    return max(0.0, float(reach)) if np.isfinite(reach) else 0.0


def _pull_targets(centers, edge_src, edge_dst, edge_weight, v):
    """每个节点的 wireCount 加权**连线重心** (mm) —— 牵引目标。无连边者返回 NaN。

    重心用的是**对端当前位置**, 所以每步都得重算: 对端也在往这边靠, 一次算死会让
    两个互相靠拢的节点各自停在"对方出发的地方"。

    ``cond.edge_index`` 是**双向**的 (cond_builder 里 (u,v) 和 (v,u) 都加了),
    所以只按 src 聚合就够, 不需要再对称化一遍。
    """
    target = np.full((v, 2), np.nan, dtype=np.float64)
    if edge_src is None or edge_src.size == 0:
        return target
    num = np.zeros((v, 2), dtype=np.float64)
    den = np.zeros(v, dtype=np.float64)
    np.add.at(num, edge_src, centers[edge_dst] * edge_weight[:, None])
    np.add.at(den, edge_src, edge_weight)
    valid = den > 0.0
    target[valid] = num[valid] / den[valid, None]
    return target


def refine_wirelength(
    lower,
    sizes,
    origin,
    side,
    cfg,
    evaluators,
    cond,
    record=None,
    baseline=None,
    frozen=None,
    hotspot=None,
    region_baseline=None,
    *,
    seed=0,
    verbose=False,
):
    """返回 ``(new_lower, stats)``。起点必须是**已合法**的布局 (Stage A 的输出)。

    只有整体仍然合法、通过守卫、且目标下降的候选才会被接受, 所以输出天然合法,
    调用方不需要再跑一次合法化。

    ``baseline`` = ``(peak, wl)`` —— **本 case 输入布局**的两个读数。守卫参照它,
    不是参照 Stage B 的起点: 起点已经被 Stage A 改过了, 拿它当零点会让 Stage A
    花掉的温度预算不算数, 于是 Stage B 又能用满整条带宽, 净变化直接翻倍冲出容错带
    (实测 acend910: Stage A 已花 +0.36K, Stage B 又爬到 +0.88K)。参照输入则让
    "净变化在带内"由构造保证。不传时退化为参照起点。

    ``frozen`` (V,) bool —— 不许动的节点。热点定位给出的那几个 chiplet 由调用方
    算好传进来 (见 hotspot.py)。被冻结的节点在整个搜索里位移恒为 0。

    ``hotspot`` —— ``locate_hotspot`` 的返回值, 只用它的 ``hot_cell_index``
    (热点格子) 做**热点区守卫**。不传则该守卫关闭。

    ``region_baseline`` —— 输入布局在**同一组热点格子**上的最高温 (℃)。热点区守卫
    的参照值。不传则退化为参照 Stage B 起点。
    """
    steps = int(cfg.get("steps", 200))
    candidates = int(cfg.get("candidates", 16))
    top_k = int(cfg.get("top_k", 4))
    radius = float(cfg.get("radius", 0.30))
    radius_min = float(cfg.get("radius_min", 0.005))
    radius_max = float(cfg.get("radius_max", 0.30))
    patience = int(cfg.get("patience", 80))
    geom_tol = float(cfg.get("geom_tol_mm", DEFAULT_GEOM_TOL_MM))
    w_thermal = float(cfg.get("thermal_weight", 1.0))
    w_wl = float(cfg.get("wl_weight", 1.0))
    w_bbox = float(cfg.get("bbox_weight", 0.3))

    guard = cfg.get("guard", {}) or {}
    thermal_tol = float(guard.get("thermal_tol_k", 0.0))
    wl_tol = float(guard.get("wl_tol", 0.0))
    bbox_tol = float(cfg.get("bbox_tol", 0.0))

    proposal = cfg.get("proposal", {}) or {}
    pull_mode = str(proposal.get("mode", "pull")).lower()
    pull_share = float(proposal.get("pull_share", 0.75))
    pull_scales = [float(s) for s in (proposal.get("scales") or [0.3, 1.0, 2.5, 6.0])]
    pull_jitter = float(proposal.get("jitter", 0.35))
    pull_max_mm = float(proposal.get("max_jump_mm", 8.0))
    hotspot_guard_on = bool((cfg.get("hotspot", {}) or {}).get("guard", True))

    stats = {
        "enabled": True,
        "steps_run": 0,
        "accepted": 0,
        "thermal_rejected": 0,
        "wl_rejected": 0,
        "bbox_rejected": 0,
        "hotspot_rejected": 0,
        "no_legal_candidate": 0,
        "no_improvement": 0,
        "pull_candidates": 0,
        "uniform_candidates": 0,
        "early_stop": False,
        "frozen_chiplets": [],
    }

    lower = np.asarray(lower, dtype=np.float64).copy()
    sizes = np.asarray(sizes, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    V = lower.shape[0]

    edge_src = edge_dst = edge_weight = None
    if cond is not None and getattr(cond, "edge_index", None) is not None and cond.edge_index.numel():
        edge_index = cond.edge_index.cpu().numpy()
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_weight = cond.edge_weight.cpu().numpy().astype(np.float64)

    def to_norm(batch):
        return np.stack([to_norm_centers(b, sizes, origin, side) for b in batch], axis=0)

    # 热点区守卫用的格子。没配 hotspot 就是 None, 该守卫整体关闭。
    hot_cells = None
    if hotspot is not None and hotspot_guard_on:
        hot_cells = np.asarray(hotspot.get("hot_cell_index", []), dtype=np.int64)
        if hot_cells.size == 0:
            hot_cells = None
    stats["hotspot_guard"] = bool(hot_cells is not None)

    def peak_and_wl(batch):
        x_norm = to_norm(batch)
        peaks, region = evaluators.peak_and_region_celsius(x_norm, cond, hot_cells)
        wls = evaluators.wirelength(x_norm, cond, sizes, origin, side).cpu().numpy().astype(np.float64)
        peaks = peaks.cpu().numpy().astype(np.float64)
        region = None if region is None else region.cpu().numpy().astype(np.float64)
        return peaks, wls, region

    if steps <= 0 or candidates <= 0 or V < 2:
        stats["enabled"] = False
        return lower, stats

    # 冻结掩码。被冻的节点在轮转里根本不会被选中, 位移由构造恒为 0。
    frozen_mask = np.zeros(V, dtype=bool)
    if frozen is not None:
        frozen_mask |= np.asarray(frozen, dtype=bool).reshape(-1)
    movable_nodes = np.flatnonzero(~frozen_mask)
    if movable_nodes.size < 2:
        stats["enabled"] = False
        stats["reason"] = f"可动 chiplet 不足 (frozen={int(frozen_mask.sum())}/{V})"
        return lower, stats
    stats["frozen_chiplets"] = [int(i) for i in np.flatnonzero(frozen_mask)]
    stats["num_movable"] = int(movable_nodes.size)

    cur = lower.copy()
    cur_peak, cur_wl, cur_region = peak_and_wl([cur])
    cur_peak, cur_wl = float(cur_peak[0]), float(cur_wl[0])
    cur_region = None if cur_region is None else float(cur_region[0])
    wl_init, bbox_init = cur_wl, bbox_area_mm2(cur, sizes)
    stats["start_peak_temp_C"] = cur_peak
    stats["start_wl_neural_mm"] = cur_wl
    stats["start_bbox_area_mm2"] = bbox_init
    stats["start_hotspot_region_C"] = cur_region

    # 守卫与目标都以**输入布局**为参照 (没传 baseline 就退回起点)
    base_peak, base_wl = (baseline if baseline is not None else (cur_peak, cur_wl))
    stats["baseline_peak_temp_C"] = float(base_peak)
    stats["baseline_wl_neural_mm"] = float(base_wl)
    stats["baseline_hotspot_region_C"] = None if region_baseline is None else float(region_baseline)

    # 天花板 = max(容错带, 本阶段起点)。兜住起点这一项是必须的: 去重叠有时会**被迫**
    # 把线长推出带宽 (实测 acend910 Stage A 出来就是 +11.7%, 任何能解开重叠的布局都
    # 至少这么长)。若天花板只有带宽, 起点在带外时每个候选都不合格, Stage B 一步都
    # 接受不了 —— 那 4 个重叠 case 的线长会永远停在 Stage A 的值, 正是冻死的表现。
    # 兜住起点后, 搜索至少能"不劣化", 并能把线长一路拉回带内。
    peak_ceiling = max(base_peak + thermal_tol, cur_peak)
    wl_ceiling = max(base_wl * (1.0 + wl_tol), cur_wl)
    stats["peak_ceiling_C"] = float(peak_ceiling)
    stats["wl_ceiling_mm"] = float(wl_ceiling)

    # 热点区守卫的天花板, 同样的"兜住起点"逻辑。参照值优先用输入布局在同组格子上的
    # 最高温 (region_baseline); 没传就用 Stage B 起点。
    region_ceiling = None
    if hot_cells is not None:
        region_base = cur_region if region_baseline is None else float(region_baseline)
        region_ceiling = max(region_base + thermal_tol, cur_region)
        stats["hotspot_region_ceiling_C"] = float(region_ceiling)

    def objective(peak, wl, bbox):
        """温度与线长**并列** (两者相对输入的变化等权), 外接框排后面。

        温度必须是个跟线长同量纲的相对量: 峰值 ~90K, 除以自己的基准就得到
        "1% 温度" 与 "1% 线长" 等权, 这正是用户口径里的"并列"。只放线长会让
        搜索一路拿温度换线长 (实测 hp11_m 就是这么从 -2.11K 爬到 +0.54K)。
        """
        return (
            w_thermal * (peak / max(base_peak, 1e-12))
            + w_wl * (wl / max(base_wl, 1e-12))
            + w_bbox * (bbox / max(bbox_init, 1e-12))
        )

    cur_obj = objective(cur_peak, cur_wl, bbox_init)

    rng = np.random.default_rng(seed)
    n_movable = int(movable_nodes.size)
    n_pull = int(round(candidates * max(0.0, min(1.0, pull_share)))) if pull_mode == "pull" else 0
    stats["proposal_mode"] = pull_mode

    def propose(node, target, norm_dir):
        """候选位移 ``(candidates, 2)`` mm + ``is_pull (candidates,)`` bool。

        牵引候选沿"指向连线重心"的方向按 ``pull_scales`` 多档取步长, 封顶
        ``pull_max_mm``; 步长档位刻意含一个 <1 的小档, 因为守卫饱和时唯一能过的
        移动就是**亚网格的小步** (见下面 _select 的注释)。剩下的是各向同性探索,
        用于牵引方向被别的 chiplet 挡住的情况。
        """
        offsets = np.empty((candidates, 2), dtype=np.float64)
        is_pull = np.zeros(candidates, dtype=bool)
        filled = 0
        if n_pull > 0 and np.isfinite(norm_dir) and norm_dir > 1e-9:
            unit = (target - centers[node]) / norm_dir
            perp = np.array([-unit[1], unit[0]], dtype=np.float64)
            # 可达距离: 不超过"几何允许的最近位置", 也不超过目标本身
            reach = min(_ray_reach(node, unit, cur, sizes, origin, side), norm_dir, pull_max_mm)
            for b in range(min(n_pull, candidates)):
                frac = pull_scales[b % len(pull_scales)]
                step = frac * reach
                # 第 0 个候选严格落在扫出的射线上 (就是那个贴边位置), 不加抖动
                lateral = 0.0 if b == 0 else rng.normal() * pull_jitter * max(step, radius)
                offsets[b] = step * unit + lateral * perp
            is_pull[: min(n_pull, candidates)] = True
            filled = min(n_pull, candidates)
        rest = candidates - filled
        if rest > 0:
            offsets[filled:] = (rng.random((rest, 2)) * 2.0 - 1.0) * radius
        stats["pull_candidates"] += filled
        stats["uniform_candidates"] += rest
        return offsets, is_pull

    def select(proxy, is_pull, keep, k):
        """在**两类候选之间轮流取**, 而不是直接取 HPWL 最低的 k 个。

        为什么必须分家: 牵引候选的方向本来就是"降线长"的方向, 按 HPWL 排序时它们
        几乎总排在前面, 于是各向同性探索**永远进不了** top_k, 神经前向只看得到牵引
        候选。实测后果 —— 守卫一旦饱和 (acend910 去重叠后被逼位移 24.9mm, 起点就顶在
        温度天花板下), 唯一能通过的移动是"亚网格小步、ΔT 恰好为 0"那类, 而那类**只有
        各向同性候选里才有**; 全被 HPWL 排序滤掉之后, 搜索一步都接受不了, 线长从
        -16% 退化到 +6%。分家之后每步至少给探索类留出名额。
        """
        picked = []
        groups = []
        for flag in (True, False):
            members = np.flatnonzero(keep & (is_pull == flag))
            if members.size:
                groups.append(members[np.argsort(proxy[members])])
        cursor_local = 0
        while len(picked) < k and any(cursor_local < g.size for g in groups):
            for g in groups:
                if cursor_local < g.size and len(picked) < k:
                    picked.append(int(g[cursor_local]))
            cursor_local += 1
        return np.asarray(picked, dtype=np.int64)

    order = _permutation(n_movable, seed, rng)
    cursor = 0
    idle = 0

    for step in range(steps):
        stats["steps_run"] = step + 1
        if cursor % n_movable == 0:
            order = _permutation(n_movable, seed + cursor, rng)
        node = int(movable_nodes[order[cursor % n_movable]])
        cursor += 1

        # 牵引目标每步重算: 对端也在动
        centers = cur + sizes * 0.5
        targets = _pull_targets(centers, edge_src, edge_dst, edge_weight, V)
        target = targets[node]
        norm_dir = float(np.linalg.norm(target - centers[node])) if np.isfinite(target).all() else np.inf

        offsets, is_pull = propose(node, target, norm_dir)
        batch = np.repeat(cur[None], candidates, axis=0)
        batch[:, node, :] = cur[node] + offsets

        legal = _legal_mask(batch, sizes, origin, side, geom_tol)
        if not legal.any():
            stats["no_legal_candidate"] += 1
            radius = max(radius * _STUCK_SCALE, radius_min)
            idle += 1
            if patience and idle >= patience:
                stats["early_stop"] = True
                break
            continue

        is_pull = is_pull[legal]
        batch = batch[legal]
        # 便宜的量先算: HPWL 排序 + 外接框过滤, 都不过 GPU
        proxy = _hpwl_proxy(batch, sizes, edge_src, edge_dst, edge_weight)
        if w_bbox > 0.0:
            areas = np.array([bbox_area_mm2(b, sizes) for b in batch])
            keep = areas <= bbox_init * (1.0 + bbox_tol)
            stats["bbox_rejected"] += int((~keep).sum())
            if not keep.any():
                radius = max(radius * _REJECT_SCALE, radius_min)
                idle += 1
                if patience and idle >= patience:
                    stats["early_stop"] = True
                    break
                continue
        else:
            areas = np.array([bbox_area_mm2(b, sizes) for b in batch])
            keep = np.ones(batch.shape[0], dtype=bool)

        k = min(int(top_k), batch.shape[0])
        short = select(proxy, is_pull, keep, k)
        sub, sub_areas = batch[short], areas[short]

        # 神经前向只跑 k 个
        peaks, wls, regions = peak_and_wl(sub)

        # 守卫参照**输入布局** (+ 兜住起点, 见上面的 ceiling), 不是上一步。
        #   参照上一步 -> 允许一段可累积的正漂移 (200 步每步放一点, 最坏漂出带宽);
        #   参照 Stage A 终点 -> Stage A 已花掉的温度预算不算数, 净变化翻倍出带。
        # 参照输入则让"净变化在带内"由构造保证; 中间步可以在带内回落再爬, 反而更
        # 容易跳出局部极小。
        ok_thermal = peaks <= peak_ceiling
        stats["thermal_rejected"] += int((~ok_thermal).sum())
        ok_wl = wls <= wl_ceiling
        stats["wl_rejected"] += int((~ok_wl).sum())
        ok = ok_thermal & ok_wl
        if region_ceiling is not None and regions is not None:
            # "不劣化热点": 热点区自己的最高温不许越过 基线+thermal_tol。全局峰值守卫
            # 管不到这一条 —— 峰值可以换个地方出现, 而热点件身边的邻居挤过来照样能把
            # 热点顶上去。被冻的 chiplet 自己不动物理上仍会因邻居靠近而升温, 所以这
            # 一条是必须的, 不是冗余。
            ok_region = regions <= region_ceiling
            stats["hotspot_rejected"] += int((~ok_region).sum())
            ok = ok & ok_region
        if not ok.any():
            radius = max(radius * _REJECT_SCALE, radius_min)
            idle += 1
            if patience and idle >= patience:
                stats["early_stop"] = True
                break
            continue

        objs = np.array([objective(p, w, a) for p, w, a in zip(peaks[ok], wls[ok], sub_areas[ok])])
        best = int(np.argmin(objs))
        if objs[best] < cur_obj - _ACCEPT_REL_TOL * max(abs(cur_obj), 1.0):
            idx = int(np.flatnonzero(ok)[best])
            cur = sub[idx].copy()
            cur_peak, cur_wl = float(peaks[idx]), float(wls[idx])
            if regions is not None:
                cur_region = float(regions[idx])
            cur_obj = float(objs[best])
            stats["accepted"] += 1
            radius = min(radius * _ACCEPT_SCALE, radius_max)
            idle = 0
        else:
            stats["no_improvement"] += 1
            radius = max(radius * _REJECT_SCALE, radius_min)
            idle += 1
            if patience and idle >= patience:
                stats["early_stop"] = True
                break

    stats["end_peak_temp_C"] = cur_peak
    stats["end_wl_neural_mm"] = cur_wl
    stats["end_hotspot_region_C"] = cur_region
    stats["end_bbox_area_mm2"] = bbox_area_mm2(cur, sizes)
    stats["thermal_delta_k"] = float(cur_peak - stats["start_peak_temp_C"])
    stats["wl_delta_mm"] = float(cur_wl - stats["start_wl_neural_mm"])
    stats["wl_delta_rel"] = float(cur_wl / max(stats["start_wl_neural_mm"], 1e-12) - 1.0)
    stats["bbox_delta_rel"] = float(
        stats["end_bbox_area_mm2"] / max(bbox_init, 1e-12) - 1.0
    )
    stats["displacement_total_mm"] = float(np.linalg.norm(cur - lower, axis=1).sum())
    # 被冻结的 chiplet 必须一位没动 —— 这是"固定热点"的可验证证据, 不是口头保证
    if frozen_mask.any():
        stats["frozen_displacement_mm"] = float(
            np.linalg.norm(cur[frozen_mask] - lower[frozen_mask], axis=1).max()
        )
    if region_ceiling is not None and cur_region is not None:
        base_region = cur_region if region_baseline is None else float(region_baseline)
        stats["hotspot_delta_k"] = float(cur_region - base_region)
        stats["hotspot_degraded_beyond_tol"] = bool(cur_region - base_region > thermal_tol)
    return cur, stats


def format_stats(stats) -> str:
    if not stats.get("enabled"):
        return f"disabled ({stats.get('reason', '')})".strip()
    text = (
        f"accepted={stats.get('accepted', 0)} "
        f"thermal_rejected={stats.get('thermal_rejected', 0)} "
        f"wl_rejected={stats.get('wl_rejected', 0)} "
        f"bbox_rejected={stats.get('bbox_rejected', 0)} "
        f"no_improvement={stats.get('no_improvement', 0)} "
        f"no_legal={stats.get('no_legal_candidate', 0)} "
        f"ΔT={stats.get('thermal_delta_k', 0.0):+.2f}K "
        f"ΔWL={stats.get('wl_delta_rel', 0.0) * 100.0:+.2f}%"
    )
    if stats.get("hotspot_guard"):
        text += (
            f" hotspot_rejected={stats.get('hotspot_rejected', 0)}"
            f" ΔT_hotspot={stats.get('hotspot_delta_k', 0.0):+.2f}K"
        )
    frozen = stats.get("frozen_chiplets") or []
    if frozen:
        text += f" frozen={len(frozen)}"
    return text
