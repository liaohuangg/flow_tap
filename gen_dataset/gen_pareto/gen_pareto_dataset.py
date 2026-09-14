#!/usr/bin/env python3
"""gen_pareto_dataset.py — 用两个冻结代理生成「热峰值 / 线长 / bbox 面积」帕累托布局数据集。

流程 (每 system 独立)
---------------------
  L1 (CPU, multiprocessing.Pool)  构造候选库: C0 (L0 原样) + G1 (QAP 重解) + G2 (功耗置换)
                                  + G3 (bbox 等距变换), 每个候选都合法且满足全部不变量
  L2 (GPU, 同 system 批量)        整库一次批量热前向 + 一次批量线长; bbox 面积解析算出
  L3 (GPU, 批量, 迭代)            在热最优的库成员上做 HRNet best-improvement 成对功耗交换,
                                  单调只接受严格改善
  选片                            取帕累托前沿, 落 sidecar; 主语料取前沿上的选点

为什么 C0 恒在库中
------------------
C0 是 L0 记录本身 (逐字节一致)。于是输出**按构造不可能被 C0 支配**: 每个 system 的主语料
至少不比现有生成方案差一个数量级, 且这同时就是消融对照臂。

输入 / 输出
-----------
  源    : placement_dataset_tw/chiplet_dataset_{5..8}.json     (id 20001..40000)
  L0    : after_dataset_process.py 对上述源跑出的结果            (C0 的来源, 也是 V4 的对照臂)
  输出  : <out-dir>/chiplet_dataset_{1..4}.json                 (重编号为 id 1..20000)
          <out-dir>/front/chiplet_dataset_{1..4}.json           (完整帕累托前沿 sidecar)
          <out-dir>/front/meta.json
          <out-dir>/provenance.json
          <l0-out-dir>/chiplet_dataset_{1..4}.json               (同一重编号下的 L0 对照)

用法
----
    python gen_dataset/gen_pareto/gen_pareto_dataset.py \
        --src-dir Dataset/dataset/placement_dataset/placement_dataset_tw \
        --l0-dir  Dataset/dataset/placement_dataset/_l0_raw \
        --chunk 5 6 7 8 \
        --out-dir Dataset/dataset/placement_dataset/placement_dataset_opt_pareto \
        --l0-out-dir Dataset/dataset/placement_dataset/placement_dataset_l0

必须用 chipdiffusion 环境。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_dataset_lib as L  # noqa: E402

# --------------------------------------------------------------------------------------
# L1: 候选库构造 (纯 CPU, 可并行; 不 import torch)
# --------------------------------------------------------------------------------------
_BANK_CFG: dict = {}


def _bank_job(payload):
    sid, rec = payload
    core = L.load_core(sid, rec)
    if core is None:
        return sid, None, "load_core_failed"
    rng = random.Random(_BANK_CFG.get("seed", 0) + int(sid.split("_")[1]))
    bank = L.build_bank(core, rng,
                        qap_tries=_BANK_CFG.get("qap_tries", 32),
                        qap_restarts=_BANK_CFG.get("qap_restarts", 8),
                        n_power_random=_BANK_CFG.get("power_random", 26),
                        with_geom=_BANK_CFG.get("geom", True))
    if not bank:
        return sid, None, "empty_bank"
    return sid, bank, None


# --------------------------------------------------------------------------------------
# L3: HRNet 引导的成对功耗交换 (best-improvement, 单调)
# --------------------------------------------------------------------------------------
def greedy_power_search(core: L.SystemCore, scorer: L.SurrogateScorer,
                        start: Sequence[float], rounds: int,
                        min_gain: float = L.MAXC_QUANTUM) -> Tuple[List[float], List[List[float]]]:
    """从 start 出发做贪心功耗交换, 只接受明确降低 HRNet 热峰值的交换。

    `min_gain` 必须 > 代理的 run-to-run 噪声底 (~1e-3 °C), 否则贪心会一路追着 GPU
    非确定性走, "改善"全在噪声里。默认取 L.MAXC_QUANTUM (1e-2 °C)。

    返回 (最优功耗向量, 评估过的所有功耗向量)。后者直接作为额外的 G2 候选入库 ——
    它们的热峰值已在搜索中算出, 而线长 (规范功耗口径) 与面积对功耗置换完全不变, 无需重算。
    """
    n = core.n
    if n < 2 or rounds <= 0:
        return list(start), []

    evaluated: List[List[float]] = []

    def evaluate(pws: List[List[float]]) -> List[float]:
        recs = [L.make_record(core, power_slots=p, sid=core.sid) for p in pws]
        keep = [(i, r) for i, r in enumerate(recs) if r is not None]
        if not keep:
            return [float("inf")] * len(pws)
        th = scorer.thermal_batched([r for _, r in keep])
        out = [float("inf")] * len(pws)
        for (i, _), t in zip(keep, th):
            out[i] = t["max_c"]
        return out

    cur = list(start)
    cur_obj = evaluate([cur])[0]
    for _ in range(rounds):
        swaps: List[List[float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(cur[i] - cur[j]) < 1e-12:
                    continue
                p = list(cur)
                p[i], p[j] = p[j], p[i]
                swaps.append(p)
        if not swaps:
            break
        vals = evaluate(swaps)
        evaluated += swaps
        best_i = min(range(len(swaps)), key=lambda k: vals[k])
        if vals[best_i] < cur_obj - min_gain:
            cur, cur_obj = swaps[best_i], vals[best_i]
        else:
            break
    return cur, evaluated


def _geom_key(c: L.Candidate) -> Tuple:
    """(σ, footprint 几何) 的签名 —— 用于挑"结构上互不相同"的贪心起点。"""
    fp = tuple((round(f[0], 6), round(f[1], 6), round(f[2], 6), round(f[3], 6))
               for f in (L.footprint_of(x) for x in c.record["chiplets"]))
    return (tuple(c.sigma) if c.sigma else None, fp)


# --------------------------------------------------------------------------------------
# 单 system 全流程
# --------------------------------------------------------------------------------------
def process_system(core: L.SystemCore, bank: List[L.Candidate], scorer: L.SurrogateScorer,
                   args) -> Optional[dict]:
    # ---- L2: 整库批量打分 (线长用 C0 的功耗向量做排序口径) ----
    records = [c.record for c in bank]
    scores = scorer.score_bank(records, canonical_powers=[core.powers] * len(records))
    for c, s in zip(bank, scores):
        c.objective = (s["max_c"], s["wl"], s["bbox_area"])
    wl_canon = scores[0]["wl"]
    area = scores[0]["bbox_area"]

    # ---- L3: 贪心功耗搜索 ----
    if args.greedy_rounds > 0 and core.n >= 2:
        by_geom: Dict[Tuple, L.Candidate] = {}
        for c in sorted(bank, key=lambda c: c.objective[0]):
            by_geom.setdefault(_geom_key(c), c)
        starts = sorted(by_geom.values(), key=lambda c: c.objective[0])[: max(1, args.greedy_starts)]
        pend: List[L.Candidate] = []
        for st in starts:
            best_pw, _evaluated = greedy_power_search(
                core, scorer, [float(c["power"]) for c in st.record["chiplets"]], args.greedy_rounds,
                min_gain=args.greedy_min_gain)
            rec = L.make_record(core, power_slots=best_pw, sid=core.sid)
            if rec is not None:
                # 只收贪心的**最终结果**, 不收搜索中评估过的那些功耗向量。后者全部被它支配:
                # 功耗置换对线长 (规范功耗口径) 和面积都中性, 而 final 是这批向量里 max_c
                # 最小的 (每轮只在严格改善时才接受), 所以它们永远进不了前沿 —— 收进库只会
                # 白白多花一次批量热前向 (实测 ~200 个候选/system, 约 27% 的时间)。
                pend.append(L.Candidate(record=rec, tag="G2:greedy", sigma=st.sigma,
                                        objective=None))
        if pend:
            th = scorer.thermal_batched([c.record for c in pend])
            for c, t in zip(pend, th):
                c.objective = (t["max_c"], wl_canon, area)
            bank += pend

    # ---- 帕累托前沿 ----
    # 用一个共同的目标量化分辨率, 让 pareto_front 和下面的守卫**用的是同一个支配关系**。
    # 否则守卫的容差 (1e-9) 与精确比较不一致, 就会把纯浮点噪声判成"被 C0 支配"而误报。
    q = L.obj_quantum([c.objective for c in bank])
    front = L.pareto_front(bank, q=q)
    c0_obj = next((c.objective for c in bank if c.tag == "C0"), None)
    if c0_obj is not None:
        # 非支配是构造保证: C0 恒在库中, 前沿里不该出现被 C0 支配的点
        for c in front:
            if L.dominates(c0_obj, c.objective, q):
                raise AssertionError(
                    f"{core.sid}: 前沿成员 {c.tag} 被 C0 支配 (帕累托实现有 bug) "
                    f"C0={c0_obj} {c.tag}={c.objective}")

    # ---- 选片 ----
    rng = random.Random(args.seed + int(core.sid.split("_")[1]) * 7919)
    if args.corpus_mode == "per_geometry":
        selected = select_per_geometry(bank)
    else:
        selected = [_select(front, rng, args)]
    # 头条统计量用该 case 的热最优行 (多行时它就是 headline)
    pick = min(selected, key=lambda c: c.objective[0])

    return {
        "sid": core.sid,
        "selected": selected,
        "headline": pick,
        "front": front,
        "c0_obj": c0_obj,
        "bank_size": len(bank),
        "front_size": len(front),
    }


def _select(front: Sequence[L.Candidate], rng: random.Random, args) -> L.Candidate:
    mode = args.front_select
    if mode == "knee":
        base = L.select_knee(front)
    elif mode == "thermal":
        base = min(front, key=lambda c: c.objective[0])
    elif mode == "random":
        base = front[rng.randrange(len(front))]
    else:
        raise ValueError(mode)
    if len(front) > 1 and rng.random() < args.mix_fraction:
        return front[rng.randrange(len(front))]
    return base


def select_per_geometry(bank: Sequence[L.Candidate]) -> List[L.Candidate]:
    """一个 case 的多行: **每个不同几何取热峰值最优的那个代表**。

    "不同几何" = (σ, footprint 铺排) 这个签名 —— 功耗置换不算新几何。于是同一 case 的
    若干功耗排列会塌缩成一个代表 (取热最优的那个), 而 G1 的多个 QAP 最优、G3 的镜像/转置
    各自成行。这正是让 flow matching 见到"同一条件对应多个布局"的东西: 实测每 case 中位
    9 个不同几何, 其中中位 4 个是非支配的。

    返回顺序按几何签名排序, 保证与并行/分片无关地确定 (行 id 由它派生, 必须稳定)。
    """
    best: Dict[Tuple, L.Candidate] = {}
    for c in bank:
        key = _geom_key(c)
        cur = best.get(key)
        if cur is None or c.objective[0] < cur.objective[0]:
            best[key] = c
    return [best[k] for k in sorted(best, key=repr)]


# --------------------------------------------------------------------------------------
# 并行: 有池用池, 无池串行
# --------------------------------------------------------------------------------------
def iter_banks(items, pool, workers: int, chunksize: int = 16):
    if pool is None or workers <= 1:
        for it in items:
            yield _bank_job(it)
        return
    for out in pool.imap(_bank_job, items, chunksize=chunksize):
        yield out


# --------------------------------------------------------------------------------------
# 中间产物 (可续跑)
# --------------------------------------------------------------------------------------
def _load_parts(parts_dir: Path):
    done, bufs = set(), {"main": {}, "front": {}, "meta": {}, "l0": {}}
    for f in sorted(parts_dir.glob("part_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        done.update(d["sids"])
        for key in bufs:
            bufs[key].update(d[key])
    return done, bufs


def _chunk_of(new_id: int) -> int:
    return (int(new_id) - 1) // L.CHUNK + 1


def assemble(args, parts_dir: Path, out_dir: Path, l0_out: Optional[Path], meta: dict) -> None:
    """把 part 文件合并成主语料 chunk (每 case 多行) + 完整前沿 sidecar + L0 对照 + groups.json。

    行 id 的分配是这个函数的核心: 主语料一个 case 有多行 (每个不同几何一行), 而
    `PlacementManifestDataset` 要求 key 是 `system_{稠密整数}` 且 chunk 号 = (id-1)//5000+1。
    所以这里按 **case 升序、几何签名序** 给所有行统一编号 1..M (确定、与并行分片无关), 并写出
    `groups.json` 记录 行 -> case。切分工具据此按 case 分组, 避免同一 case 的多个布局
    被切到 train/val 两侧造成泄漏。
    """
    parts = [json.loads(f.read_text(encoding="utf-8"))
             for f in sorted(parts_dir.glob("part_*.json"))]

    # ---- 行 id 规划: case 升序, 每个 case 的行连续 ----
    counts: Dict[int, int] = {}
    for d in parts:
        for nid_s, entries in d["main"].items():
            counts[int(nid_s)] = len(entries)
    start, rid = {}, 0
    for nid in sorted(counts):
        start[nid] = rid + 1
        rid += counts[nid]
    n_rows = rid
    n_chunks = max(1, (n_rows - 1) // 5000 + 1)

    # ---- 主语料: 按行 id 落进对应的 chunk ----
    for ci in range(1, n_chunks + 1):
        lo, hi = (ci - 1) * 5000 + 1, ci * 5000
        bucket: Dict[str, dict] = {}
        for d in parts:
            for nid_s, entries in d["main"].items():
                s = start[int(nid_s)]
                for k, e in enumerate(entries):
                    r = s + k
                    if lo <= r <= hi:
                        bucket[f"system_{r}"] = e["record"] | {"system_id": f"system_{r}"}
        L.write_json_atomic(out_dir / f"chiplet_dataset_{ci}.json", bucket)
        print(f"[pareto] 写出 {out_dir / f'chiplet_dataset_{ci}.json'} ({len(bucket)} 行)", flush=True)

    # ---- groups.json: 行 -> case, 供切分工具按 case 分组 ----
    row_to_case = {r: nid for nid, s in start.items() for r in range(s, s + counts[nid])}
    L.write_json_atomic(out_dir / "groups.json", {
        "schema_version": 1,
        "n_rows": n_rows, "n_cases": len(counts),
        "note": "主语料一个 case 多行 (每个不同几何一行); 切分必须按 case 分组, 否则同一"
                " case 的不同布局会同时进 train 和 val, 验证指标虚高。",
        "row_to_case": {str(r): c for r, c in sorted(row_to_case.items())},
    }, indent=0)
    print(f"[pareto] 写出 {out_dir / 'groups.json'} ({n_rows} 行 / {len(counts)} case)", flush=True)

    # ---- L0 对照: 仍是每 case 一行, 沿用 case id ----
    if l0_out is not None:
        n_l0_chunks = _chunk_of(meta["n_systems"])
        for ci in range(1, n_l0_chunks + 1):
            bucket = {}
            for d in parts:
                for nid, rec in d["l0"].items():
                    if _chunk_of(nid) == ci:
                        bucket[f"system_{nid}"] = rec
            L.write_json_atomic(Path(l0_out) / f"chiplet_dataset_{ci}.json", bucket)
            print(f"[pareto] 写出 {Path(l0_out) / f'chiplet_dataset_{ci}.json'} ({len(bucket)} systems)",
                  flush=True)

    # ---- 完整前沿 sidecar (每 case 一行, key 用 case id) ----
    front_dir = out_dir / "front"
    n_case_chunks = _chunk_of(meta["n_systems"])
    for ci in range(1, n_case_chunks + 1):
        bucket, fmeta = {}, {}
        for d in parts:
            for nid, entries in d["front"].items():
                if _chunk_of(nid) != ci:
                    continue
                for k, rec in enumerate(entries):
                    bucket[f"system_{nid}__c{k}"] = rec
            for nid, m in d["meta"].items():
                if _chunk_of(nid) == ci:
                    fmeta[f"system_{nid}"] = m
        L.write_json_atomic(front_dir / f"chiplet_dataset_{ci}.json", bucket)
        L.write_json_atomic(front_dir / f"meta_{ci}.json", fmeta, indent=1)
        print(f"[pareto] 写出 {front_dir / f'chiplet_dataset_{ci}.json'} ({len(bucket)} 前沿成员)",
              flush=True)

    meta["n_rows"] = n_rows
    meta["n_chunks"] = n_chunks
    L.write_json_atomic(out_dir / "provenance.json", meta, indent=2)
    print(f"[pareto] 写出 {out_dir / 'provenance.json'}", flush=True)


def _rid_to_case(start: Dict[int, int], counts: Dict[int, int]) -> Dict[int, int]:
    """{行 id -> case id}。行按 case 升序连续编号。"""
    return {r: nid for nid, s in start.items() for r in range(s, s + counts[nid])}


# --------------------------------------------------------------------------------------
def run(args) -> dict:
    src_dir, l0_dir = Path(args.src_dir), Path(args.l0_dir)
    out_dir = Path(args.out_dir)
    l0_out = Path(args.l0_out_dir) if args.l0_out_dir else None
    parts = out_dir / ".parts"
    parts.mkdir(parents=True, exist_ok=True)

    all_items: List[Tuple[str, dict]] = []
    for k in args.chunk:
        src = L.load_chunk(src_dir / f"chiplet_dataset_{k}.json")
        l0 = L.load_chunk(l0_dir / f"chiplet_dataset_{k}.json")
        for sid in L.sorted_sids(src):
            if sid in l0:
                all_items.append((sid, l0[sid]))
    if not all_items:
        raise SystemExit("没有任何 (源, L0) 配对")
    src_ids = [int(s.split("_")[1]) for s, _ in all_items]
    base = min(src_ids) - 1
    print(f"[pareto] {len(all_items)} systems, 源 id {min(src_ids)}..{max(src_ids)}, "
          f"重编号 -{base} -> 1..{len(all_items)}", flush=True)

    _BANK_CFG.update(seed=args.seed, qap_tries=args.qap_tries, qap_restarts=args.qap_restarts,
                     power_random=args.power_random, geom=not args.no_geom)

    # resume: 已完成的 system 直接跳过
    done, bufs = _load_parts(parts) if args.resume else (set(), None)
    if done:
        print(f"[pareto] 续跑: 已有 {len(done)} 个 system 完成", flush=True)
    bufs = bufs if bufs is not None else {"main": {}, "front": {}, "meta": {}, "l0": {}}
    n_parts = len(list(parts.glob("part_*.json")))

    todo = [(s, r) for s, r in all_items if s not in done]
    if args.start:
        todo = todo[args.start:]
    if args.limit:
        todo = todo[: args.limit]
    rec_by_sid = dict(all_items)
    print(f"[pareto] 本次处理 {len(todo)} 个 system", flush=True)

    # 池必须在 CUDA 初始化之前 fork (worker 不需要 CUDA)
    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None
    scorer = L.SurrogateScorer(device=args.device, thermal_batch=args.thermal_batch)

    stats = defaultdict(float)
    deltas: Dict[str, List[float]] = {"max_c": [], "wl": [], "area": []}
    t0 = time.time()
    n_done = 0
    buf_sids: List[str] = []

    def flush_part():
        nonlocal n_parts, buf_sids
        if not buf_sids:
            return
        L.write_json_atomic(parts / f"part_{n_parts:05d}.json",
                            {"sids": buf_sids, **{k: v for k, v in bufs.items()}})
        n_parts += 1
        for v in bufs.values():
            v.clear()
        buf_sids = []

    try:
        for sid, bank, err in iter_banks(todo, pool, args.workers):
            if err is not None:
                stats[f"drop_{err}"] += 1
                continue
            core = L.load_core(sid, rec_by_sid[sid])
            if core is None:
                stats["drop_core"] += 1
                continue
            res = process_system(core, bank, scorer, args)
            if res is None:
                stats["drop_infeasible"] += 1
                continue
            nid = int(sid.split("_")[1]) - base
            # 主料一个 case 多行 (每个不同几何一行); 行 id 在 assemble 时按 case 顺序统一分配,
            # 所以这里先不带 system_id。
            bufs["main"][str(nid)] = [{"record": c.record, "tag": c.tag} for c in res["selected"]]
            bufs["front"][str(nid)] = [c.record for c in res["front"]]
            bufs["l0"][str(nid)] = core.ref | {"system_id": f"system_{nid}"}
            bufs["meta"][str(nid)] = {
                "src_id": sid, "bank": res["bank_size"], "front": res["front_size"],
                "rows": len(res["selected"]),
                "tags": sorted({c.tag for c in res["front"]}),
            }
            buf_sids.append(sid)
            n_done += 1
            stats["bank"] += res["bank_size"]
            stats["front"] += res["front_size"]
            if res["c0_obj"] and res["headline"].objective:
                c0o, selo = res["c0_obj"], res["headline"].objective
                # 与 pareto_front 同一套量化分辨率: 小于一个分辨率的差就是噪声, 不算改善
                q = L.obj_quantum([c0o, selo])
                q0, q1 = L.quantize(c0o, q), L.quantize(selo, q)
                n_better = 0
                for i, name in enumerate(("max_c", "wl", "area")):
                    d = selo[i] - c0o[i]
                    deltas[name].append(d)
                    stats[f"d_{name}"] += d
                    if q1[i] < q0[i]:
                        stats[f"better_{name}"] += 1
                        n_better += 1
                    elif q1[i] > q0[i]:
                        stats[f"worse_{name}"] += 1
                if n_better >= 2:
                    stats["better_2axis"] += 1
                if n_better >= 1:
                    stats["better_1axis"] += 1
            if n_done % args.subchunk == 0:
                flush_part()
            if n_done % args.print_every == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                print(f"[pareto] {n_done}/{len(todo)}  {el:.0f}s  {rate:.1f} sys/s  "
                      f"ETA {(len(todo) - n_done) / max(rate, 1e-9) / 60:.1f} min  "
                      f"库均 {stats['bank'] / n_done:.1f}  前沿均 {stats['front'] / n_done:.1f}",
                      flush=True)
    finally:
        flush_part()
        if pool is not None:
            pool.close()
            pool.join()

    n_total = len(all_items)
    if n_done == 0 and not done:
        raise SystemExit("一个 system 都没处理成功")

    print(f"\n[pareto] 完成 {n_done} 个 (累计 {len(done) + n_done}/{n_total}), "
          f"用时 {(time.time() - t0) / 60:.1f} min")
    n = max(n_done, 1)
    if stats["bank"]:
        print(f"  候选库 {stats['bank'] / n:.1f}/system   前沿 {stats['front'] / n:.1f}/system")
        for name in ("max_c", "wl", "area"):
            d = deltas[name]
            med = st.median(d) if d else float("nan")
            print(f"  Δ{name}: 中位 {med:+.3f}  均值 {stats[f'd_{name}'] / n:+.3f}  "
                  f"改善 {stats[f'better_{name}']:.0f} 变差 {stats[f'worse_{name}']:.0f} "
                  f"({100 * stats[f'better_{name}'] / n:.1f}%)")
        print(f"  改善 ≥1 轴: {stats['better_1axis']:.0f}/{n} ({100 * stats['better_1axis'] / n:.1f}%)"
              f"   ≥2 轴: {stats['better_2axis']:.0f}/{n} ({100 * stats['better_2axis'] / n:.1f}%)")
        print(f"  S3 闸门: 中位 Δmax_c ≤ 0 ? {st.median(deltas['max_c']) <= 0 if deltas['max_c'] else '?'}"
              f"   中位 Δwl ≤ 0 ? {st.median(deltas['wl']) <= 0 if deltas['wl'] else '?'}"
              f"   双轴 ≥ 50% ? {stats['better_2axis'] / n >= 0.5}")
        for k in sorted(k for k in stats if k.startswith("drop_")):
            print(f"  丢弃 {k[5:]}: {stats[k]:.0f}")

    if args.dry_run:
        print("[pareto] --dry-run: 不落盘")
        return dict(stats)

    if len(done) + n_done < n_total and not args.force_assemble:
        print(f"[pareto] ⚠ 只完成 {len(done) + n_done}/{n_total}, 跳过合并。"
              f"补齐后重跑即可续跑合并 (或用 --force-assemble 强行合并当前已有的)。")
        return dict(stats)

    prov = {
        "schema_version": 1,
        "pipeline": "gen_pareto_dataset.py (L1 候选库 + L2 代理批量打分 + L3 HRNet 贪心功耗搜索)",
        "source": {"dir": str(src_dir), "chunks": list(args.chunk),
                   "src_ids": [min(src_ids), max(src_ids)],
                   "renumber_offset": base,
                   "l0_dir": str(l0_dir)},
        "n_systems": n_total,
        "seed": args.seed,
        "params": {
            "qap_tries": args.qap_tries, "qap_restarts": args.qap_restarts,
            "power_random": args.power_random, "geom": not args.no_geom,
            "greedy_rounds": args.greedy_rounds, "greedy_starts": args.greedy_starts,
            "front_select": args.front_select, "mix_fraction": args.mix_fraction,
            "corpus_mode": args.corpus_mode,
            "greedy_min_gain": args.greedy_min_gain,
        },
        "objectives": ["therm_max_c (GNNHRNet)", "wirelength (WirelengthGNN, 规范功耗口径)",
                       "bbox_area (解析)"],
        "note": "id 已重编号为 1..N; 老 id = 新 id + renumber_offset。"
                "front/ 下是每个 system 的完整帕累托前沿。",
    }
    assemble(args, parts, out_dir, l0_out, prov)
    return dict(stats)


def all_items_dict_get(items, sid):
    for s, r in items:
        if s == sid:
            return r
    raise KeyError(sid)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--src-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_tw")
    ap.add_argument("--l0-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/_l0_raw")
    ap.add_argument("--out-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_opt_pareto")
    ap.add_argument("--l0-out-dir", type=Path,
                    default=L.PROJECT / "Dataset/dataset/placement_dataset/placement_dataset_l0")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个 system (0=全部)")
    ap.add_argument("--start", type=int, default=0, help="从第 N 个 system 开始 (0=开头)")
    ap.add_argument("--subchunk", type=int, default=500,
                    help="每处理这么多 system 落一次盘 (崩溃最多丢这么多)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="忽略已有 part 文件, 从头跑")
    ap.add_argument("--seed", type=int, default=20240913)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=16, help="CPU 候选库构造进程数 (1=串行)")
    ap.add_argument("--thermal-batch", type=int, default=32)
    # 候选库
    ap.add_argument("--qap-tries", type=int, default=32)
    ap.add_argument("--qap-restarts", type=int, default=8)
    ap.add_argument("--power-random", type=int, default=26)
    ap.add_argument("--no-geom", action="store_true", help="关掉 G3 等距变换")
    # L3
    ap.add_argument("--greedy-rounds", type=int, default=4)
    ap.add_argument("--greedy-starts", type=int, default=1)
    ap.add_argument("--greedy-min-gain", type=float, default=L.MAXC_QUANTUM,
                    help="贪心接受一次交换所需的最小热峰值改善 (°C); 必须 > 代理噪声底 ~1e-3")
    ap.add_argument("--print-every", type=int, default=100,
                    help="每多少个 system 打一行进度 (与 --subchunk 的落盘粒度解耦)")
    # 选片
    ap.add_argument("--corpus-mode", choices=["per_geometry", "single"], default="per_geometry",
                    help="per_geometry (默认): 一个 case 写多行, 每个不同几何一行 (行内取热最优功耗)。"
                         "single: 一个 case 一行, 由 --front-select 决定 (老行为)。")
    ap.add_argument("--front-select", choices=["knee", "thermal", "random"], default="thermal",
                    help="默认 thermal: 实测热最优成员在 19/20 的 system 上是纯功耗置换, "
                         "线长与面积与 C0 完全相同 (规范功耗口径下功耗置换对线长中性), "
                         "于是它**弱支配 C0** 且中位改善 -12.4°C。knee 会按前沿自身的值域"
                         "归一化, 把 40°C 的热轴和百分之几的线长轴等同看待, 选出的点退回 C0 附近。")
    ap.add_argument("--mix-fraction", type=float, default=0.2,
                    help="以该概率从前沿里随机取一个成员 (保布局多样性), 否则取选点")
    ap.add_argument("--dry-run", action="store_true", help="只统计, 不落盘")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
