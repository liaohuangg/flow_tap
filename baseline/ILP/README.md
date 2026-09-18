# ILP 布局基线（`baseline/ILP/`）

以每个 chiplet 的**左下角坐标为决策变量**的 MILP 布局基线。同一套模型跑两个目标
（`--objective`），各自出一套结果：

| `--objective` | 目标 | 产物 |
|---|---|---|
| `wl` | **只有线长**（面积是软约束，不是目标项） | `resultEval/ILP_result/` |
| `bbox` | 外接框 **W+H**（权重调大）+ 长宽比 **\|W−H\|**（权重调小） | `resultEval/ILP_bbox_result/` |

`wl` 那条线用来给 AT / RL / FM 标定"线长理论上能做到多好"这个下界；`bbox` 那条线
标定"面积/形状理论上能做到多好"。

## 口径（三条，都是用户明确指定的）

1. **`wl` 目标函数只有线长**。面积**不**作为目标项出现。
2. **面积（画布上限）是软约束**：写成带 slack 的罚项，系数取 λ 量纲的 `1e-6`。
   塞得进画布时 slack 自动归零、退化成硬约束；塞不进时模型仍然可行，且罚项小到
   不会主导线长。实测 12 个 case 里 slack 基本都是 0（画布由 `2 × Σ footprint 面积`
   推出来，利用率只有 ~50%，本来就宽松）。
3. **不跑 Case6 / Case8 / Case9 / Case10**（20/36/44/61 芯粒，规模太大）。
   跑的是另外 12 个，最大的是 Case7（28 芯粒）。

## 口径（三条，都是用户明确指定的）

1. **目标函数只有线长**。面积**不**作为目标项出现。
2. **面积（画布上限）是软约束**：写成带 slack 的罚项，系数取 λ 量纲的 `1e-6`。
   塞得进画布时 slack 自动归零、退化成硬约束；塞不进时模型仍然可行，且罚项小到
   不会主导线长。实测 12 个 case 里 slack 基本都是 0（画布由 `2 × Σ footprint 面积`
   推出来，利用率只有 ~50%，本来就宽松）。
3. **不跑 Case6 / Case8 / Case9 / Case10**（20/36/44/61 芯粒，规模太大）。
   跑的是另外 12 个，最大的是 Case7（28 芯粒）。

## 模型

**变量**：`FX_i, FY_i` = footprint 左下角（连续，`[0, 2S]`）；`r_i` = 旋转二值
（宽高相同的芯粒固定为 0，省变量）；`AXE/AYE` = clump 形态间的 |Δx|/|Δy|（连续，≥0）；
`slack_x/y` = 画布越界量。

**约束**：
- 画布（软）：`FX_i + FW_i(r_i) ≤ S + slack_x`，y 同理。
- 不重叠：footprint 上做 4 二值 big-M（`pL+pR+pD+pU ≥ 1`，`M = 2S + max_fp_dim`），
  模式照搬 `MPDAP/src/ilp_method_EMIB_chiplet.py:1092-1137`。
- clump 距离：对每个连通对、每个形态组合，
  `AXE[a][b] ≥ ±(FX_i + base_a + slope_a·r_i - FX_j - base_b - slope_b·r_j)`。
- 对称破缺：布局可整体平移，故 w.l.o.g. `min FX = min FY = 0`（`Σ tX ≥ 1`）。
  这是 strip-packing 类模型里求解器卡住的最大单一原因。
- 无连接芯粒：加 `ε·(FX+FY)` 平局打破，防止哑元乱飘撑大 bbox。

**线长为什么能免 abs 二值**：clump 坐标是 footprint 左下角的**仿射函数**，偏移只由
`(width, height, hubump)` 这些数据常数决定（见 `ilp_core.xform_bases/yform_bases`），
所以 clump 间曼哈顿距离 = `|ΔFX + c| + |ΔFY + c'|`。每根轴上只有 **3 个不同形态**
（x: 左/上=下/右；y: 左=右/上/下），于是每对只要 9+9 个辅助变量而不是 16+16，
且 18 个目标系数全部严格为正 —— 最小化会自然把它们压到精确的绝对值。

**λ 怎么来的**：不是精确弧流量，是**容量感知的贪心分配**（`wl_oracle.arc_flow`）。
原因：参考模型是逐 net 的（`f[i][h][j][k][n]`），想拿精确 f 就得把那个模型整个重跑，
Case7 是 85 万变量、单次 25 秒，比参考本身还贵；而且连接矩阵对称，**net 之间不能简单
聚合**（聚合后每个芯粒净供给都是 0，守恒约束直接退化，得到一个远低于真实值的松弛解）。
贪心抓住了参考问题最关键的特征 —— **容量紧绑**（实测 hp6_m 上无容量约束最优 12938、
带容量 19218，差 48%）—— 且是 O(nets×16 log 16)，瞬时。

**外层逐次线性化**：`MILP(λ) → 用精确 reference_cost 评估 → λ ← damp·新弧流量 + (1-damp)·旧λ`，
重复 K 轮，按**真实线长**保留历史最优。注意 λ 只负责引导；
**报出去的所有线长都来自精确的 `reference_cost`**（和 `eval_layout.eval_wirelength`
同一条调用）。

## `bbox` 目标：W+H（主）+ |W−H|（次）

用户口径原话：*"只缩小约束面积 W+H（这个参数调大一些），并且添加长宽比约束 |W−H|
（这个调小一些），也就是在优化 W+H 的前提下，优化长宽比。"*

翻译成权重（`run_ilp_cases.OBJECTIVES["bbox"]`，都是除以画布 S 的无量纲量）：

```
obj = (1.0 / S)·(maxX + maxY)        ← W+H，主
    + (0.1 / S)·aspect               ← |W−H|，次 (square 配置取 0.5)
    + (0.01 / wl_ref)·wl             ← 极小的线长平局打破项
    + 1e-6·(slack_x + slack_y)/S     ← 画布软约束
```

`w_aspect` 从 0.1 提到 0.5 就是 `square` 配置，效果是拿一点 W+H 换方正度。
实测 xerox6_m：`bbox` 给 W+H 57.88 / \|W−H\| 6.90，`square` 给 58.58 / **1.26**。

**关键：这个目标是精确的，必须求到 MIPGap=0。**
`maxX / maxY / aspect` 都是模型里的**精确变量** —— W+H 和 \|W−H\| 就是字面量，
不存在线性化误差（唯一近似的是 `w_wl` 那个极小的线长平局项）。所以
`stage_plan(..., exact=True)` 把**末段 `MIPGap` 直接取 0**，产物带
`proven_optimal: true` 和 `best_bound` —— 这才配得上"面积能做到多好"这个下界。

> 曾经按 wl 目标那套分段阈值（0.10 / 0.02 / 0.005）跑，第 1 段就在 gap 8.9% 上停下，
> 报出 hp6_m 的 W+H **47.53**，而真最优是 **44.83**（gap=0，12 秒证明）。
> 那 47.53 不是"近似"，是**错**。分段截断在近似目标上是省时间，在精确目标上是自废武功。

前两段仍留宽松阈值（0.05 / 0.01）：大 case 上证明不动时，至少先抓到好可行解，
再由 `TimeLimit` 兜底，把**真实 gap 如实记进** raw / run_log。所以表里要能区分
"证明最优"和"时限到、gap=X%"——`proven_optimal` 字段就是干这个的。

`bbox` 实例的 5 个配置：

| seed | 名字 | 差别 |
|---|---|---|
| 1 | `bbox` | w_aspect=0.1，允许旋转 → **正文引用这个数** |
| 2 | `square` | w_aspect=0.5（更看重方正度） |
| 3 | `norot` | 禁用旋转 |
| 4 | `deep` | 2× 预算、gap_tight=0.4 |
| 5 | `nowl` | **w_wl=0**，纯 W+H+\|W−H\|，无任何线长项 → 最干净的面积下界 |

## 五个"种子"其实是五个确定性配置

MILP 是确定性的，报 5 个随机种子是假的。`seed=1..5` 是 5 个固定、写明的配置：

| seed | 名字 | 差别 |
|---|---|---|
| 1 | `canon` | λ 均匀初值，允许旋转，K=3，完整预算 → **正文引用这个数** |
| 2 | `flowinit` | λ 用 greedy 解的弧流量初始化，允许旋转，K=5 |
| 3 | `norot` | 禁用旋转（模型更小） |
| 4 | `deep` | 2× 预算，`MIPGap` 收紧到 0.4× |
| 5 | `damp` | 阻尼 0.5、K=8（外层多迭代） |

这样下游 `_split_stem` / `compare_*.csv` / `result_table/` 全部零改动就能接。
比较时建议同时给 `ILP_best`（5 个取最小）和 `ILP_seed均值`；对确定性方法，
`ILP_best` 才是它真正的能力上界。

## 预算

| case 规模 | quick | full |
|---|---|---|
| n ≤ 8 | 60 s | 300 s |
| 10 ≤ n ≤ 13 | 180 s | 900 s |
| n = 28（Case7） | 900 s | 5400 s |

**预算的表是整次求解的总量，不是每轮的量** —— 表里那个数会被 `budget_mult` 缩放，
再**按轮数 K 平分**给每一轮 MILP。

> 早先是每轮各拿一整份，于是 `seed5`（K=8）在 Case7/full 下是 `8 × 5400s = 12 小时`
> 一个任务，而且 5 个配置的总耗时差好几倍、互相根本没法比。现在 5 个配置拿同样的
> 总预算，区别只在**怎么花**。

`wl` 目标分段（`stage_plan(exact=False)`）：25/50/25，`MIPGap` 0.10 / 0.02 /
`0.005·gap_tight`，`MIPFocus` 1/2/3 —— 先找可行解、再平衡、最后压界。
`bbox` 目标分段（`exact=True`）：15/25/**60**，`MIPGap` 0.05 / 0.01 / **0** ——
最后一段拿六成预算去**证明最优**。

实际绝大多数小 case 秒级就 `OPTIMAL`，跑不到预算上限；大 case（Case7）会吃满。

## 跑

```bash
cd /root/placement/flow_tap
PY=/root/anaconda3/envs/chipdiffusion/bin/python

$PY baseline/ILP/ilp_core.py --selftest              # 1. 仿射常数自检 (必须过)
$PY baseline/ILP/run_ilp_cases.py --cases hp6_m --seeds 1 --profile quick --verbose
$PY baseline/ILP/check_objective.py                  # 2. 回归自检

# 3. 全量 (12 case x 5 配置) x 两个目标
$PY baseline/ILP/run_ilp_cases.py --all --seeds 1 2 3 4 5 --objective wl   --profile full --workers 10
$PY baseline/ILP/run_ilp_cases.py --all --seeds 1 2 3 4 5 --objective bbox --profile full --workers 6
# 或走 wrapper (会顺带跑 check_objective): PROFILE=full bash baseline/ILP/run_ilp_cases.sh

# 4. 评估 —— 每个目标都要清**两个**缓存!
for m in ILP ILP_bbox; do
  rm -f  resultEval/${m}_result/eval_cache.json
  rm -rf resultEval/${m}_result/eval_out
  $PY resultEval/eval_layout.py --method $m --workers 8
done
# -> resultEval/ILP_result/result.csv       (12 case x 5 = 60 行, 与 AT/FM 同 14 列)
# -> resultEval/ILP_bbox_result/result.csv  (同样 60 行)
```

## 产物

- `resultEval/ILP_result/format_result/<case>_seed<k>.json` —— **唯一交付物**，
  格式照抄 `resultEval/RL_result/format_result/acend910_seed2.json`
  （`system_id` / `chiplets`[`name, x-position, y-position, width, height, rotation,
  power, hubump`] / `connections`[`node1, node2, wireCount`]，`indent=2`）。
  `x-position` = **本体左下角** = footprint 左下角 + hubump；
  `width/height` 是**摆放后**尺寸（`rotation=1` 时已交换，写盘处有断言）。
- `baseline/ILP/raw/<case>_seed<k>.json` —— 旁证：模型规模、`mip_gap`、`best_bound`、
  **`proven_optimal`**、每轮的 `obj_model` vs `obj_eval`、画布越界量、bbox、`wall_s`。
- `baseline/ILP/run_log.csv` —— 每个 (case, 配置) 一行。

`bbox` 目标把上面三处换成 `resultEval/ILP_bbox_result/`、`raw_bbox/`、`run_log_bbox.csv`。
`run_log` 的列是写死的，所以 `mip_gap` / `proven_optimal` 只在 raw json 里
（加列会让已存在的文件表头与数据错位）。

下游没有任何东西读 `raw/` 和 `run_log.csv`。

## 陷坑

- **陈旧缓存（两个）**：`run_method` 的 `need(stem)` 在缓存行已有字段时返回 `False`，
  改写布局后**静默**沿用旧温度；`eval_thermal` 还会复用 `eval_out/<stem>/*.grid.steady`。
  → `run_ilp_cases.py` 写盘前会自动删这两个；手工重跑时务必自己清。
- **Gurobi 要先 `update()`** 才能查 `NumVars/NumBinVars/NumConstrs`，否则一律返回 0。
- **`get_input` 返回的 `xc/yc` 是相对 footprint 左下角的偏移**，绝对位置 = `xl + xc`
  （见 `gen_wirelength_dataset.py:274`）。`--selftest` 检查的就是这个关系。
- **旋转/宽高交换**：写盘时断言 `width == w + r(h-w)`。忘了交换会静默得到偏小的 bbox。
- **取整造成重叠**：`round(...,6)` 让每个坐标最多偏 5e-7，所以**两个贴边芯粒的重叠
  可达 1e-6**。`separate_rounding` 的容差必须大于这个数（现在是 `SEPARATE_TOL=1e-4`），
  沿重叠小的那根轴推开 `margin=1e-5`。
  早先用 `tol=5e-7` —— 比取整误差本身还小 —— 于是把这种**必然产物**判成"真重叠"、
  直接放弃整份解，静默丢掉求解器好不容易找到的更优布局。
- **`imap_unordered` 只传一个参数**：它不是 `starmap`，不会拆包。`_worker` 必须
  收 `job` 元组再自己解（`case, seed, profile, objective, verbose = job`）。
  签名对不上时，**第一个结果回来才炸**，而且炸在 pool 的迭代里 —— 前面已经算好、
  已经写盘的布局全部拿不到，`run_log.csv` 一行不留。改并行派发后务必先用
  `--workers 4` 跑个小批验证。
- **日志要 `flush=True`**：stdout 重定向到文件时是块缓冲，父子进程都不刷的话
  日志里只剩 Gurobi 许可横幅，看着像"跑得好好的"，其实是全憋在缓冲区。
- **零 hubump 的连通芯粒**：`compute_hubump` 对 s=0 返回 0 → `pmax=0` → 参考 ILP 不可行。
  `check_case` 会直接跳过这类 case 并记日志。
- **纯线长是帕累托的一个极端**：只压线长会让芯粒挤成一坨，`max_temp_C` 大概率是四个
  方法里最差的（AT 是热感知的）。这正是它的意义 —— 定义线长最优那个角。

## 一个值得注意的结果

hp6_m（6 芯粒）：MILP **6 秒解到 OPTIMAL，线长 2642.0 mm**，bbox 700.7 mm²。
同期 AT 在它自己的 50 点 `wl_weight` 扫描里最好只到 **12577.09 mm**（bbox 728.31 mm²）。

差距这么大不是 bug —— 交叉验证过：官方 `AT_result.csv` 里 `acend910,1` 是
`42319.936396`，本仓库的 `reference_cost` 重算同一份布局给 `42319.94`，逐位一致。
根因是**目标口径不同**：AT 优化的是中心距 HPWL，而 TAP-2.5D 的**布线**代价奖励的是
clump 物理相邻 —— 贴边的两个芯粒之间，相对的两个 clump 只隔 `2×hubump`（≈0.09 mm）。
我们的 MILP 直接在这个代价上优化，所以会主动去找紧密堆叠；HPWL 只是部分捕捉了这一点。
