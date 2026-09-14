# 帕累托布局数据集生成器 — 步骤文档

> 目的：把 `placement_dataset_tw` chunks 5–8 的 20000 个芯粒系统，转化成一份
> **热峰值更低、且同一个 case 有多个不同布局**的训练语料，同时**逐字保持抽象条件不变**。

---

## 0. 一句话

对每个 case，先造一个 ~41 条的**合法候选库**（每条都是一次完整的「几何 + 互联 + 功耗」实现），
用两个**冻结代理**给整库打分，再用 HRNet 做一轮功耗贪心搜索，最后**按不同几何各挑一个代表**写进主语料。

---

## 1. 输入 / 输出

| | |
|---|---|
| 源 | `Dataset/dataset/placement_dataset/placement_dataset_tw/chiplet_dataset_{5,6,7,8}.json`（id 20001–40000） |
| C0 锚 | `Dataset/dataset/placement_dataset/_l0_raw/chiplet_dataset_{5,6,7,8}.json`（对上面这 4 个 chunk 跑 `adp` 的产物，id 20001–40000） |
| 主语料 | `placement_dataset_opt_pareto/chiplet_dataset_{1..N}.json`，key `system_{稠密行号}`，**一个 case 多行** |
| 分组表 | 同目录 `groups.json`：`行号 -> case 号`，供切分工具按 case 分组 |
| 前沿 | 同目录 `front/`：每个 case 的**完整帕累托前沿** |
| L0 对照 | `placement_dataset_l0/chiplet_dataset_{1..4}.json`（每 case 一行，同重编号） |
| 溯源 | 同目录 `provenance.json` |

## 2. 冻结什么、自由什么

每个 case 看成「**抽象条件** + **一次实现**」：

- **冻结（I1–I4 硬断言，违反即丢弃并计数）**
  - `|V|`、footprint 多重集（W,H 为整数且 4≤W,H≤30）
  - 功耗多重集
  - 边的**无权多重集** + 边权多重集（即图同构类）
- **自由**
  - `σ`：图节点 → 槽位 的双射（重贴标签）
  - 功耗的槽位置换
  - 整幅铺排的 bbox 等距变换（镜像 x / 镜像 y / 转置）

**明确禁止**：归零/增删/缩放 wireCount；缩小或置换 footprint；改 `|V|`；整体缩放坐标；放宽合法性。

---

## 3. 七步流程

### 第 1 步：读 L0，确定 C0 锚 `[load_core]`

从源 chunk 和 `_l0_raw` 里各取同名 system，构造 `SystemCore`：

```
SystemCore: sid, names[]  n
            fp[]        槽位的 footprint (x, y, fw, fh)   —— 来自 L0 记录
            rotation[]  槽位朝向
            edges[]     (u, v, wireCount)  —— 来自 L0 记录，节点按名字索引
            powers[]    每颗芯粒的功耗  —— 来自 L0 记录
            ref         L0 原始记录（= C0）
```

**C0 = `core.ref` 逐字节照搬，永远在候选库里**。这是「输出按构造不会比现有语料更差」的保证。

### 第 2 步：造候选库 `[build_bank]`（CPU，16 workers）

| 组 | 怎么造 | 条数 | 动了什么 |
|---|---|---|---|
| **C0** | L0 记录逐字节 | 1 | 无 |
| **G1** | QAP 重解 `σ`：先按连通性贪心出一批初始 σ，再做 2-opt + 随机扰动，保留**互不相同**的最优解。每个 σ 都要重新派生互联线 | ~30 | 重贴标签 |
| **G2** | 功耗置换（见第 4 节） | ~32 | 只用功耗 |
| **G3** | 沿 bbox 中心镜像 x / 镜像 y / 转置 | 3 | 铺排等距变换 |

每条候选都走 `make_record(...)` 生成一条**合法且满足全部不变量**的记录；不可行
（`solve_hubump` 无解 / body 边长 ≤0）直接丢弃。

### 第 3 步：L2 批量打分 `[score_bank]`（GPU）

一个 case 的整库**一次批量**过两个代理。**为什么能批量**：同一 case 的所有候选共享
footprint 铺排 → 共享画布（画布边长只依赖 footprint），于是可以拼成一个 batch 一次前向。

- **热峰值 `max_c`**：GNNHRNet（`gnnhrnet_pwin/best.pth`，peak_window_w=0.4）
- **线长 `wl`**：WirelengthGNN（`best_wlmodel_total_60k.pt`），**用 C0 的功耗向量做排序口径**
- **bbox 面积**：解析计算，免费

> ⚠️ 线长必须用规范功耗口径。实测线长代理对功耗有虚假依赖（功耗置换会让预测动 0.15% 中位 /
> 1.08% 最大），而真值 CPLEX 线长**严格与功耗无关**。所以所有候选的线长都在 C0 的功耗向量下算。

### 第 4 步：L3 HRNet 贪心功耗搜索 `[greedy_power_search]`

从起点功耗向量出发，**best-improvement 成对交换**：

1. 枚举所有两两交换 `(i,j)`，得到 n(n−1)/2 个新功耗向量
2. **一次批量**前向 HRNet，拿到每个的热峰值
3. 取热峰值最低的那个；**只有当它比当前低超过 `--greedy-min-gain`（默认 0.01 °C）才接受**
4. 重复 ≤4 轮，或直到没有可接受的交换

> 门槛必须 > 代理的 run-to-run 噪声底（实测 ~1e-3 °C），否则贪心会追着 GPU 非确定性跑。
> 只收**最终结果**入库：搜索中途评估过的那些向量全部被最终结果支配（功耗置换对线长和面积
> 都中性，而最终结果是这批里 max_c 最小的），收进来只会白白多花一次批量前向（实测 ~27% 时间）。

### 第 5 步：帕累托前沿 `[pareto_front]`

三目标 `(max_c, wl, bbox_area)` 全部最小化。**比较前先按分辨率量化**：

```
热 1e-2 °C   线长 1e-5 相对   面积 1e-6 mm²     (都是实测噪声的 ~10 倍)
```

不量化的话，纯浮点噪声会造出大量假「互相不支配」的点 —— 实测 G3 镜像与 C0 的 bbox 面积
只差 4.5e-13（镜像本就保面积），精确比较会认为「C0 面积更大」，于是前沿里混进一个跟 C0 物理等价的点。

### 第 6 步：选片 `[select_per_geometry]`

**按「不同几何」分组，每组取热峰值最优的那个代表。**

「不同几何」= `(σ, footprint 铺排)` 这个签名 —— **功耗置换不算新几何**。于是：
- 同一 case 的几十个功耗排列 → 塌缩成 1 个代表（取热最优的）
- G1 的多个 QAP 最优、G3 的镜像/转置 → 各自成行

**实测每 case 中位 9 个不同几何**（库均 41 条，中位前沿 4 条）。主语料就写这 ~9 行。

### 第 7 步：落盘 `[assemble]`

行号按 **case 升序、几何签名序** 统一分配成 1..M（确定、与并行分片无关），
每 5000 行一个 `chiplet_dataset_{k}.json`，同时写 `groups.json`。

**切分必须按 case 分组**：`split_placement_dataset.py` 读到 `groups.json` 后按 case 洗牌切分，
再把行展开。否则同一个 case 的 9 个布局会同时进 train 和 val，验证集里出现与训练集几乎相同的
布局，**指标虚高**。

---

## 4. 功耗到底是怎么生成的（重点）

**功耗在整个流水线里只做一件事：在槽位之间置换。多重集 `core.powers` 从 L0 读入后永远不变。**

写进记录时：`make_record(core, power_slots=pw)` → `chiplets[i]["power"] = pw[i]`，
即「**槽位 i 吃多少功耗**」。

三种来源：

**(a) G2 确定性策略（6 条）** — `_power_orders(core)`：把槽位按某个 key 排序，
把功耗从大到小依次发下去（`values = sorted(powers, reverse=True)`）：

| tag | 排序 key | 含义 |
|---|---|---|
| `density` | 槽位面积 × 外围得分 | **= `adp` 的默认规则**（避免小芯片被塞大功耗） |
| `power` | 外围得分 | 大功耗排到外围 |
| `rev_density` | −面积×外围得分 | 取反 |
| `connectivity` | 该槽位 incident 线数之和 | **连得最多的吃最大功耗** |
| `connectivity_rev` | 取反 | |
| `area_asc` | −面积 | 大功耗给大芯片 |

**(b) G2 随机置换（26 条）** — `rng.shuffle(pw)`。

**(c) L3 HRNet 贪心（第 4 步）** — 从某个起点出发做 best-improvement 成对交换，
**目标函数是 HRNet 的热峰值**。

### 这里有一个关键事实

`adp`（也就是现有语料的构造方式）**本来就是**「几何 → 互联 → 功耗」：
- hubump 由 incident 线数之和推出（`solve_hubump(fw, fh, 2·Σ_incident w)`）
- 功耗由 `assign_power` 按**布局的外围得分**发下去

所以 `adp --power-mode density` 本身就是在**几何固定时重新分配功耗**，用的是一条手搓启发式
（「大功耗排外围」）。**我的 (a)/(b)/(c) 是同一个搜索空间，只是把那条手搓规则换成了训好的 HRNet。**

**hubump 与功耗无关**：`make_record` 里 `hu = solve_hubump(fp[i][2], fp[i][3], 2·s_sum[i])`，
`s_sum` 只由 σ 决定。所以功耗置换**不动几何、不动互联线、不动 footprint**。

---

## 5. 当前进度

| 步骤 | 状态 |
|---|---|
| S0 打分器 `score_dataset.py` | ✅ 完成 |
| S1 候选库 + 不变量 | ✅ 完成（6979 条候选零违反，C0 逐字节复现 200/200） |
| S2 批量热打分 | ✅ 完成（与逐条 maxdiff 0.0006 °C） |
| S3 试点 200 case | ✅ 通过（中位 Δmax_c −1.001 °C，改善 179/200） |
| S4 L3 贪心 | ✅ 中位 −1.0 °C 达标 |
| **多行主语料 + 分组切分** | 🔧 **刚改完，未验证** |
| S5 全量 20k 落盘 | ⏳ 待跑（按 1.0 sys/s 估 **~5.6 h**） |
| V1/V3/V5/V4 验证 | ⏳ 待跑（V3 是真值 HotSpot + CPLEX 的决定性闸门） |

### 已知短板（必须说清楚）

1. **几何轴没被真正优化过**。G1/G2 都只是在一个**固定的 footprint 铺排**上换标签，
   整个语料里每个 case 的**铺排**其实只有 C0 那一个（加镜像/转置）。
   所以语料说的是「给定几何，把功耗摆到热最优」，**不是**「给定功耗，生成好铺排」。
2. **评测方向可能是反的**。`benchmark/cases_hubump/Case*.json` 里 chiplet 只有
   `width/height/power/hubump/footprint_w/h`、**没有 x/y** —— 评测时功耗是条件、几何是要生成的结果。
   而本语料是「几何固定、功耗去适配几何」。
3. **面积轴是死的**。实测 200 个 case 里 0 个在面积上有改善（镜像/重贴标签都保面积）。
   前沿实际是二维 (T, WL)。
4. **功耗落在噪声里**。L3 的边际收益中位只有 −1.0 °C，而代理 run-to-run 噪声底 ~1e-3 °C，
   量级上安全但不大。
