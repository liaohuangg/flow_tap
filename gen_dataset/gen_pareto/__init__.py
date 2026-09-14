"""gen_pareto — 生成「热峰值 / 线长 / bbox 面积」帕累托布局数据集的全套代码。

文件一览
--------
  opt_dataset_lib.py        公共库: SystemCore / load_core / make_record /
                            assert_invariants (I1-I9) / SurrogateScorer / pareto_front 等。
  gen_pareto_dataset.py     旧 σ/ρ 置换版管道 (换标签 + 换功耗)。已被现行版取代, 仅保留
                            greedy_power_search (功耗置换, 线长中性的热杠杆) 供复用。
  gen_thermal_diverse.py    现行管道: 固定条件 (图 + 功耗) + 径向铺开 + 代理打分 + 帕累托。
                            可选 --greedy-rounds 打开功耗置换这条线长中性的热杠杆。
  geometry_gen.py           径向铺开几何生成器 (C0 绕质心等比放大, s 上限收在线长劣化内)。
  score_dataset.py          独立打分器: 任意数据集 -> per-system (max_c/wl/bbox_area/...),
                            支持 --compare 配对比较。
  probe_front.py            单 system 前沿可视化探针 (C0 + 铺开 + perm)。
  PARETO_PIPELINE.md        方案文档 (方案、实测事实、不变量、验证门槛)。
"""
