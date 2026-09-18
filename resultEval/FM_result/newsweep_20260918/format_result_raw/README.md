# format_result_raw —— 合法化**之前**的采样结果

这批文件是 FM 采样器**未经任何合法化**的原始输出，转成 placement_dataset 格式（与 `../format_result/`
同一套格式：chiplet 的 x/y/width/height/rotation/power + 从 benchmark 回填的 hubump），用于
"合法化消融"（合法化前 vs 后的真机指标对比）。

- 命名：`<case>_t<t>-w<w>_raw_seed<seed>.json`
  - `t` = 热引导权重，`w` = 线长引导权重（`.` 写成 `p`，如 `t0p01-w2`）
  - 中间的 `_raw` 用来与合法化版本 `<case>_t<t>-w<w>_seed<seed>.json` 区分（两者键完全相同，可直接配对）
- 规模：1041 个解 = 12 个 case（acend910 / cpu-dram / hp11_m / hp6_m / hp8_m / multigpu / syn1 / syn4 /
  xerox6_m / xerox7_m / xerox8_m / Case7）× 各自权重网格 × 3 个固定 seed
- 对应的合法化结果：`../format_result/`（1030 个，剔除了合法化后仍非法的 11 个），真机指标见 `../result.csv`

合法性（真机口径，manifest 统计）：

| | 完全合法 | 均值 legality_2 |
|---|---|---|
| 合法化前（本目录） | 75 / 1041 = **7.2%** | 0.858（最差 0.000） |
| 合法化后（`../format_result/`） | 1030 / 1041 = **98.9%** | — |

生成脚本：`LayoutGenModel/logs/output/cases_hubump/newsweep_raw/` 由
`.tmp_raw_convert.py`（复用 `resultEval/convert_FM_layout_to_case.convert_one` 做逐 chiplet 校验与 hubump 回填）
从两个批次 manifest 转换而来；原始采样文件在
`LayoutGenModel/logs/output/cases_hubump/selected-wsweep-*/seed_*/placement/`。
