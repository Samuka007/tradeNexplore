# 实验项目索引

## 核心实验（13 个）

| 编号 | 名称 | 报告 | 结果 | 核心发现 |
|------|------|------|------|----------|
| 01 | 消融实验 | [report_zh_updated.md](01-ablation/report_zh_updated.md) | [experiment_results.json](01-ablation/experiment_results.json) | 制度变迁 > 交易成本 > 过拟合 |
| 02 | Walk-forward | [report_zh.md](02-walk_forward/report_zh.md) | [results.json](02-walk_forward/results.json) | dual_crossover 在短窗口几乎无交易 |
| 03 | 鲁棒优化 | [report_zh.md](03-robust_opt/report_zh.md) | [results.json](03-robust_opt/results.json) | GP 退化为不交易；PSO 38.5% 胜率 |
| 04 | 结构复杂度 | [report_zh.md](04-comparison/report_zh.md) | [results.json](04-comparison/results.json) | Classic 50/200 仅交易 1 次即最优 |
| 05 | MoE | [report_zh.md](05-moe/report_zh.md) | [results.json](05-moe/results.json) | MoE 有害（$478 < $606） |
| 06 | 仓位控制 | [report_zh.md](06-position/report_zh.md) | — | 连续信号首次与 BH 持平 |
| 07 | PSO 粒子/迭代 | [report_zh.md](07-pso-tradeoff/report_zh.md) | [results.json](07-pso-tradeoff/results.json) | ≥30 粒子均收敛到 $2,296 |
| 08 | PSO 惯性 | [report_zh.md](08-pso-inertia/report_zh.md) | [results.json](08-pso-inertia/results.json) | 三种策略无差异 |
| 09 | GP 惩罚 | [report_zh.md](09-gp-parsimony/report_zh.md) | [results.json](09-gp-parsimony/results.json) | λ=1000 首次击败 BH（单种子） |
| 10 | 仓位 scale | [report_zh.md](10-position-scale/report_zh.md) | [results.json](10-position-scale/results.json) | 0.1-10 最优 |
| 11 | GP 种群/代数 | [report_zh.md](11-gp-tradeoff/report_zh.md) | [results.json](11-gp-tradeoff/results.json) | 75×20: seed=42 偶然 $3,143 |
| 12 | GP 函数集 | [report_zh.md](12-gp-functionset/report_zh.md) | [results.json](12-gp-functionset/results.json) | extended > minimal > original |
| 13 | Walk-forward PSO | [report_zh.md](13-walkforward-pso/report_zh.md) | [results.json](13-walkforward-pso/results.json) | 5 窗口，40% 胜率 |

## 补充分析

| 分析 | 文档 | 数据 | 核心发现 |
|------|------|------|----------|
| 多种子验证 | — | [seed_robustness.json](seed_robustness.json) | PSO 10/10 > BH, GP 0/10 < BH |
| 手续费敏感性 | [fee_sensitivity.md](fee_sensitivity.md) | [fee_sensitivity.json](fee_sensitivity.json) | 盈亏平衡点 4.4% |
| 市场阶段 | [market_regimes.md](market_regimes.md) | [market_regimes.json](market_regimes.json) | 无策略统治所有阶段 |
| PSO 收敛 | [pso_convergence.md](pso_convergence.md) | [pso_convergence.json](pso_convergence.json) | Basin A 深+慢 vs B 浅+快 |
| Basin 分析 | — | [basin_analysis.json](basin_analysis.json) | 少交易 = 高收益 |
| 2D Landscape | — | [landscape_2d.json](landscape_2d.json) | Basin A (120,180) vs B (35,100) |
| Equity Curves | — | [equity_curves.json](equity_curves.json) | Classic 50/200 仅交易 1 次 |

## 综合文档

| 文档 | 说明 |
|------|------|
| [FINAL_REPORT.md](FINAL_REPORT.md) | 项目综合报告（中文） |
| [NOTES.md](NOTES.md) | 实验洞察记录（写作素材） |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已知问题与修复记录 |
| [design/report_zh.md](design/report_zh.md) | 方法论概述 |

## 代码修复记录

1. **Walk-forward stepping bug**: `start = train_end` → `start = start + pd.DateOffset(years=test_years)`
2. **Backtest day 0 block**: 移除 `and i > 0`
3. **Backtest Infinity overflow**: 添加 `min(..., 1e12)` 上限
