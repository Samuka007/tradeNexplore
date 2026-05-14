# 实验项目索引

## 核心实验

| # | 实验 | 核心发现 | 可信度 |
|---|------|----------|--------|
| 01 | [消融实验](../01-ablation/report_zh_updated.md) | 制度变迁 > 交易成本 > 过拟合 | ✅ 高 |
| 02 | [Walk-forward](../02-walk_forward/report_zh.md) | 5 窗口，40% 胜率 | ✅ 高 |
| 03 | [鲁棒优化](../03-robust_opt/report_zh.md) | PSO 38.5% 胜率，GP 退化 | ✅ 高 |
| 04 | [结构复杂度](../04-comparison/report_zh.md) | Classic 50/200 仅交易 1 次 | ✅ 高 |
| 05 | [MoE](../05-moe/report_zh.md) | MoE 有害 | ✅ 高 |
| 06 | [仓位控制](../06-position/report_zh.md) | 连续信号 = BH | ✅ 高 |
| 07 | [PSO 粒子/迭代](../07-pso-tradeoff/report_zh.md) | ≥30 粒子稳定 | ✅ 高 |
| 08 | [PSO 惯性](../08-pso-inertia/report_zh.md) | 三种策略无差异 | ✅ 高 |
| 09 | [GP 惩罚](../09-gp-parsimony/report_zh.md) | ~~λ=1000 甜蜜点~~ → **λ=500（exp17 推翻）** | ⚠️ 已修正 |
| 10 | [仓位 scale](../10-position-scale/report_zh.md) | 0.1-10 最优 | ✅ 高 |
| 11 | [GP 种群/代数](../11-gp-tradeoff/report_zh.md) | 75×20 偶然 $3,143 | ✅ 高 |
| 12 | [GP 函数集](../12-gp-functionset/report_zh.md) | extended > minimal > original | ✅ 高 |
| 13 | [Walk-forward PSO](../13-walkforward-pso/report_zh.md) | 平均 $1,636 < single $2,296 | ✅ 高 |
| 14 | [GP+PSO Hybrid](../14-gp-pso-hybrid/report_zh.md) | 极简树 +7%，复杂树 -51% | ✅ 高 |
| 15 | [GP Warm-start](../15-gp-warmstart/report_zh.md) | +40%，6/10 击败 BH | ✅ 高 |
| 16 | [GP+PSO λ 扫描](../16-gp-pso-lambda-sweep/report_zh.md) | λ=500 GP-alone 最佳 | ✅ 高 |
| **17** | **[GP 系统网格搜索](../17-systematic-hyperparam/report_zh.md)** | **λ=500 真实甜蜜点（42 runs）** | ✅ **最高** |

## 综合报告

| 文档 | 说明 |
|------|------|
| [FINAL_REPORT.md](FINAL_REPORT.md) | **唯一综合报告**。基于最终数据，修正所有矛盾，每项结论标注 grounding |
| [METHODOLOGY.md](METHODOLOGY.md) | 方法论文档（参数、函数集、评估协议） |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | 已知 bug 与修复记录 |
| [FUTURE_WORK.md](FUTURE_WORK.md) | 未来改进方向 |

## 补充分析

| 分析 | 文档 | 数据 |
|------|------|------|
| 多种子验证 | — | [seed_robustness.json](../analysis/seed_robustness/seed_robustness.json) |
| 手续费敏感性 | [fee_sensitivity.md](../analysis/fee_sensitivity/fee_sensitivity.md) | [fee_sensitivity.json](../analysis/fee_sensitivity/fee_sensitivity.json) |
| 市场阶段 | [market_regimes.md](../analysis/market_regimes/market_regimes.md) | [market_regimes.json](../analysis/market_regimes/market_regimes.json) |
| PSO 收敛 | [pso_convergence.md](../analysis/pso_convergence/pso_convergence.md) | [pso_convergence.json](../analysis/pso_convergence/pso_convergence.json) |
| GP 预算敏感 | [gp_budget_test.md](../analysis/gp_budget/gp_budget_test.md) | [gp_budget_test.json](../analysis/gp_budget/gp_budget_test.json) |

## 代码修复

1. Walk-forward stepping bug
2. Backtest day 0 block
3. Backtest Infinity overflow
