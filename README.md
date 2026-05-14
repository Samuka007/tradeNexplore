# 基于 PSO 与 GP 的 BTC 交易策略优化研究

## 项目概述

本项目系统研究了粒子群优化（PSO）与遗传规划（GP）在 BTC 交易策略优化中的表现，重点在于**理解优化算法本身的行为**，而非追求绝对收益。

**课程**: CITS4404 — 人工智能与自适应系统

## 核心发现

1. **PSO 显著优于 GP**：PSO 10/10 种子击败 Buy-and-Hold（平均 $2,297），GP 0/10（平均 $1,594）
2. **制度变迁是策略失败的主因**：反转时间顺序带来 25× 收益提升
3. **交易频率是隐藏的决定性因素**：Classic 50/200 SMA 仅交易 1 次即最优
4. **不存在全能策略**：不同市场阶段需要不同策略
5. **手续费是主要障碍**：盈亏平衡点约 4.4%
6. **Warm-start 使 GP 提升 40%**：注入人类规则后 GP 平均 $2,362，首次接近 PSO

## 项目结构

```
exp/                        # 实验目录
├── 01-ablation/            # 消融实验
├── 02-walk_forward/        # Walk-forward 验证
├── 03-robust_opt/          # 鲁棒优化
├── 04-comparison/          # 结构复杂度与 GP 函数集
├── 05-moe/                 # Mixture-of-Experts
├── 06-position/            # 仓位控制
├── 07-pso-tradeoff/        # PSO 粒子/迭代 trade-off
├── 08-pso-inertia/         # PSO 惯性策略
├── 09-gp-parsimony/        # GP 惩罚系数敏感性
├── 10-position-scale/      # 仓位 scale 敏感性
├── 11-gp-tradeoff/         # GP 种群/代数 trade-off
├── 12-gp-functionset/      # GP 函数集对比
├── 13-walkforward-pso/     # Walk-forward vs 单次切分
├── 14-gp-pso-hybrid/       # GP 结构 + PSO 参数
├── 15-gp-warmstart/        # GP Warm-start 初始化
├── 16-gp-pso-lambda-sweep/ # GP+PSO 的 λ 扫描
├── analysis/               # 补充分析
│   ├── seed_robustness/    # 多种子验证
│   ├── fee_sensitivity/    # 手续费敏感性
│   ├── market_regimes/     # 市场阶段分析
│   ├── pso_convergence/    # PSO 收敛分析
│   ├── gp_budget/          # GP 预算敏感性
│   └── landscape/          # 2D Landscape
├── reports/                # 综合报告
│   ├── FINAL_REPORT.md     # 完整项目报告
│   ├── EXECUTIVE_SUMMARY.md # 执行摘要
│   ├── RESULTS_TABLE.md    # 全实验结果汇总
│   ├── NOTES.md            # 实验洞察
│   ├── KNOWN_ISSUES.md     # 已知问题与修复
│   └── INDEX.md            # 项目索引
└── design/                 # 方法论设计

tiny_bot/                   # 核心代码包
├── backtest.py             # 回测引擎
├── strategy.py             # 策略定义
├── pso.py                  # PSO 实现
├── gp.py                   # GP 实现
├── filters.py              # 技术指标过滤器
├── data.py                 # 数据加载
└── __init__.py
```

## 快速导航

| 你想了解什么 | 阅读文档 |
|-------------|----------|
| 项目核心结论 | [exp/reports/EXECUTIVE_SUMMARY.md](exp/reports/EXECUTIVE_SUMMARY.md) |
| 完整分析报告 | [exp/reports/FINAL_REPORT.md](exp/reports/FINAL_REPORT.md) |
| 所有实验数据 | [exp/reports/RESULTS_TABLE.md](exp/reports/RESULTS_TABLE.md) |
| 实验目录索引 | [exp/reports/INDEX.md](exp/reports/INDEX.md) |

## 运行实验

```bash
# 使用虚拟环境
.venv/bin/python exp/07-pso-tradeoff/run.py

# 或安装依赖后
uv run python exp/07-pso-tradeoff/run.py
```

## 技术栈

- **语言**: Python 3.11
- **核心库**: numpy, pandas, yfinance
- **数据**: Yahoo Finance BTC-USD 日数据（2014-2022）

## 关键数据

### 多种子验证（10 seeds）

| 算法 | 平均测试 | 标准差 | 击败 BH |
|------|----------|--------|---------|
| **PSO** | **$2,297** | **$77** | **10/10** |
| **GP** | **$1,594** | **$432** | **0/10** |
| **GP Warm-start** | **$2,362** | **$894** | **6/10** |

### 最优策略对比

| 策略 | 测试收益 | 交易次数 |
|------|----------|----------|
| Classic 50/200 SMA | $2,236 | **1** |
| PSO position_sma | **$2,366** | 6 |
| Buy-and-Hold | $2,170 | 1 |

## 版本历史

57+ commits，包含 16 个实验、8 项补充分析、独立 reviewer 审查、3 个代码修复。
