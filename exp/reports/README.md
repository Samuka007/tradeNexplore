# 基于 PSO 与 GP 的 BTC 交易策略优化研究

## 项目概述

本项目系统研究了粒子群优化（PSO）与遗传规划（GP）在 BTC 交易策略优化中的表现，重点在于**理解优化算法本身的行为**，而非追求绝对收益。

## 核心发现

1. **PSO 显著优于 GP**：PSO 10/10 种子击败 Buy-and-Hold，GP 0/10
2. **制度变迁是策略失败的主因**：反转切分带来 25× 收益提升
3. **交易频率是隐藏的决定性因素**：Classic 50/200 仅交易 1 次即最优
4. **不存在全能策略**：不同市场阶段需要不同策略
5. **手续费是主要障碍**：盈亏平衡点约 4.4%

## 项目结构

```
exp/
├── 01-ablation/          # 消融实验（制度变迁、交易成本、过拟合）
├── 02-walk_forward/      # Walk-forward 验证
├── 03-robust_opt/        # 鲁棒优化
├── 04-comparison/        # 结构复杂度与 GP 函数集对比
├── 05-moe/               # Mixture-of-Experts
├── 06-position/          # 仓位控制（离散 vs 连续）
├── 07-pso-tradeoff/      # PSO 粒子/迭代 trade-off
├── 08-pso-inertia/       # PSO 惯性策略对比
├── 09-gp-parsimony/      # GP 惩罚系数与深度敏感性
├── 10-position-scale/    # 仓位 scale 敏感性
├── 11-gp-tradeoff/       # GP 种群/代数 trade-off
├── 12-gp-functionset/    # GP 函数集对比
├── 13-walkforward-pso/   # Walk-forward vs 单次切分 PSO
├── 14-gp-pso-hybrid/     # GP 结构 + PSO 参数优化
├── 15-gp-warmstart/      # GP Warm-start 初始化（运行中）
├── 16-gp-pso-lambda-sweep/ # GP+PSO 的 λ 扫描（运行中）
├── analysis/             # 补充分析
│   ├── seed_robustness/  # 多种子验证
│   ├── fee_sensitivity/  # 手续费敏感性
│   ├── market_regimes/   # 市场阶段分析
│   ├── pso_convergence/  # PSO 收敛分析
│   ├── gp_budget/        # GP 预算敏感性
│   └── landscape/        # 2D Landscape + Basin 分析
├── reports/              # 综合报告
│   ├── FINAL_REPORT.md   # 完整项目报告
│   ├── EXECUTIVE_SUMMARY.md # 执行摘要
│   ├── RESULTS_TABLE.md  # 全实验结果汇总表
│   ├── NOTES.md          # 实验洞察记录
│   ├── KNOWN_ISSUES.md   # 已知问题与修复
│   └── INDEX.md          # 项目索引
└── design/               # 方法论设计文档
```

## 快速导航

| 你想了解什么 | 阅读文档 |
|-------------|----------|
| 项目核心结论 | [EXECUTIVE_SUMMARY.md](EXECUTIVE_SUMMARY.md) |
| 完整分析报告 | [FINAL_REPORT.md](FINAL_REPORT.md) |
| 所有实验数据 | [RESULTS_TABLE.md](RESULTS_TABLE.md) |
| 实验目录索引 | [INDEX.md](INDEX.md) |
| 写作素材 | [NOTES.md](NOTES.md) |
| 代码修复记录 | [KNOWN_ISSUES.md](KNOWN_ISSUES.md) |

## 关键数据

### 多种子验证（10 seeds）

| 算法 | 平均测试 | 标准差 | 击败 BH |
|------|----------|--------|---------|
| **PSO** | **$2,297** | **$77** | **10/10** |
| **GP** | **$1,594** | **$432** | **0/10** |

### 最优策略对比

| 策略 | 测试收益 | 交易次数 |
|------|----------|----------|
| Classic 50/200 SMA | $2,236 | **1** |
| PSO position_sma | **$2,366** | 6 |
| Buy-and-Hold | $2,170 | 1 |

## 运行实验

```bash
# 单个实验
cd exp/07-pso-tradeoff
python run.py

# 使用虚拟环境
../../.venv/bin/python run.py
```

## 技术栈

- **语言**: Python 3.11
- **核心库**: numpy, pandas, yfinance
- **优化算法**: 自定义 PSO 和 GP 实现
- **数据**: Yahoo Finance BTC-USD 日数据（2014-2022）

## 课程信息

- **课程**: CITS4404
- **主题**: 进化计算在交易策略优化中的应用
- **重点**: 算法行为分析而非交易收益
