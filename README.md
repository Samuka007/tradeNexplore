# 基于 PSO 与 GP 的 BTC 交易策略优化研究

## 核心结论

1. **PSO 是稳健的优化器**：10/10 种子击败 Buy-and-Hold（平均 $2,297，标准差 $77）
2. **GP 原始表现不稳定**：10/10 种子未击败 BH（平均 $1,594），但 warm-start 使 GP 提升至 $2,362（6/10 击败 BH）
3. **λ=500 是 GP 真实甜蜜点**：系统网格搜索（42 runs, 3 seeds, 7 λ × 2 depths）证明
4. **单次切分高估性能 40%**：Walk-forward 平均 $1,636 < 单次切分 $2,296
5. **交易频率是决定性因素**：Classic 50/200 仅交易 1 次即获 $2,236
6. **PSO+GP Hybrid 无效**：分离优化仅在极简树上有效，复杂树上破坏性能

## 唯一综合报告

**[exp/reports/FINAL_REPORT.md](exp/reports/FINAL_REPORT.md)** — 基于最终数据，修正所有矛盾，每项结论标注 grounding 来源。

## 项目结构

```
exp/
├── 01-ablation/              # 消融实验
├── 02-walk_forward/          # Walk-forward 验证
├── 03-robust_opt/            # 鲁棒优化
├── 04-comparison/            # 结构复杂度
├── 05-moe/                   # Mixture-of-Experts
├── 06-position/              # 仓位控制
├── 07-pso-tradeoff/          # PSO 粒子/迭代 trade-off
├── 08-pso-inertia/           # PSO 惯性策略
├── 09-gp-parsimony/          # GP 惩罚敏感性
├── 10-position-scale/        # 仓位 scale
├── 11-gp-tradeoff/           # GP 种群/代数
├── 12-gp-functionset/        # GP 函数集
├── 13-walkforward-pso/       # Walk-forward vs 单次切分
├── 14-gp-pso-hybrid/         # GP 结构 + PSO 参数
├── 15-gp-warmstart/          # GP Warm-start
├── 16-gp-pso-lambda-sweep/   # GP+PSO λ 扫描
├── 17-systematic-hyperparam/ # GP 系统网格搜索
├── analysis/                 # 补充分析
│   ├── seed_robustness/
│   ├── fee_sensitivity/
│   ├── market_regimes/
│   ├── pso_convergence/
│   ├── gp_budget/
│   └── landscape/
├── reports/                  # 综合报告
│   ├── FINAL_REPORT.md       # ← 唯一综合报告
│   ├── INDEX.md              # 实验目录索引
│   ├── METHODOLOGY.md        # 方法论文档
│   ├── KNOWN_ISSUES.md       # 已知 bug 与修复
│   ├── FUTURE_WORK.md        # 未来方向
│   └── NOTES.md              # 写作素材（部分结论已修正）
└── design/                   # 方法论设计

tiny_bot/                     # 核心代码包
```

## 运行实验

```bash
.venv/bin/python exp/07-pso-tradeoff/run.py
```

## 技术栈

- Python 3.11, numpy, pandas, yfinance
- BTC-USD 日数据 2014-2022

## 版本

17 组核心实验 + 8 项补充分析 ≈ 100 次独立运行，所有核心结论均指向具体数据文件。
