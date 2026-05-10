# 实验测试计划

**日期:** 2026-05-10
**版本:** 1.0
**算法:** PSO + GP（ABC 和 Harmony Search 不做，原因见 §6）

---

## 1. 实验目标

对比 **参数优化（PSO）** 与 **结构发现（GP）** 两种范式在 BTC 交易策略优化上的表现。通过三个递进实验回答：

| 实验 | 核心问题 |
|------|----------|
| **Exp 1** | PSO 能否在固定策略结构中找到比传统基线更好的参数？ |
| **Exp 2** | 让 GP 自由演化交易规则，能否超越人工设计的结构 + PSO 调参？ |
| **Exp 3** | GP 发现结构 → PSO 精调参数，层级组合是否优于各自单独使用？ |

---

## 2. 数据配置

| 参数 | 值 |
|------|-----|
| 数据源 | `data/btc_daily_2014_2022.csv`（Kaggle Bitcoin Historical Dataset） |
| 价格列 | `close`（收盘价） |
| 训练集 | 2014-01-01 → 2017-12-31（~1130 天） |
| 验证集 | 2018-01-01 → 2019-12-31（~730 天） |
| 测试集 | 2020-01-01 → 2022-03-01（~791 天） |
| 回测规则 | $1000 初始现金，3% 手续费，全仓买卖，强制平仓 |

---

## 3. 基线策略

所有实验统一对比以下基线：

| 基线 | 实现 | 说明 |
|------|------|------|
| **Buy-and-Hold** | `backtester.buy_and_hold()` | 第一天全仓买入，最后一天卖出 |
| **GoldenCross** | `strategy.GoldenCross(50, 200)` | 传统的 50/200 SMA 金叉策略 |
| **DeathCross** | `strategy.DeathCross(50, 200)` | 传统的 50/200 SMA 死叉策略 |

---

## 4. 实验设计

### 4.1 实验 1：PSO 参数优化

**目标：** 在固定策略结构中，PSO 能否找到超越基线的参数组合。

**配置：**

| 参数 | 值 |
|------|-----|
| 算法 | PSO |
| 粒子数 | 30 |
| 最大迭代 | 50 |
| 惯性权重 | 自适应 0.9 → 0.4 |
| c1, c2 | 2.05, 2.05 |
| 独立运行次数 | **10** |
| 策略变体 | dual_crossover (14D) + MACD (7D) |
| 评估数据 | 训练集优化，验证集监控，测试集最终评估 |

**dual_crossover 搜索空间（14 维）：**

```
参数向量: [w1, w2, w3, d1, d2, d3, α3, w4, w5, w6, d4, d5, d6, α6]
         └── HIGH 分量 ────┘              └── LOW 分量 ────┘
```

| 维度 | 参数 | 范围 | 类型 |
|------|------|------|------|
| w1-w3 | HIGH 滤波器权重 | [0.0, 1.0] | 连续，自动归一化 |
| d1-d3 | HIGH 滤波器窗口 | [2, 200] | 整数 |
| α3 | HIGH EMA 衰减 | [0.01, 0.99] | 连续 |
| w4-w6 | LOW 滤波器权重 | [0.0, 1.0] | 连续，自动归一化 |
| d4-d6 | LOW 滤波器窗口 | [2, 200] | 整数 |
| α6 | LOW EMA 衰减 | [0.01, 0.99] | 连续 |

**MACD 搜索空间（7 维）：**

| 维度 | 参数 | 范围 |
|------|------|------|
| d1 | 快线窗口 | [2, 200] |
| α1 | 快线 EMA 衰减 | [0.01, 0.99] |
| d2 | 慢线窗口 | [2, 200] |
| α2 | 慢线 EMA 衰减 | [0.01, 0.99] |
| d3 | 信号线窗口 | [2, 200] |
| α3 | 信号线 EMA 衰减 | [0.01, 0.99] |
| threshold | MACD 触发阈值 | [0.0, 1.0] |

**输出指标（每次运行）：**

| 指标 | 来源 | 说明 |
|------|------|------|
| train_fitness | 训练集回测 | 最终现金（USD） |
| val_fitness | 验证集回测 | 防止过拟合 |
| test_fitness | 测试集回测 | 最终评估 |
| n_trades | BacktestResult | 交易次数 |
| win_rate | BacktestResult | 胜率（%） |
| sharpe_ratio | BacktestResult | 夏普比率 |
| max_drawdown | BacktestResult | 最大回撤 |
| convergence | OptResult.history | 50 代收敛曲线 |

**汇总报告（10 次运行后）：**

| 统计量 | 计算方式 |
|--------|----------|
| Mean ± Std | 10 次 test_fitness 的均值和标准差 |
| Best | 10 次中 test_fitness 最高的一次 |
| Worst | 10 次中 test_fitness 最低的一次 |
| Convergence | 10 条收敛曲线的均值 ± 标准差阴影 |

---

### 4.2 实验 2：GP 结构发现 vs PSO

**目标：** GP 自由演化交易规则树，与人工设计的 dual_crossover + PSO 对比。

**配置：**

| 参数 | 值 |
|------|-----|
| 算法 | GeneticProgramming |
| 种群大小 | 100 |
| 进化代数 | 50 |
| 交叉率 | 0.9 |
| 变异率 | 0.1 |
| 最大树深度 | 5 |
| 锦标赛大小 | 3 |
| 独立运行次数 | **10** |
| 函数集 | `+`, `-`, `*`, `>`, `<`, `AND`, `IF` |
| 终端集 | `price`, `sma(N)`, `lma(N)`, `ema(N,α)`, `const` |

**对比对象：** 实验 1 中 PSO 在 dual_crossover 上的最佳结果。

**输出指标：** 同实验 1，外加：
- 最佳 GP 树的文本表示（`repr(tree)`）
- 树深度和节点数

---

### 4.3 实验 3：层级组合（GP → PSO）

**目标：** 用 GP 发现结构，再用 PSO 在该结构上精调参数。

**两阶段流程：**

```
Stage 1: GP 演化规则树（同实验 2）
    ↓ 选出最佳树
Stage 2: 从 GP 树中提取参数模板 → PSO 精调模板参数
    ↓
对比: GP 原始 vs PSO 精调后
```

**Stage 2 配置：**

| 参数 | 值 |
|------|-----|
| 算法 | PSO（同实验 1 参数） |
| 粒子数 | 30 |
| 最大迭代 | 50 |
| 独立运行 | 取实验 2 中 3 个最佳 GP 树，各跑 5 次 PSO |

**额外输出：**

| 指标 | 说明 |
|------|------|
| improvement | `fitness_PSO_refined - fitness_GP_original` |
| improvement_pct | 相对提升百分比 |

---

## 5. 技术实现细节

### 5.1 需要新增/修复的代码

| 组件 | 文件 | 状态 | 说明 |
|------|------|:---:|------|
| `data_loader` 列名修复 | `data_loader.py` | ✅ 已修 | 默认列名改为 `date`/`close` |
| GPAdapter | 实验脚本内 | ⚠️ 待写 | `TreeStrategy` 不支持 `GPNode`，需简单适配器 |
| 实验运行脚本 | `experiments/run_all.py` | ❌ 待写 | 一键跑三个实验 |
| 结果汇总脚本 | 实验脚本内 | ❌ 待写 | 10 次运行的统计汇总量化分析 |

### 5.2 GPAdapter 设计

`ExperimentRunner.run_structural()` 要求 `strategy_factory(tree) -> Strategy`，但 `TreeStrategy` 的 `generate_signals()` 对 `GPNode` 类型会 fallback 到全零信号。需要一个小适配器：

```python
class GPAdapter:
    """Wrap GP tree evaluation as a Strategy-compatible object."""
    def __init__(self, gp, tree):
        self._gp = gp
        self._tree = tree

    def generate_signals(self, prices):
        return self._gp.evaluate(self._tree, prices)

    def describe(self):
        return f"GP({self._tree})"
```

### 5.3 实验运行时间估算

| 实验 | 单次运行 | × 次数 | 估算总时间 |
|------|----------|--------|-----------|
| Exp 1 PSO (dual_crossover) | ~3s | 10 | ~30s |
| Exp 1 PSO (MACD) | ~2s | 10 | ~20s |
| Exp 2 GP | ~30s | 10 | ~5min |
| Exp 3 层级 | ~30s(GP) + ~3s(PSO) | 3 × 5 | ~2min |
| **总计** | | | **≈ 8 分钟** |

> 实测环境：Windows, Python 3.12, Intel CPU。时间可能因机器性能浮动。

---

## 6. 不做 ABC 和 Harmony Search 的理由

1. **任务书只要求 "one or more algorithms"** — PSO + GP 已满足
2. **ABC 和 HS 与 PSO 属于同一范式（连续参数优化），三者对比只能回答"谁跑得快"，无法回答"参数优化 vs 结构演化"这类更有学术深度的问题**
3. **PSO vs GP 范式差异足够形成有洞察力的对比** — 这才是课程想探讨的核心
4. **时间约束** — 距离截止日不足两周，将时间投入高质量实验分析和报告写作，比多塞两个半成品算法更有价值
5. **报告中会提及 ABC/HS 作为文献调研内容**，但在实验设计中明确标注为"超出本次实验范围"

---

## 7. 报告结构预期（3000 字）

| 章节 | 字数 | 内容 |
|------|------|------|
| Introduction | 200 | Part 1 调研回顾 + 实验目标 |
| Bot Design | 400 | WMA 滤波器、策略参数化、假设空间 |
| Algorithm Implementation | 600 | PSO 和 GP 的实现细节与参数选择 |
| Experimental Design | 500 | 数据划分、三个实验的设计逻辑 |
| Results | 800 | 收敛曲线、测试集性能对比、统计显著性 |
| Discussion | 400 | 算法行为分析、过拟合讨论、层级组合有效性 |
| Conclusion | 100 | 主要发现 + 未来方向 |

---

## 8. 附录：实验执行命令

```bash
# 确保环境正确
pip install -e ".[dev]"

# 运行全部实验（待实现）
python experiments/run_all.py

# 单独运行实验 1
python experiments/run_all.py --experiment 1

# 查看结果
python experiments/run_all.py --report
```
