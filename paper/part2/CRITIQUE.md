# 系统缺陷分析：基于一线文献的对比

## 已引用文献 vs 应引用文献

当前 refs.bib 有 11 条，全是"算法本身"的引用（Kennedy 1995 PSO, Koza 1992 GP, 等）。
没有一条是 **为什么这些结果很重要** 或 **别人已经发现了什么** 的文献。

对标的顶刊/顶会文献：

| 我们的声明 | 应引用的文献 | 缺失影响 |
|---|---|---|
| GP 过拟合，需要正则化 | Allen & Karjalainen 1999 ✅（已引） | — |
| λ=500 是甜点 | Poli 2008 parsimony pressure ✅（已引） | 但缺 Poli & McPhee 2003 "Parsimony Pressure Made Easy"（GECCO），该文给出了理论界 |
| 数据窥探/过拟合 | White 2000 "Reality Check" (JASA)；Sullivan et al. 1999 "Data-Snooping" (JF)；Harvey & Liu 2015 "Backtesting" | **完全缺失**——这是金融领域最重要的方法论贡献，我们的 walk-forward 论证没有一条文献支撑 |
| Walk-forward 比单次分割更诚实 | Lopez de Prado 2018 "Advances in Financial Machine Learning" Ch.11-12 | **完全缺失**——我们声称"W-F 验证至关重要"但没有引用任何方法论文献 |
| PSO 应用于交易 | Fei et al. 2014 ✅（已引） | — |
| 表示设计 > 算法选择 | **没有对应文献**——这是我们的核心声称，但没有前人工作来定位 | 如果前人已经发现同一结论，我们的贡献只是复现；如果没有人提出过，我们的声称就是无锚定的 |

## 系统缺陷

### 1. 缺少文献定位（最严重）

论文提出了一个核心论点："representation design dominates algorithm selection"，但：
- 没有引用任何发表过类似结论的论文
- 没有引用任何反驳类似结论的论文
- 没有说明这个论点是新的、是验证前人的、还是推翻前人的

在顶刊/顶会中，每条主张都需要锚定在文献中。我们的论文读起来像是一个独立发现，但 GP 在交易中过拟合这个事实已经被 Allen & Karjalainen (1999) 报告过了，walk-forward 的必要性已经被 White (2000) 和 Lopez de Prado (2018) 详细论证过了，数据窥探 bias 已经被 Sullivan et al. (1999) 用 SPA/RC 检验解决了。

**我们的贡献不应该是"发现 GP 会过拟合"——这已经是共识。** 我们的贡献应该是：在 BTC 日线数据 + 3% 手续费 + position_sizing 表示这个特定设定下，量化了 PSO 和 GP 的性能差距，并给出了具体的正则化参数建议。

### 2. 没有统计检验

所有数字都是均值和标准差。没有任何：
- 置信区间
- t-test 或 Wilcoxon 检验
- 校正多重比较（Bonferroni, FDR, 或 SPA/RC）
- bootstrap

在 White (2000) 之后，金融交易规则的论文如果不做 SPA/RC 检验，在 JF/JFE 根本不会被送审。即使是 GECCO/EvoFIN 级别，也期望至少有 Wilcoxon 或 Mann-Whitney。

我们声称"10/10 seeds beat BH"，但没有检查这些种子是否在统计上显著优于 BH（考虑到只有 10 个观察值，而且 BTC 价格是非平稳的，BH 本身的方差可能很大）。

### 3. 没有风险调整收益

所有结果都是绝对收益（$）。没有：
- 夏普比率
- Sortino 比率
- 最大回撤
- Calmar 比率

在交易策略论文中，只报告绝对收益而不报告风险调整收益是严重的遗漏。一个 $2,366 的策略如果最大回撤是 60%，可能不如一个 $1,500 但回撤只有 10% 的策略。

### 4. 单资产 + 单时间段 + 滚动训练窗口只做了一半

- 只用了 BTC 一种资产（课程要求如此，但应该在局限性中更坦率地说明）
- 训练集和测试集的比例（3:1）没有经过敏感性分析
- Walk-forward 只做了 PSO position_sma 的 5 窗口（GP 的 WF 做了但数据不一致）
- 没有 expanding-window WF（只有 rolling-window）

### 5. λ=500 的"甜点"可能只是数据过拟合

我们用 3 seeds × 7 λ × 2 depths = 42 runs 找到了 λ=500。但：
- 在 7 个 λ 值中做选择，没有做 Bonferroni 校正
- 3 seeds 的统计效力极低（每个 λ 只有 6 个观察值）
- λ=500 的 3/6 beat BH，λ=5000 的 1/6 also beat BH——差异不显著
- 没有做 cross-validation 或 nested-validation

### 6. 缺少与 baseline 的公平比较

论文中的 GP 对比对象是 PSO+position_sma，但这两者的表示空间完全不同：
- PSO 搜索 3 个连续参数
- GP 搜索无限维的树结构

如果 GP 的表示也是 position_sma（3 个参数），PSO vs GP 应该收敛到同一个结果。我们没有做这个关键对照实验。

### 7. 没有讨论 non-stationarity 的底层原因

实验 01 的反向价格测试得到了 $20,182（vs BH $2,170），但我们只说了"这确认了 non-stationarity"。在金融文献中，这是一个需要深入分析的现象——具体而言，是 trending vs mean-reverting regimes 的切换。我们没有对 BTC 在 2020-2022 期间的具体 regime 变化做任何分析。

## 建议修改优先级

1. **必须加引用**：White 2000 (SPA/RC), Sullivan et al. 1999 (data-snooping), Lopez de Prado 2018 (AFML Ch.11-12), Agapitos et al. 2010 (GP trading without data-mining bias)
2. **必须加统计检验**：至少 Wilcoxon signed-rank 或 Mann-Whitney U，对主要结论做显著性检验
3. **优先加风险指标**：至少 Sharpe ratio 和最大回撤
4. **加 GP+position_sma 对照实验**（如果时间允许）
5. **改写 Discussion**：与已知文献对话，而不是独立宣称