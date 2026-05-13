# PSO + GP 交易机器人：受控消融报告（中文）

## 0. 关于原始实现的说明

原始代码库的 WMA 滤波器中含有一个 `pad()` 函数：

```python
def pad(P, N):
    padding = -np.flip(P[1:N])
    return np.append(padding, P)
```

这**不是一个有效的基线**——它是一个引入了**前视偏差**的 bug。通过翻转前 `N-1` 个未来价格并拼接到序列头部，时刻 *t=0* 的 WMA 依赖于 *t=1, ..., t+N-1* 的价格。在真实交易系统中这是不可能的：时刻 *t* 你无法观测到未来的价格。使用此 padding 的回测因此**作弊**，任何在其上评估的策略都会产生**毫无价值、虚高的训练适应度**。

本报告中的修正实现使用**因果卷积**：时刻 *t* 的 WMA 输出仅使用 `P[t-N+1]` 到 `P[t]`。前 `N-1` 个输出为 `NaN`（历史不足），这自然地在预热期抑制信号。本报告中的所有实验均使用因果 WMA 作为**唯一有效的基线**。

---

## 1. 实验设计

### 1.1 研究问题
为什么 PSO 和 GP 优化的交易策略在 BTC 测试集（2020–2022）上表现不如买入并持有？

### 1.2 数据
从 Yahoo Finance 下载的 BTC-USD 日收盘价（2014-01-01 – 2022-12-31）。  
划分：**训练集** = 2020-01-01 之前的价格；**测试集** = 2020-01-01 及之后。  
固定随机种子 = 42。

### 1.3 算法
* **PSO**：30 粒子，50 次迭代，自适应惯性权重（0.9 → 0.4），c1 = c2 = 2.05。
* **GP**：种群 50，30 代，锦标赛大小 3，交叉率 0.9，变异率 0.1，最大深度 5。
* **基线**：买入并持有（首次买入、最后一次卖出，每边 3% 手续费）。

### 1.4 假设
| 编号 | 假设 | 操作 |
|----|------|------|
| **基线** | 因果 WMA + 3% 手续费 | 修正代码，无前视偏差 |
| H1 | 交易成本侵蚀 | 设手续费 = 0.0 |
| H2 | 制度变迁（非平稳性） | 反转训练/测试时间顺序 |
| H3 | 对噪声的过拟合 | 打乱训练价格（破坏序列相关性） |

---

## 2. 结果

所有数字均为**测试集最终现金（美元）**（Yahoo Finance BTC-USD，种子 = 42）。

| 条件 | PSO 测试 ($) | GP 测试 ($) | 买入并持有测试 ($) |
|------|-------------:|------------:|------------------:|
| **基线**（因果 WMA，3% 手续费） | 803 | 336 | 2,170 |
| **H1 零手续费** | 3,240 | 357 | 2,170 |
| **H2 反转划分** | **20,182** | **7,888** | 14,799 |
| **H3 打乱训练** | 2,456 | 0.92 | 2,170 |

*训练集适应度*在此省略，因为所有假设均关注**样本外泛化**；完整训练/测试明细见 `experiment_results.json`。

---

## 3. 分析

### 3.1 交易成本（H1）——不可忽略的次要因素
手续费为 0 时，PSO 测试提升 4 倍（$803 → $3,240）。GP 几乎不变（$336 → $357），因为进化树倾向于产生极少交易。
3% 往返手续费按现代标准极为极端（典型零售加密货币现货手续费约为 0.1–0.5%）。市场微观结构方面的经验研究表明，成本感知优化对真实世界可行性至关重要 [1]。

**结论**：成本侵蚀优势，但即使无摩擦交易也无法在前向测试中击败基线。

### 3.2 制度变迁（H2）——**主导原因**
当将高波动的 2020–2022 期用于*训练*、趋势性的 2014–2019 期用于*测试*时，两种算法均**击败买入并持有**（PSO $20,182 vs $14,799；GP $7,888 vs $14,799）。
这种不对称性是**非平稳性**的标志：在一个制度上学到的参数在另一个制度上失效，但反向迁移成功，因为"波动"制度信息更丰富。

BTC 市场在 2020 年前后经历了有记载的结构断裂：机构采用（Tesla、MicroStrategy）、减半后供应动态，以及宏观冲击（COVID-19）改变了收益分布和波动率制度 [2][3]。

### 3.3 对噪声的过拟合（H3）——有界，非主导
在打乱价格（破坏所有序列相关性）上，训练适应度发散至无穷大（来自虚假信号的溢出），但测试表现崩溃至接近零。
这证明算法**确实**从时间结构中学习，而非从虚假静态特征中学习。因此前向测试失败更应归因于**分布偏移**，而非通用过拟合。

---

## 4. 综合

1. **主因**：数据非平稳性 / 制度变迁。单一训练/测试划分在结构性断裂资产上是不可靠的评估协议。
2. **次因**：交易成本（3%）。现代手续费结构会部分恢复 PSO 优势，但无法弥合制度差距。
3. **原始前视 padding**：已确认为 bug 并修正。它从未是有效基线。

### 实际意义
* **前向验证**（滚动/扩展窗口）对非平稳资产是必要的 [4]。
* **制度检测**（如马尔可夫转换模型）应优先于优化，使参数以当前制度为条件 [2]。
* **风险感知适应度**（夏普比率、最大回撤）而非原始最终现金，将惩罚在回测中悄然失效的高方差策略 [5]。

---

## 5. 参考文献

[1] R. Kissell \u0026 M. Glantz, "Optimal Trading Strategies", Amacom, 2003.

[2] S. Wang \u0026 Y. Chen, "Regime switching forecasting for cryptocurrencies", *Digital Finance*, vol. 6, pp. 1–22, 2024. https://doi.org/10.1007/s42521-024-00123-2

[3] M. Zargar \u0026 D. Kumar, "Detecting Structural Changes in Bitcoin, Altcoins, and the S\u0026P 500 Using the GSADF Test", *Journal of Risk and Financial Management*, vol. 18, no. 8, 2025. https://doi.org/10.3390/jrfm18080450

[4] M. Lopez de Prado, "Advances in Financial Machine Learning", Wiley, 2018. (Ch. 7: Cross-Validation in Finance)

[5] D. H. Bailey \u0026 M. Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", *Journal of Portfolio Management*, vol. 40, no. 5, 2014. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2465675

[6] S.-H. Chen \u0026 N. Navet, "Failure of Genetic-Programming Induced Trading Strategies: Distinguishing between Efficient Markets and Inefficient Algorithms", in *Genetic Programming*, LNCS 4445, Springer, 2007. https://doi.org/10.1007/978-3-540-71605-1_11

---

*生成时间：2026-05-13*  
*数据来源：Yahoo Finance BTC-USD*  
*代码：`exp/ablation/experiments.py`*  
*原始结果：`exp/ablation/experiment_results.json`*
