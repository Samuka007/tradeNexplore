# Part 2 核心框架架构设计

> 基于规格书分析与现成框架调研的架构决策

---

## 1. 技术栈决策

### 1.1 规格书硬性约束

| 约束 | 内容 |
|------|------|
| **不能用的** | 外部交易机器人代码、通用优化库（scipy.optimize 等） |
| **必须手写的** | 优化算法（PSO/ABC/GP/HS）、Bot 构建、回测评估 |
| **可以参考的** | 论文附带的代码（需注明出处） |
| **可以用的** | numpy、pandas、matplotlib、plotly |

### 1.2 技术栈选择

| 层次 | 选择 | 说明 |
|------|------|------|
| **数据加载** | Kaggle CSV + pandas | 规格指定 Kaggle Bitcoin Historical Dataset |
| **补充数据** | ccxt (v4.5.44) | 可选，获取实时 OHLCV |
| **信号/卷积** | numpy `np.convolve` | 规格明确给出了代码 |
| **TA 指标参考** | pandas-ta / ta / TA-Lib | **仅开发阶段交叉验证用**，正式代码不 import |
| **回测引擎** | **自己实现** | backtrader/vectorbt 都不符合规格要求 |
| **优化算法** | **自己实现** | 不能用 pyswarms/DEAP/NiaPy 等现成库 |
| **可视化** | matplotlib + plotly | 收敛曲线、权益曲线、K 线图 |

**关键结论**：除了 numpy/pandas/matplotlib，基本没有可以直接用的现成框架。回测和优化都需要自己写。

### 1.3 TA 指标参考是什么

`pandas-ta` / `ta` / `TA-Lib` 这些库标注为"⚠️ 仅参考"，意思是：

- 这些库能计算 SMA/EMA/MACD
- **但我们的实现方式不同**——规格书要求用**卷积**方式（`np.convolve`），而不是调用这些库的函数
- 它们的唯一价值是：开发阶段用它们算出 SMA(20)，和我们的卷积结果对比，确认数值一致
- **正式代码不会 import 这些库**

---

## 2. 规格书硬性要求汇总

### 2.1 信号生成（卷积方式）

三种 WMA 滤波器，全部通过**卷积**实现：

```python
def pad(P, N):
    """Edge condition: flip first N-1 points"""
    padding = -np.flip(P[1:N])
    return np.append(padding, P)

def sma_filter(N):
    """SMA: equal weights"""
    return np.ones(N) / N

def lma_filter(N):
    """LMA: triangular weights"""
    k = np.arange(N)
    return (2 / (N + 1)) * (1 - k / N)

def ema_filter(N, alpha):
    """EMA: exponential decay"""
    k = np.arange(N)
    return alpha * (1 - alpha) ** k

def wma(P, N, kernel):
    """通用加权移动平均——卷积方式"""
    return np.convolve(pad(P, N), kernel, 'valid')

def crossover_detector(diff):
    """检测符号变化: kernel = [1, -1]/2"""
    kernel = np.array([1, -1]) / 2
    sign_signal = np.sign(diff)
    return np.convolve(sign_signal, kernel, 'valid')
```

### 2.2 回测评估规则

| 规则 | 值 |
|------|----|
| 起始资金 | $1000 USD + 0 BTC |
| 手续费 | 每笔交易 3% |
| 仓位 | 全仓买入/卖出 |
| 结束 | 强制清算剩余 BTC → 现金 |
| **Fitness** | **最终持有的现金** |

### 2.3 数据要求

| 数据集 | 用途 |
|--------|------|
| Kaggle Bitcoin Historical Dataset (2014-2022) | 基础数据源 |
| 2014-2019 | 训练（优化用） |
| 2020-2022 | 测试（最终评估，unseen data） |

### 2.4 参数化策略设计

规格书给出的复合策略公式：

```
HIGH = (w1·SMA(d1) + w2·LMA(d2) + w3·EMA(d3, α3)) / Σwi
LOW  = (w4·SMA(d4) + w5·LMA(d5) + w6·EMA(d6, α6)) / Σwi
```

- 单边组件：7D 参数 `[w1, w2, w3, d1, d2, d3, α3]`
- 双边交叉：14D 参数
- MACD（含第三组件）：21D 参数
