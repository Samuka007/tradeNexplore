# 算法接口设计

> 核心问题：连续参数优化（PSO/ABC/HS）与结构发现（GP）如何共存

---

## 1. 核心矛盾

| | PSO/ABC/HS | GP |
|---|---|---|
| **搜索对象** | 固定结构内的连续参数 | 结构本身（树） |
| **表示** | `np.ndarray` 浮点向量 | `PrimitiveTree` 可变深度树 |
| **操作符** | 速度更新、邻域扰动、音高调节 | 子树交叉、节点变异 |
| **维度** | 固定（7D/14D/21D） | 可变（树大小不固定） |

**结论**：强行统一接口会导致 GP 被塞进不合适的抽象，或者 PSO/ABC/HS 要处理不必要的复杂度。

---

## 2. 设计参考来源

| 框架 | 模式 | 启发 |
|------|------|------|
| **DEAP** | `creator.create()` + `Fitness` 抽象 | Fitness 独立于染色体表示，向量/树共享同一比较逻辑 |
| **NiaPy** | `initialization_function` 工厂 | 算法不假设表示，委托给调用方选择 |
| **gplearn** | sklearn `fit(X,y)` 包装 | 把树搜索包装成参数估计器接口 |
| **PySwarms** | `GlobalBestPSO.optimize(func, iters)` | 干净的连续优化接口，返回 `(cost, pos)` |

参考链接：
- DEAP Fitness 抽象: https://github.com/DEAP/deap/blob/master/deap/base.py
- DEAP Creator 模式: https://github.com/DEAP/deap/blob/master/deap/creator.py
- NiaPy Algorithm 基类: https://github.com/NiaOrg/NiaPy/blob/master/src/niapy/algorithms/algorithm.py
- gplearn BaseSymbolic: https://github.com/trevorstephens/gplearn/blob/master/gplearn/genetic.py
- PySwarms GlobalBestPSO: https://github.com/ljvmiranda921/pyswarms/blob/master/pyswarms/single/global_best.py

---

## 3. 推荐方案：三层架构

### 全景图

```
┌──────────────────────────────────────────────────┐
│              ExperimentRunner                     │
│  run_continuous() · run_structural()             │
│  run_hierarchical()                              │
├──────────────────────────────────────────────────┤
│                                                  │
│  ContinuousOptimizer          StructuralOptimizer│
│  ┌──────┬──────┬──────┐      ┌──────────────┐   │
│  │ PSO  │ ABC  │  HS  │      │     GP       │   │
│  └──────┴──────┴──────┘      └──────────────┘   │
│         ↓                           ↓            │
│  VectorStrategy              TreeStrategy        │
│  (params → signals)          (tree → signals)    │
│                  ↓                               │
│            Backtester                            │
│       signals → fitness (final_cash)             │
│            + metrics (sharpe, drawdown, ...)     │
└──────────────────────────────────────────────────┘
```

### Layer 1: 统一的 Strategy 接口（算法无关）

```python
from typing import Protocol
import numpy as np


class Strategy(Protocol):
    """任何能从价格数据生成交易信号的东西"""

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """返回信号序列: +1=买, -1=卖, 0=持有"""
        ...

    def describe(self) -> str:
        """人类可读的策略描述（用于报告/可视化）"""
        ...
```

### Layer 2: 两套优化器接口（承认表示不同）

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class OptResult:
    """统一的优化结果"""
    best: object                    # 最优解（向量 or 树）
    best_fitness: float
    history: list[float]            # 收敛曲线
    metadata: dict = field(default_factory=dict)


class ContinuousOptimizer(ABC):
    """PSO / ABC / HS 共享的接口——优化连续参数向量"""

    @abstractmethod
    def optimize(
        self,
        fitness_fn,                          # (params: np.ndarray) -> float
        bounds: list[tuple[float, float]],    # [(min, max), ...] 每个维度
    ) -> OptResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class StructuralOptimizer(ABC):
    """GP 专用接口——搜索树结构空间"""

    @abstractmethod
    def optimize(
        self,
        fitness_fn,                          # (tree) -> float
        n_generations: int,
    ) -> OptResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass
```

### Layer 3: Strategy 工厂——把优化结果变成 Strategy

```python
class VectorStrategy:
    """连续参数向量 → 固定结构策略（SMA crossover / MACD 等）"""

    def __init__(self, params: np.ndarray, strategy_type: str = 'dual_crossover'):
        self.params = params
        self.type = strategy_type
        self._unpack()

    def _unpack(self):
        if self.type == 'dual_crossover':
            # [w1,w2,w3, d1,d2,d3, α3, w4,w5,w6, d4,d5,d6, α6]
            self.high_weights = self.params[0:3]
            self.high_durations = self.params[3:6]
            self.high_alpha = self.params[6]
            self.low_weights = self.params[7:10]
            self.low_durations = self.params[10:13]
            self.low_alpha = self.params[13]

    def generate_signals(self, prices):
        # 用卷积算 SMA/LMA/EMA → 加权组合 → 差分 → 符号检测
        high_w = self.high_weights / (self.high_weights.sum() + 1e-10)
        d0 = max(int(round(self.high_durations[0])), 2)
        d1 = max(int(round(self.high_durations[1])), 2)
        d2 = max(int(round(self.high_durations[2])), 2)
        high_signal = (
            high_w[0] * wma(prices, d0, sma_filter(d0))
            + high_w[1] * wma(prices, d1, lma_filter(d1))
            + high_w[2] * wma(prices, d2, ema_filter(d2, self.high_alpha))
        )
        # 同理算 low_signal ...
        # diff = high_signal - low_signal
        # signals = crossover_detector(diff)
        # 返回 +1/-1/0

    def describe(self) -> str:
        return f"DualCrossover(high_dur={self.high_durations}, low_dur={self.low_durations})"


class TreeStrategy:
    """GP 树 → 可执行策略"""

    def __init__(self, tree):
        self.tree = tree

    def generate_signals(self, prices):
        # 遍历树求值 → 输出信号
        ...

    def describe(self) -> str:
        return str(self.tree)  # 树的文本表示
```

---

## 4. ExperimentRunner 统一调度

```python
class ExperimentRunner:
    """统一的实验调度层——不关心底层是向量还是树"""

    def __init__(self, data, backtester):
        self.data = data
        self.backtester = backtester

    # ── Parametric algorithms (PSO/ABC/HS) ──

    def run_continuous(
        self,
        optimizer: ContinuousOptimizer,
        bounds: list[tuple[float, float]],
        strategy_type: str,
    ) -> dict:
        """PSO/ABC/HS 的统一调用路径"""

        def fitness_fn(params):
            strategy = VectorStrategy(params, strategy_type)
            signals = strategy.generate_signals(self.data['close'])
            return self.backtester.evaluate(signals).final_cash

        result = optimizer.optimize(fitness_fn, bounds)
        result.metadata['strategy'] = VectorStrategy(result.best, strategy_type)
        return result

    # ── Structural algorithm (GP) ──

    def run_structural(self, optimizer: StructuralOptimizer) -> dict:
        """GP 的调用路径"""

        def fitness_fn(tree):
            strategy = TreeStrategy(tree)
            signals = strategy.generate_signals(self.data['close'])
            return self.backtester.evaluate(signals).final_cash

        result = optimizer.optimize(fitness_fn, n_generations=50)
        result.metadata['strategy'] = TreeStrategy(result.best)
        return result

    # ── Two-stage: GP → PSO/ABC hierarchical ──

    def run_hierarchical(
        self,
        gp: StructuralOptimizer,
        param_algo: ContinuousOptimizer,
    ) -> dict:
        """两阶段：GP 发现结构 → PSO/ABC 精调参数"""

        # Stage 1: GP 发现好的结构
        gp_result = self.run_structural(gp)
        gp_strategy = gp_result.metadata['strategy']

        # Stage 2: 从 GP 树中提取参数化模板
        template = extract_param_template(gp_strategy.tree)
        bounds = template.to_bounds()

        def refined_fitness(params):
            strategy = template.instantiate(params)
            signals = strategy.generate_signals(self.data['close'])
            return self.backtester.evaluate(signals).final_cash

        refined_result = param_algo.optimize(refined_fitness, bounds)
        return {
            'gp_result': gp_result,
            'refined_result': refined_result,
            'improvement': refined_result.best_fitness - gp_result.best_fitness,
        }
```

---

## 5. 设计优势

| 优势 | 说明 |
|------|------|
| **解耦** | PSO/ABC/HS 完全不需要知道 GP 的存在，反之亦然 |
| **可扩展** | 新加算法只需实现 `ContinuousOptimizer` 或 `StructuralOptimizer` |
| **组合灵活** | hierarchical 实验在 ExperimentRunner 层编排，不污染算法代码 |
| **回测独立** | 回测器只接收信号序列，不关心信号怎么来的 |
| **公平比较** | 所有算法走同一个 backtester，同样的数据、同样的手续费 |
