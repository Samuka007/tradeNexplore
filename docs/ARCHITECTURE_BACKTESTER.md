# 回测引擎设计

> 严格遵循 CITS4404 规格书的回测实现

---

## 1. 数据结构

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Trade:
    """单笔交易记录"""
    entry_idx: int
    entry_price: float
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """回测结果——包含 fitness 和各项指标"""
    final_cash: float              # 即 fitness
    trades: list[Trade]
    equity_curve: np.ndarray       # 逐日权益
    n_trades: int
    n_wins: int
    sharpe_ratio: float
    max_drawdown: float
    total_return_pct: float
```

---

## 2. 核心回测引擎

```python
class Backtester:
    """
    严格遵循规格书的回测引擎。

    规则:
    - $1000 起始，0 BTC
    - 全仓买入/卖出
    - 每笔交易 3% 手续费
    - 结束时强制清算剩余 BTC
    - Fitness = 最终现金
    """

    def __init__(
        self,
        prices: np.ndarray,
        initial_cash: float = 1000.0,
        fee_pct: float = 0.03,
    ):
        self.prices = prices
        self.initial_cash = initial_cash
        self.fee_pct = fee_pct

    def evaluate(self, signals: np.ndarray) -> BacktestResult:
        """
        Args:
            signals: +1=买, -1=卖, 0=持有，长度与 prices 相同
        Returns:
            BacktestResult (final_cash 即 fitness)
        """
        cash = self.initial_cash
        holding = 0.0           # BTC 数量
        holding_cash = True
        trades: list[Trade] = []
        equity = np.zeros(len(self.prices))
        current_trade: Optional[Trade] = None

        for i in range(len(self.prices)):
            price = self.prices[i]
            equity[i] = cash + holding * price   # 逐日市值

            # ── 买入 ──
            if signals[i] == 1 and holding_cash:
                fee = cash * self.fee_pct
                holding = (cash - fee) / price
                cash = 0.0
                holding_cash = False
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=price,
                    shares=holding,
                )

            # ── 卖出 ──
            elif signals[i] == -1 and not holding_cash:
                gross = holding * price
                fee = gross * self.fee_pct
                cash = gross - fee
                holding = 0.0
                holding_cash = True
                if current_trade is not None:
                    current_trade.exit_idx = i
                    current_trade.exit_price = price
                    current_trade.pnl = (
                        cash
                        - current_trade.shares * current_trade.entry_price
                    )
                    trades.append(current_trade)
                    current_trade = None

        # ── 强制清算（规格要求）──
        if not holding_cash:
            gross = holding * self.prices[-1]
            fee = gross * self.fee_pct
            cash = gross - fee
            equity[-1] = cash
            if current_trade is not None:
                current_trade.exit_idx = len(self.prices) - 1
                current_trade.exit_price = self.prices[-1]
                current_trade.pnl = (
                    cash
                    - current_trade.shares * current_trade.entry_price
                )
                trades.append(current_trade)

        return BacktestResult(
            final_cash=cash,
            trades=trades,
            equity_curve=equity,
            n_trades=len(trades),
            n_wins=sum(1 for t in trades if t.pnl > 0),
            sharpe_ratio=self._sharpe(equity),
            max_drawdown=self._max_drawdown(equity),
            total_return_pct=(
                (cash - self.initial_cash) / self.initial_cash * 100
            ),
        )

    # ── 指标计算 ──

    def _sharpe(self, equity: np.ndarray, periods_per_year: int = 252) -> float:
        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float(
            np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        )

    def _max_drawdown(self, equity: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return float(np.min(dd))
```

---

## 3. Fitness Function 对接方式

```python
# 方式 1：简单版——只返回 final_cash（规格书默认）
def make_fitness(prices, strategy_type):
    backtester = Backtester(prices)

    def fitness(params):
        strategy = VectorStrategy(params, strategy_type)
        signals = strategy.generate_signals(prices)
        result = backtester.evaluate(signals)
        return result.final_cash  # 标量 → 优化器直接用

    return fitness


# 方式 2：复合版——惩罚过度交易和回撤
def make_fitness_penalized(prices, strategy_type):
    backtester = Backtester(prices)

    def fitness(params):
        strategy = VectorStrategy(params, strategy_type)
        signals = strategy.generate_signals(prices)
        result = backtester.evaluate(signals)

        # 惩罚交易次数过多（手续费吃掉利润）
        if result.n_trades > 50:
            penalty = 0.01 * (result.n_trades - 50)
        else:
            penalty = 0

        return result.final_cash * (1 - penalty)

    return fitness


# 使用示例：
# prices = load_btc_data()
# fitness = make_fitness(prices, 'dual_crossover')
# pso = PSO(n_particles=30)
# result = pso.optimize(fitness, bounds=[...])
```

---

## 4. 常见陷阱防坑

| 坑 | 影响 | 怎么防 |
|---|---|---|
| **前视偏差 (Look-ahead bias)** | 用了未来的数据做决策 | 信号在第 N 根 K 线计算，交易在第 N 根收盘价执行。SMA 用 `prices[0:i+1]` 计算，信号在 i 处触发，用的是当前价（已包含在 SMA 中）。如需更严格，SMA 应基于 `prices[0:i]`（不含当前），在第 i 根的 open 执行 |
| **NaN 指标** | 程序崩溃 | 跳过前 `max(window_sizes)` 根 K 线，信号默认为 0（持有） |
| **浮点窗口** | PSO/ABC 输出 float | `duration` 需要 `int(round())` 才能做窗口大小 |
| **权重归一化** | scale 影响搜索 | `w_i / sum(w)` 归一化，防止权重的 scale 干扰优化器 |
| **手续费扣除时机** | 低估成本 | 在交易时立即扣除，不是事后算 |
| **全仓精度** | 部分成交 | 用浮点 BTC 数量（加密货币支持小数） |
| **信号长度不对齐** | 数组越界 | filters.py 中 pad + valid 卷积保证输出长度 = 输入长度 |
