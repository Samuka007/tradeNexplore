"""Backtest engine for trading strategy evaluation.

Implements the exact rules from the project specification:
- $1000 USD initial cash, 0 BTC
- Full position buy/sell
- 3% fee per transaction
- Forced liquidation of remaining BTC at end
- Fitness = final cash value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Trade:
    """Single trade record."""

    entry_idx: int
    entry_price: float
    entry_cost: float = 0.0
    exit_idx: Optional[int] = None
    exit_price: Optional[float] = None
    shares: float = 0.0
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Backtest results including fitness and performance metrics."""

    final_cash: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    n_trades: int = 0
    n_wins: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0

    @property
    def fitness(self) -> float:
        """Primary fitness metric — final cash value."""
        return self.final_cash


class Backtester:
    """Evaluate trading signals on historical price data.

    Rules:
    - Start with $1000 cash, 0 BTC
    - Buy/sell entire position
    - 3% transaction fee on each trade
    - Liquidate any remaining BTC at final price
    """

    def __init__(
        self,
        prices: np.ndarray,
        initial_cash: float = 1000.0,
        fee_pct: float = 0.03,
    ):
        self.prices = np.asarray(prices, dtype=np.float64)
        self.initial_cash = float(initial_cash)
        self.fee_pct = float(fee_pct)

    def evaluate(self, signals: np.ndarray) -> BacktestResult:
        """Evaluate signal sequence and return backtest results.

        Args:
            signals: Array of +1 (buy), -1 (sell), 0 (hold).
                     Length must match prices.

        Returns:
            BacktestResult with fitness and metrics.
        """
        signals = np.asarray(signals)
        if len(signals) != len(self.prices):
            raise ValueError(
                f"Signals length ({len(signals)}) must match prices ({len(self.prices)})"
            )

        cash = self.initial_cash
        holding_btc = 0.0
        holding_cash = True
        trades: list[Trade] = []
        equity = np.zeros(len(self.prices), dtype=np.float64)
        current_trade: Optional[Trade] = None

        for i in range(len(self.prices)):
            price = self.prices[i]

            if signals[i] == 1 and holding_cash:
                entry_cost = cash
                fee = cash * self.fee_pct
                holding_btc = (cash - fee) / price
                cash = 0.0
                holding_cash = False
                current_trade = Trade(
                    entry_idx=i,
                    entry_price=price,
                    entry_cost=entry_cost,
                    shares=holding_btc,
                )

            elif signals[i] == -1 and not holding_cash:
                gross = holding_btc * price
                fee = gross * self.fee_pct
                cash = gross - fee
                holding_btc = 0.0
                holding_cash = True
                if current_trade is not None:
                    current_trade.exit_idx = i
                    current_trade.exit_price = price
                    current_trade.pnl = cash - current_trade.entry_cost
                    trades.append(current_trade)
                    current_trade = None

            equity[i] = cash + holding_btc * price

        if not holding_cash:
            gross = holding_btc * self.prices[-1]
            fee = gross * self.fee_pct
            cash = gross - fee
            equity[-1] = cash
            if current_trade is not None:
                current_trade.exit_idx = len(self.prices) - 1
                current_trade.exit_price = self.prices[-1]
                current_trade.pnl = cash - current_trade.entry_cost
                trades.append(current_trade)

        return BacktestResult(
            final_cash=cash,
            trades=trades,
            equity_curve=equity,
            n_trades=len(trades),
            n_wins=sum(1 for t in trades if t.pnl > 0),
            sharpe_ratio=self._sharpe(equity),
            max_drawdown=self._max_drawdown(equity),
            total_return_pct=(cash - self.initial_cash) / self.initial_cash * 100,
            win_rate=(sum(1 for t in trades if t.pnl > 0) / len(trades) * 100)
            if trades
            else 0.0,
            avg_trade_pnl=np.mean([t.pnl for t in trades]) if trades else 0.0,
        )

    def _sharpe(self, equity: np.ndarray, periods_per_year: int = 252) -> float:
        returns = np.diff(equity) / (equity[:-1] + 1e-10)
        if len(returns) < 2 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))

    def _max_drawdown(self, equity: np.ndarray) -> float:
        if len(equity) == 0:
            return 0.0
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10)
        return float(np.min(dd))


def buy_and_hold(prices: np.ndarray, initial_cash: float = 1000.0, fee_pct: float = 0.03) -> float:
    """Compute buy-and-hold baseline fitness.

    Buys at first price, holds until end, sells at last price.

    Returns:
        Final cash value.
    """
    if len(prices) == 0:
        return initial_cash
    btc = initial_cash * (1 - fee_pct) / prices[0]
    return btc * prices[-1] * (1 - fee_pct)


def make_fitness(
    prices: np.ndarray,
    strategy_type: str = "dual_crossover",
    use_baseline: bool = False,
):
    """Factory for fitness functions used by optimizers.

    Args:
        prices: Price sequence to evaluate on.
        strategy_type: Strategy type identifier.
        use_baseline: If True, normalize fitness against buy-and-hold.

    Returns:
        Fitness function: (params: np.ndarray) -> float.
    """
    from trading_bot.strategy import VectorStrategy

    backtester = Backtester(prices)
    baseline = buy_and_hold(prices) if use_baseline else None

    def fitness(params: np.ndarray) -> float:
        strategy = VectorStrategy(params, strategy_type)
        signals = strategy.generate_signals(prices)
        result = backtester.evaluate(signals)
        if baseline is not None and baseline > 0:
            return result.fitness / baseline
        return result.fitness

    return fitness


def make_fitness_penalized(
    prices: np.ndarray,
    strategy_type: str = "dual_crossover",
    max_trades: int = 50,
    penalty_rate: float = 0.01,
):
    """Factory for penalized fitness functions.

    Penalizes strategies that trade too frequently (fees eat profits).

    Args:
        prices: Price sequence to evaluate on.
        strategy_type: Strategy type identifier.
        max_trades: Trade count threshold before penalty kicks in.
        penalty_rate: Penalty as fraction of fitness per excess trade.

    Returns:
        Fitness function: (params: np.ndarray) -> float.
    """
    from trading_bot.strategy import VectorStrategy

    backtester = Backtester(prices)

    def fitness(params: np.ndarray) -> float:
        strategy = VectorStrategy(params, strategy_type)
        signals = strategy.generate_signals(prices)
        result = backtester.evaluate(signals)
        penalty = 0.0
        if result.n_trades > max_trades:
            penalty = penalty_rate * (result.n_trades - max_trades)
        return result.fitness * (1 - penalty)

    return fitness
