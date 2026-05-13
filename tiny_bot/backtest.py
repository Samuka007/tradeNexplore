"""Minimal backtest engine."""

import numpy as np


def backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    cash: float = 1000.0,
    fee: float = 0.03,
) -> dict:
    """Evaluate signal sequence and return metrics dict.

    Keys: final_cash, n_trades, n_wins, sharpe_ratio, max_drawdown,
          total_return_pct, win_rate, equity_curve.
    """
    assert len(prices) == len(signals), "prices and signals must match"

    holding_cash = True
    btc = 0.0
    trades = []
    entry_cost = 0.0
    equity = np.zeros(len(prices), dtype=np.float64)

    for i in range(len(prices)):
        price = prices[i]
        if signals[i] == 1 and holding_cash:
            entry_cost = cash
            btc = cash * (1 - fee) / price
            cash = 0.0
            holding_cash = False
        elif signals[i] == -1 and not holding_cash:
            cash = btc * price * (1 - fee)
            btc = 0.0
            holding_cash = True
            trades.append(cash - entry_cost)
        equity[i] = cash + btc * price

    if not holding_cash:
        cash = btc * prices[-1] * (1 - fee)
        equity[-1] = cash
        trades.append(cash - entry_cost)

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t > 0)
    returns = np.diff(equity) / (equity[:-1] + 1e-10)
    sharpe = (
        float(np.mean(returns) / np.std(returns) * np.sqrt(252))
        if len(returns) > 1 and np.std(returns) > 1e-10
        else 0.0
    )
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-10)
    max_dd = float(np.min(dd)) if len(equity) > 0 else 0.0

    return {
        "final_cash": float(cash),
        "n_trades": n_trades,
        "n_wins": n_wins,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "total_return_pct": (cash - 1000.0) / 1000.0 * 100,
        "win_rate": (n_wins / n_trades * 100) if n_trades else 0.0,
        "equity_curve": equity.tolist(),
    }


def buy_and_hold(prices: np.ndarray, cash: float = 1000.0, fee: float = 0.03) -> float:
    """Buy at first price, hold to end, sell at last price."""
    if len(prices) == 0:
        return cash
    btc = cash * (1 - fee) / prices[0]
    return float(btc * prices[-1] * (1 - fee))
