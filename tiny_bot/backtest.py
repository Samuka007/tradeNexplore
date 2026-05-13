"""Minimal backtest engine."""

import numpy as np


def backtest(
    prices: np.ndarray,
    signals: np.ndarray,
    cash: float = 1000.0,
    fee: float = 0.03,
) -> dict:
    """Evaluate signal sequence and return metrics dict.

    Signals can be:
      - discrete: {-1, 0, +1}  (legacy, triggers full buy/sell)
      - continuous: [0, 1]     (position sizing, 0=empty, 1=full)
    Auto-detected from unique values.

    Keys: final_cash, n_trades, n_wins, sharpe_ratio, max_drawdown,
          total_return_pct, win_rate, equity_curve.
    """
    assert len(prices) == len(signals), "prices and signals must match"

    # Auto-detect signal type
    unique_vals = set(np.unique(np.round(signals, 6)))
    is_discrete = unique_vals.issubset({-1.0, 0.0, 1.0})

    btc = 0.0
    trades = []
    entry_cost = 0.0
    equity = np.zeros(len(prices), dtype=np.float64)

    if is_discrete:
        # Legacy discrete logic
        holding_cash = True
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
    else:
        # Continuous position sizing
        prev_target = 0.0
        for i in range(len(prices)):
            price = prices[i]
            target = float(np.clip(signals[i], 0.0, 1.0))

            current_value = cash + btc * price
            current_pos = (btc * price) / current_value if current_value > 0 else 0.0

            if abs(target - current_pos) > 1e-6 and i > 0:
                if target > current_pos:
                    # Buy delta
                    delta = target - current_pos
                    buy_value = delta * current_value
                    cash_needed = buy_value / (1 - fee)
                    actual = min(cash, cash_needed)
                    cash -= actual
                    btc += actual * (1 - fee) / price
                else:
                    # Sell delta
                    delta = current_pos - target
                    sell_value = delta * current_value
                    sell_btc = min(btc, sell_value / price)
                    cash += sell_btc * price * (1 - fee)
                    btc -= sell_btc

                if abs(target - prev_target) > 1e-6:
                    trades.append(cash + btc * price - entry_cost if entry_cost > 0 else 0)

                if target > prev_target:
                    entry_cost = cash + btc * price

                prev_target = target

            equity[i] = cash + btc * price

        final_cash = cash + btc * prices[-1] * (1 - fee)
        if abs(prev_target) > 1e-6:
            trades.append(final_cash - entry_cost)
        cash = final_cash
        equity[-1] = cash

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
