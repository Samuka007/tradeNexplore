"""Visualization utilities for experiment results."""

from __future__ import annotations

from typing import Optional

import numpy as np

from trading_bot.backtester import BacktestResult


def plot_convergence(
    histories: dict[str, list[float]],
    title: str = "Convergence Comparison",
    save_path: Optional[str] = None,
):
    """Plot convergence curves for multiple algorithms.

    Args:
        histories: Dict mapping algorithm name to fitness history list.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, history in histories.items():
        ax.plot(history, label=name, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fitness (Final Cash)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_equity_curves(
    results: dict[str, BacktestResult],
    title: str = "Equity Curves",
    save_path: Optional[str] = None,
):
    """Plot equity curves for multiple strategies.

    Args:
        results: Dict mapping strategy name to BacktestResult.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        ax.plot(result.equity_curve, label=name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdowns(
    results: dict[str, BacktestResult],
    title: str = "Drawdown Analysis",
    save_path: Optional[str] = None,
):
    """Plot drawdown curves for multiple strategies.

    Args:
        results: Dict mapping strategy name to BacktestResult.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, result in results.items():
        equity = result.equity_curve
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / (peak + 1e-10) * 100
        ax.fill_between(range(len(dd)), dd, alpha=0.3, label=name)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_trade_points(
    prices: np.ndarray,
    results: dict[str, BacktestResult],
    title: str = "Trade Points",
    save_path: Optional[str] = None,
):
    """Plot price with buy/sell markers.

    Args:
        prices: Price array.
        results: Dict mapping strategy name to BacktestResult.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(prices, label="Price", alpha=0.7, color="black")

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, result), color in zip(results.items(), colors):
        for trade in result.trades:
            ax.scatter(
                trade.entry_idx,
                prices[trade.entry_idx],
                marker="^",
                color=color,
                s=50,
                alpha=0.7,
            )
            ax.scatter(
                trade.exit_idx,
                prices[trade.exit_idx],
                marker="v",
                color=color,
                s=50,
                alpha=0.7,
            )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_report(
    results: dict[str, BacktestResult],
    output_dir: str = "results",
) -> dict:
    """Generate a summary report of backtest results.

    Args:
        results: Dict mapping strategy name to BacktestResult.
        output_dir: Directory to save plots.

    Returns:
        Dictionary with summary statistics.
    """
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, result in results.items():
        summary[name] = {
            "final_cash": result.final_cash,
            "total_return_pct": result.total_return_pct,
            "n_trades": result.n_trades,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "avg_trade_pnl": result.avg_trade_pnl,
        }

    plot_equity_curves(results, save_path=str(out / "equity_curves.png"))
    plot_drawdowns(results, save_path=str(out / "drawdowns.png"))

    return summary
