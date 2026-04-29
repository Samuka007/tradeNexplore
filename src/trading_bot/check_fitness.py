"""CLI script to check fitness of a fixed trading policy.

Usage:
    python -m trading_bot.check_fitness --params "1,0,0,10,20,30,0.3,1,0,0,15,25,35,0.3"
    python -m trading_bot.check_fitness --strategy macd --params "12,0.2,26,0.1,9,0.2,0.0" --synthetic

This demonstrates the framework working with a fixed (non-optimized) policy.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from trading_bot.backtester import Backtester, buy_and_hold
from trading_bot.data_loader import load_or_generate_data
from trading_bot.strategy import VectorStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_params(param_str: str) -> np.ndarray:
    """Parse comma-separated parameter string into numpy array."""
    try:
        values = [float(x.strip()) for x in param_str.split(",")]
        return np.array(values, dtype=np.float64)
    except ValueError as e:
        raise ValueError(f"Invalid parameter format: {e}") from e


def check_fitness(
    params: np.ndarray,
    strategy_type: str = "dual_crossover",
    use_synthetic: bool = False,
    use_baseline: bool = False,
) -> dict:
    """Check fitness of a fixed policy on train and test data.

    Args:
        params: Strategy parameters as numpy array.
        strategy_type: Strategy type identifier.
        use_synthetic: If True, use synthetic data instead of real BTC data.
        use_baseline: If True, include buy-and-hold baseline comparison.

    Returns:
        Dictionary with train/test fitness and backtest metrics.
    """
    dataset = load_or_generate_data(fallback_to_synthetic=use_synthetic)

    strategy = VectorStrategy(params, strategy_type)
    logger.info("Strategy: %s", strategy.describe())

    train_signals = strategy.generate_signals(dataset.train_prices)
    train_backtest = Backtester(dataset.train_prices).evaluate(train_signals)

    test_signals = strategy.generate_signals(dataset.test_prices)
    test_backtest = Backtester(dataset.test_prices).evaluate(test_signals)

    result = {
        "train": {
            "fitness": train_backtest.fitness,
            "return_pct": train_backtest.total_return_pct,
            "n_trades": train_backtest.n_trades,
            "win_rate": train_backtest.win_rate,
            "sharpe": train_backtest.sharpe_ratio,
            "max_drawdown": train_backtest.max_drawdown,
        },
        "test": {
            "fitness": test_backtest.fitness,
            "return_pct": test_backtest.total_return_pct,
            "n_trades": test_backtest.n_trades,
            "win_rate": test_backtest.win_rate,
            "sharpe": test_backtest.sharpe_ratio,
            "max_drawdown": test_backtest.max_drawdown,
        },
        "params": params.tolist(),
        "strategy_type": strategy_type,
    }

    if use_baseline:
        train_baseline = buy_and_hold(dataset.train_prices)
        test_baseline = buy_and_hold(dataset.test_prices)
        result["train"]["buy_hold"] = train_baseline
        result["train"]["vs_buy_hold"] = (
            train_backtest.fitness / train_baseline if train_baseline > 0 else 0.0
        )
        result["test"]["buy_hold"] = test_baseline
        result["test"]["vs_buy_hold"] = (
            test_backtest.fitness / test_baseline if test_baseline > 0 else 0.0
        )

    return result


def print_results(results: dict) -> None:
    """Pretty-print fitness check results."""
    print("\n" + "=" * 60)
    print("FITNESS CHECK RESULTS")
    print("=" * 60)
    print(f"Strategy: {results['strategy_type']}")
    print(f"Parameters: {results['params']}")
    print()
    print("-" * 30)
    print("TRAIN SET")
    print("-" * 30)
    for key, value in results["train"].items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:12.4f}")
        else:
            print(f"  {key:15s}: {value:12}")
    print()
    print("-" * 30)
    print("TEST SET")
    print("-" * 30)
    for key, value in results["test"].items():
        if isinstance(value, float):
            print(f"  {key:15s}: {value:12.4f}")
        else:
            print(f"  {key:15s}: {value:12}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for fitness checking CLI."""
    parser = argparse.ArgumentParser(
        description="Check fitness of a fixed trading policy"
    )
    parser.add_argument(
        "--params",
        type=str,
        default="1,0,0,10,20,30,0.3,1,0,0,15,25,35,0.3",
        help="Comma-separated parameter values (14D for dual_crossover, 7D for macd)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="dual_crossover",
        choices=["dual_crossover", "macd"],
        help="Strategy type",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of real BTC data",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Include buy-and-hold baseline comparison",
    )

    args = parser.parse_args(argv)

    try:
        params = parse_params(args.params)
        results = check_fitness(
            params=params,
            strategy_type=args.strategy,
            use_synthetic=args.synthetic,
            use_baseline=args.baseline,
        )
        print_results(results)
        return 0
    except Exception as e:
        logger.error("Fitness check failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
