"""AI Trading Bot Framework using nature-inspired optimization algorithms.

This package provides:
- Convolution-based WMA filters (SMA, LMA, EMA)
- Trading strategy implementations (dual crossover, MACD, GP tree)
- Backtest engine with fitness evaluation
- Optimization algorithm interfaces (PSO, ABC, HS, GP)
- Experiment runner for comparing algorithms
- Data loading utilities for Kaggle BTC dataset
"""

from trading_bot.backtester import (
    Backtester,
    BacktestResult,
    Trade,
    buy_and_hold,
    make_fitness,
    make_fitness_penalized,
)
from trading_bot.data_loader import Dataset, load_kaggle_btc_data, load_or_generate_data
from trading_bot.filters import (
    crossover_detector,
    ema_filter,
    lma_filter,
    pad,
    sma_filter,
    wma,
)
from trading_bot.strategy import Strategy, VectorStrategy, TreeStrategy

__version__ = "0.1.0"

__all__ = [
    "Backtester",
    "BacktestResult",
    "Trade",
    "buy_and_hold",
    "make_fitness",
    "make_fitness_penalized",
    "Dataset",
    "load_kaggle_btc_data",
    "load_or_generate_data",
    "crossover_detector",
    "ema_filter",
    "lma_filter",
    "pad",
    "sma_filter",
    "wma",
    "Strategy",
    "VectorStrategy",
    "TreeStrategy",
]
