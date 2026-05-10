"""Data loading utilities for Bitcoin historical data.

Supports the Kaggle Bitcoin Historical Dataset with automatic
train/test splitting per the project specification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """Container for train/validation/test price data."""

    train_prices: np.ndarray
    test_prices: np.ndarray
    val_prices: Optional[np.ndarray] = None
    train_dates: Optional[np.ndarray] = None
    test_dates: Optional[np.ndarray] = None
    val_dates: Optional[np.ndarray] = None
    full_df: Optional[pd.DataFrame] = None

    @property
    def train_years(self) -> tuple[int, int]:
        if self.train_dates is not None and len(self.train_dates) > 0:
            return (int(self.train_dates[0]), int(self.train_dates[-1]))
        return (2014, 2019)

    @property
    def test_years(self) -> tuple[int, int]:
        if self.test_dates is not None and len(self.test_dates) > 0:
            return (int(self.test_dates[0]), int(self.test_dates[-1]))
        return (2020, 2022)

    @property
    def val_years(self) -> tuple[int, int]:
        if self.val_dates is not None and len(self.val_dates) > 0:
            return (int(self.val_dates[0]), int(self.val_dates[-1]))
        return (2018, 2019)


def load_kaggle_btc_data(
    csv_path: str | Path,
    date_column: str = "date",
    price_column: str = "close",
    train_end: str = "2017-12-31",
    val_start: str = "2018-01-01",
    val_end: str = "2019-12-31",
    test_start: str = "2020-01-01",
) -> Dataset:
    """Load Bitcoin historical data from Kaggle CSV and split train/val/test.

    The Kaggle Bitcoin Historical Dataset typically contains columns:
    Date, Open, High, Low, Close, Volume, etc.

    Splits per spec:
    - Train: 2014-2017
    - Validation: 2018-2019 (hyperparameter tuning, early stopping)
    - Test: 2020-2022 (final evaluation, unseen data)

    Args:
        csv_path: Path to the CSV file (e.g. 'data/btc_daily_2014_2022.csv').
        date_column: Name of the date column.
        price_column: Name of the price column to use (typically 'Close').
        train_end: Last date for training data (inclusive).
        val_start: First date for validation data (inclusive).
        val_end: Last date for validation data (inclusive).
        test_start: First date for test data (inclusive).

    Returns:
        Dataset with train/val/test split.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"BTC data file not found: {csv_path}\n"
            "Please download the Kaggle Bitcoin Historical Dataset and place it at:\n"
            "  data/btc_daily_2014_2022.csv\n"
            "Dataset URL: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data"
        )

    logger.info("Loading BTC data from %s", csv_path)
    df = pd.read_csv(csv_path)

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    df["_year"] = df[date_column].dt.year

    train_mask = df[date_column] <= pd.Timestamp(train_end)
    val_mask = (df[date_column] >= pd.Timestamp(val_start)) & (df[date_column] <= pd.Timestamp(val_end))
    test_mask = df[date_column] >= pd.Timestamp(test_start)

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    train_prices = train_df[price_column].to_numpy(dtype=np.float64)
    val_prices = val_df[price_column].to_numpy(dtype=np.float64) if len(val_df) > 0 else np.array([], dtype=np.float64)
    test_prices = test_df[price_column].to_numpy(dtype=np.float64)

    logger.info(
        "Loaded %d train (%s-%s), %d val (%s-%s), %d test (%s-%s) points",
        len(train_prices),
        train_df[date_column].iloc[0].date() if len(train_df) > 0 else "N/A",
        train_df[date_column].iloc[-1].date() if len(train_df) > 0 else "N/A",
        len(val_prices),
        val_df[date_column].iloc[0].date() if len(val_df) > 0 else "N/A",
        val_df[date_column].iloc[-1].date() if len(val_df) > 0 else "N/A",
        len(test_prices),
        test_df[date_column].iloc[0].date() if len(test_df) > 0 else "N/A",
        test_df[date_column].iloc[-1].date() if len(test_df) > 0 else "N/A",
    )

    return Dataset(
        train_prices=train_prices,
        test_prices=test_prices,
        val_prices=val_prices,
        train_dates=train_df["_year"].to_numpy(),
        test_dates=test_df["_year"].to_numpy(),
        val_dates=val_df["_year"].to_numpy() if len(val_df) > 0 else None,
        full_df=df,
    )


def generate_synthetic_data(
    n_train: int = 2000,
    n_val: int = 500,
    n_test: int = 1000,
    seed: int = 42,
    trend: float = 0.05,
    volatility: float = 0.03,
    start_price: float = 5000.0,
) -> Dataset:
    """Generate synthetic BTC-like price data for testing.

    Uses geometric Brownian motion with drift to simulate realistic
    crypto price movements.

    Args:
        n_train: Number of training data points.
        n_val: Number of validation data points.
        n_test: Number of test data points.
        seed: Random seed for reproducibility.
        trend: Daily drift (annualized ~trend * 252).
        volatility: Daily volatility.
        start_price: Starting price in USD.

    Returns:
        Dataset with synthetic train/val/test prices.
    """
    rng = np.random.default_rng(seed)

    def _generate(n: int) -> np.ndarray:
        returns = rng.normal(trend / 252, volatility / np.sqrt(252), size=n)
        prices = start_price * np.exp(np.cumsum(returns))
        return prices.astype(np.float64)

    train_prices = _generate(n_train)
    val_prices = _generate(n_val)
    test_prices = _generate(n_test)

    logger.info(
        "Generated synthetic data: %d train, %d val, %d test points",
        n_train, n_val, n_test,
    )

    return Dataset(
        train_prices=train_prices,
        test_prices=test_prices,
        val_prices=val_prices,
        train_dates=np.repeat(2017, n_train),
        val_dates=np.repeat(2019, n_val),
        test_dates=np.repeat(2021, n_test),
    )


def load_or_generate_data(
    csv_path: str | Path = "data/btc_daily_2014_2022.csv",
    fallback_to_synthetic: bool = True,
    **kwargs,
) -> Dataset:
    """Load real BTC data or fall back to synthetic data.

    Args:
        csv_path: Path to the Kaggle BTC CSV.
        fallback_to_synthetic: If True, generate synthetic data when CSV is missing.
        **kwargs: Passed to generate_synthetic_data if fallback is used.

    Returns:
        Dataset with train/test split.
    """
    try:
        return load_kaggle_btc_data(csv_path)
    except FileNotFoundError:
        if fallback_to_synthetic:
            logger.warning(
                "Real BTC data not found at %s — falling back to synthetic data",
                csv_path,
            )
            return generate_synthetic_data(**kwargs)
        raise
