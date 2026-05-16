"""Load BTC data and split into train (<2020) and test (2020-2022).

All experiments use the same yfinance source for consistency.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def load_btc_data():
    """Return (train_prices, test_prices, full_dataframe) from yfinance BTC-USD.

    Uses the same source and date range as all experiments in exp/.
    """
    df = yf.download(
        "BTC-USD",
        start="2014-01-01",
        end="2022-12-31",
        progress=False,
        auto_adjust=True,
    )
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    train = (
        df[df["Date"] < "2020-01-01"]["Close"]
        .to_numpy(dtype=np.float64)
        .flatten()
    )
    test = (
        df[df["Date"] >= "2020-01-01"]["Close"]
        .to_numpy(dtype=np.float64)
        .flatten()
    )
    return train, test, df


def buy_and_hold_value(test):
    """Final cash from buy-and-hold with $1000 initial."""
    return 1000.0 * test[-1] / test[0]
