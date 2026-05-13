"""Load BTC data and split into train (<2020) and test (2020-2022)."""

from pathlib import Path
import numpy as np
import pandas as pd


def load_btc_data(csv_path: str):
    """Return (train_prices, test_prices) as numpy arrays."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"BTC data not found: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    train = df[df["date"] < "2020-01-01"]["close"].to_numpy(dtype=np.float64)
    test = df[df["date"] >= "2020-01-01"]["close"].to_numpy(dtype=np.float64)
    return train, test
