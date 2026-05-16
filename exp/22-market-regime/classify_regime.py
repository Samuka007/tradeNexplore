"""Classify BTC daily data into bull/bear/sideways regimes using 90-day returns.

Outputs:
  - regimes.json: { "dates": [...], "labels": [...], "close": [...],
      "regime_periods": { "bull": [[start_date, end_date], ...], ... } }
  - regimes.pdf: price chart with regime coloring
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
import pandas as pd


def load_btc_csv(csv_path: str) -> np.ndarray:
    """Load close prices chronologically from CSV file."""
    df = pd.read_csv(csv_path)
    # CSV is newest-first; reverse to chronological
    df = df.iloc[::-1].reset_index(drop=True)
    return df["close"].values, df["date"].values


def classify_regimes(close: np.ndarray, window: int = 90,
                     bull_threshold: float = 0.30,
                     bear_threshold: float = -0.30) -> np.ndarray:
    """Classify each day as 'bull', 'bear', or 'sideways' based on rolling returns.

    First (window-1) days are classified as 'sideways' since return data is insufficient.
    Returns array of strings.
    """
    n = len(close)
    labels = np.full(n, "sideways", dtype=object)
    if n <= window:
        return labels

    # Compute rolling returns: (price[t] - price[t-window]) / price[t-window]
    for i in range(window, n):
        ret = (close[i] - close[i - window]) / close[i - window]
        if ret > bull_threshold:
            labels[i] = "bull"
        elif ret < bear_threshold:
            labels[i] = "bear"
        # else stays "sideways"

    return labels


def find_regime_periods(dates: np.ndarray, labels: np.ndarray) -> dict:
    """Find contiguous periods for each regime type.

    Returns dict mapping regime -> list of [start_date, end_date, start_idx, end_idx].
    """
    periods = {"bull": [], "bear": [], "sideways": []}
    n = len(labels)
    i = 0
    while i < n:
        regime = labels[i]
        start = i
        while i < n and labels[i] == regime:
            i += 1
        periods[str(regime)].append({
            "start_date": str(dates[start]),
            "end_date": str(dates[i - 1]),
            "start_idx": int(start),
            "end_idx": int(i - 1),
            "length": i - start,
        })
    return periods


def plot_regimes(dates: np.ndarray, close: np.ndarray, labels: np.ndarray,
                 output_path: str):
    """Plot price with regime coloring to PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    date_dt = pd.to_datetime(dates)
    colors = {"bull": "green", "bear": "red", "sideways": "gray"}

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(date_dt, close, color="black", linewidth=0.8, alpha=0.5)

    # Shade regime backgrounds
    n = len(labels)
    i = 0
    while i < n:
        regime = labels[i]
        start = i
        while i < n and labels[i] == regime:
            i += 1
        ax.axvspan(date_dt[start], date_dt[i - 1],
                   alpha=0.15, color=colors.get(str(regime), "gray"))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", alpha=0.3, label="Bull (>+30%)"),
        Patch(facecolor="red",   alpha=0.3, label="Bear (<−30%)"),
        Patch(facecolor="gray",  alpha=0.3, label="Sideways"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    ax.set_title("BTC/USD Close Price with 90-Day Return Regime Classification")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "btc_daily_2014_2022.csv"
    close, dates = load_btc_csv(str(DATA_PATH))
    labels = classify_regimes(close)

    periods = find_regime_periods(dates, labels)

    # Summary
    for regime in ["bull", "bear", "sideways"]:
        ps = periods[regime]
        total_days = sum(p["length"] for p in ps)
        print(f"{regime}: {len(ps)} periods, {total_days} days")

    # Save regimes.json
    output = {
        "dates": dates.tolist(),
        "labels": labels.tolist(),
        "close": close.tolist(),
        "regime_periods": periods,
    }
    with open("regimes.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved regimes.json")

    # Plot
    plot_regimes(dates, close, labels, "regimes.pdf")
    print("Saved regimes.pdf")


if __name__ == "__main__":
    main()
