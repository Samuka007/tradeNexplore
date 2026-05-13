"""Minimal Mixture-of-Experts (MoE) trading bot.

Architecture:
1. K-means (K=2) on (returns, volatility) of the training set for regime splitting.
2. Per-regime PSO to optimize trivial SMA parameters [d_fast, d_slow].
3. Router: IF volatility > N THEN expert_A ELSE expert_B, with N optimized by PSO.
4. Fitness = final cash on full training set.
"""

import sys
from pathlib import Path

# Ensure repo root is on PYTHONPATH so tiny_bot is importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
import json
import numpy as np
import pandas as pd
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO

SEED = 42
np.random.seed(SEED)


def load_btc_data():
    """Load BTC data from repo CSV and split into train/test."""
    script_dir = Path(__file__).parent
    csv_path = script_dir / "../../data/btc_daily_2014_2022.csv"
    if not csv_path.exists():
        csv_path = Path("data/btc_daily_2014_2022.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    train = df[df["date"] < "2020-01-01"]["close"].to_numpy(dtype=np.float64)
    test = df[df["date"] >= "2020-01-01"]["close"].to_numpy(dtype=np.float64)
    return train, test


def compute_features(prices, vol_window=10):
    """Compute (returns, volatility) features for each timestep.

    returns[t]   corresponds to prices[t+1] vs prices[t]
    volatility   is rolling std of returns ending at each point.
    Returns array shape (len(prices)-1, 2).
    """
    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    vol = np.zeros(n)
    for i in range(n):
        start = max(0, i - vol_window + 1)
        vol[i] = np.std(returns[start : i + 1])
    return np.column_stack([returns, vol])


def kmeans(X, k=2, max_iter=100, seed=42):
    """Manual K-means using numpy only (no sklearn)."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    idx = rng.choice(n_samples, k, replace=False)
    centroids = X[idx].copy()

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array(
            [
                X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
                for i in range(k)
            ]
        )

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


def moe_signals(prices, params_high, params_low, N, vol_window=10):
    """Generate MoE signals using volatility-based router."""
    sig_high = VectorStrategy(params_high, "trivial_sma").signals(prices)
    sig_low = VectorStrategy(params_low, "trivial_sma").signals(prices)

    returns = np.diff(prices) / prices[:-1]
    n = len(returns)
    vol = np.zeros(n)
    for i in range(n):
        start = max(0, i - vol_window + 1)
        vol[i] = np.std(returns[start : i + 1])

    router = np.zeros(len(prices), dtype=int)
    # vol[i] corresponds to decision at prices[i+1]
    router[1:] = np.where(vol > N, sig_high[1:], sig_low[1:])
    router[0] = sig_low[0]
    return router


def main():
    train, test = load_btc_data()

    print("=" * 70)
    print("Mixture-of-Experts (MoE) Trading Bot")
    print("=" * 70)

    # --- Baselines ---
    bh_train = buy_and_hold(train)
    bh_test = buy_and_hold(test)
    print(f"\nBuy-and-Hold       train=${bh_train:>10,.0f}  test=${bh_test:>10,.0f}")

    sma_bounds = [(2, 200), (2, 200)]

    # PSO trivial SMA (no MoE)
    pso_sma = PSO(n_particles=30, max_iter=50, seed=SEED)
    sma_res = pso_sma.optimize(
        lambda p: backtest(train, VectorStrategy(p, "trivial_sma").signals(train))[
            "final_cash"
        ],
        sma_bounds,
    )
    sma_train = backtest(
        train, VectorStrategy(sma_res["best"], "trivial_sma").signals(train)
    )
    sma_test = backtest(
        test, VectorStrategy(sma_res["best"], "trivial_sma").signals(test)
    )
    print(
        f"PSO SMA (no MoE)   train=${sma_train['final_cash']:>10,.0f}  test=${sma_test['final_cash']:>10,.0f}"
    )

    # Classic 50/200 SMA
    classic_train = backtest(
        train, VectorStrategy(np.array([50.0, 200.0]), "trivial_sma").signals(train)
    )
    classic_test = backtest(
        test, VectorStrategy(np.array([50.0, 200.0]), "trivial_sma").signals(test)
    )
    print(
        f"Classic 50/200 SMA train=${classic_train['final_cash']:>10,.0f}  test=${classic_test['final_cash']:>10,.0f}"
    )

    # --- Step 1: Regime Splitting via K-means ---
    features = compute_features(train, vol_window=10)
    labels, centroids = kmeans(features, k=2, seed=SEED)

    # Identify high-vol vs low-vol cluster
    vol_mean_0 = features[labels == 0, 1].mean()
    vol_mean_1 = features[labels == 1, 1].mean()
    high_idx, low_idx = (0, 1) if vol_mean_0 > vol_mean_1 else (1, 0)
    high_mean_vol = features[labels == high_idx, 1].mean()
    low_mean_vol = features[labels == low_idx, 1].mean()

    print(
        f"\nK-means: high-vol cluster={high_idx} (mean_vol={high_mean_vol:.5f}), "
        f"low-vol cluster={low_idx} (mean_vol={low_mean_vol:.5f})"
    )

    # Create regime masks aligned to full price array (features[i] -> prices[i+1])
    mask_high = np.zeros(len(train), dtype=int)
    mask_low = np.zeros(len(train), dtype=int)
    mask_high[1:] = (labels == high_idx).astype(int)
    mask_low[1:] = (labels == low_idx).astype(int)

    # --- Step 2: Expert Training ---
    print("\nTraining high-volatility expert...")
    pso_high = PSO(n_particles=30, max_iter=50, seed=SEED)
    res_high = pso_high.optimize(
        lambda p: backtest(
            train,
            VectorStrategy(p, "trivial_sma").signals(train) * mask_high,
        )["final_cash"],
        sma_bounds,
    )
    print(
        f"  Best params: [{res_high['best'][0]:.2f}, {res_high['best'][1]:.2f}], fitness=${res_high['fitness']:,.0f}"
    )

    print("Training low-volatility expert...")
    pso_low = PSO(n_particles=30, max_iter=50, seed=SEED)
    res_low = pso_low.optimize(
        lambda p: backtest(
            train,
            VectorStrategy(p, "trivial_sma").signals(train) * mask_low,
        )["final_cash"],
        sma_bounds,
    )
    print(
        f"  Best params: [{res_low['best'][0]:.2f}, {res_low['best'][1]:.2f}], fitness=${res_low['fitness']:,.0f}"
    )

    # --- Step 3: Router Optimization (1 extra dimension) ---
    print("Optimizing router threshold N...")
    vol_train = compute_features(train, vol_window=10)[:, 1]
    vol_min = float(vol_train.min())
    vol_max = float(vol_train.max())

    pso_router = PSO(n_particles=20, max_iter=30, seed=SEED)
    res_router = pso_router.optimize(
        lambda N: backtest(
            train,
            moe_signals(train, res_high["best"], res_low["best"], N[0]),
        )["final_cash"],
        bounds=[(vol_min, vol_max)],
    )
    N_best = float(res_router["best"][0])
    print(f"  Best threshold N={N_best:.6f}, fitness=${res_router['fitness']:,.0f}")

    # --- Final Evaluation ---
    moe_train = backtest(
        train, moe_signals(train, res_high["best"], res_low["best"], N_best)
    )
    moe_test = backtest(
        test, moe_signals(test, res_high["best"], res_low["best"], N_best)
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Method':<25} {'Train':>12} {'Test':>12} {'vs BH?':>8}")
    print("-" * 70)

    methods = [
        ("Buy-and-Hold", bh_train, bh_test),
        ("PSO SMA (no MoE)", sma_train["final_cash"], sma_test["final_cash"]),
        ("Classic 50/200 SMA", classic_train["final_cash"], classic_test["final_cash"]),
        ("MoE (ours)", moe_train["final_cash"], moe_test["final_cash"]),
    ]

    for name, tr, te in methods:
        wins = "YES" if te > bh_test else "NO"
        print(f"{name:<25} ${tr:>10,.0f} ${te:>10,.0f} {wins:>8}")

    # --- Save results ---
    results = {
        "buy_and_hold": {"train": float(bh_train), "test": float(bh_test)},
        "experiments": [
            {
                "name": "PSO_SMA_no_MoE",
                "train_cash": float(sma_train["final_cash"]),
                "test_cash": float(sma_test["final_cash"]),
                "best_params": [float(x) for x in sma_res["best"]],
            },
            {
                "name": "Classic_50_200_SMA",
                "train_cash": float(classic_train["final_cash"]),
                "test_cash": float(classic_test["final_cash"]),
            },
            {
                "name": "MoE",
                "train_cash": float(moe_train["final_cash"]),
                "test_cash": float(moe_test["final_cash"]),
                "high_vol_expert_params": [float(x) for x in res_high["best"]],
                "low_vol_expert_params": [float(x) for x in res_low["best"]],
                "router_threshold": N_best,
                "kmeans_centroids": centroids.tolist(),
            },
        ],
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results.json")


if __name__ == "__main__":
    main()
