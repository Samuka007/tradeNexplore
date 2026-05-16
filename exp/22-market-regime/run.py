"""Run GP separately on bull, bear, and sideways regime data.

For each regime type:
  1. Extract contiguous periods from regimes.json.
  2. Accumulate periods from the end as test until reaching MIN_TEST_DAYS (30).
  3. Remaining periods become train (concatenated).
  4. Run GP on train with 10 seeds (pop=75, gen=20, λ=500).
  5. Evaluate best tree on test vs buy-and-hold.

Outputs:
  - bull_results.json, bear_results.json, sideways_results.json
  - results.json (summary)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import numpy as np
from tiny_bot.gp import GP
from tiny_bot.backtest import backtest, buy_and_hold

SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
POP_SIZE = 75
GENERATIONS = 20
PARSIMONY = 500.0
MAX_DEPTH = 5
MIN_TEST_DAYS = 30
N_WORKERS = 4


def load_regimes(path: str = "regimes.json") -> dict:
    with open(path) as f:
        return json.load(f)


def split_train_test(close: list, periods: list, min_test_days: int = MIN_TEST_DAYS):
    """Split periods into train (chronologically earlier) and test (later).

    Accumulates periods from the end until test has >= min_test_days.
    Returns (train_prices, test_prices, train_info, test_info).
    Returns (None, None, None, None) if not enough data.
    """
    # Periods are in chronological order
    test_periods = []
    test_days = 0
    for p in reversed(periods):
        test_days += p["length"]
        test_periods.insert(0, p)
        if test_days >= min_test_days:
            break

    n_test = len(test_periods)
    train_periods = periods[:-n_test] if n_test < len(periods) else []

    if not train_periods:
        # Not enough for a train/test split: use first 70% of all data as train
        total_days = sum(p["length"] for p in periods)
        if total_days < min_test_days * 2:
            return None, None, None, None
        split_day = int(total_days * 0.7)
        # Walk through periods to find split point
        train_parts = []
        test_parts = []
        days_seen = 0
        for p in periods:
            if days_seen + p["length"] <= split_day:
                train_parts.append(np.array(
                    close[p["start_idx"]:p["end_idx"] + 1], dtype=np.float64))
                days_seen += p["length"]
            else:
                test_parts.append(np.array(
                    close[p["start_idx"]:p["end_idx"] + 1], dtype=np.float64))
        if train_parts and test_parts:
            return (np.concatenate(train_parts), np.concatenate(test_parts),
                    f"{len(train_parts)} periods", f"{len(test_parts)} periods")
        return None, None, None, None

    # Concatenate
    train_parts = [np.array(close[p["start_idx"]:p["end_idx"] + 1], dtype=np.float64)
                   for p in train_periods]
    test_parts = [np.array(close[p["start_idx"]:p["end_idx"] + 1], dtype=np.float64)
                  for p in test_periods]
    train_prices = np.concatenate(train_parts)
    test_prices = np.concatenate(test_parts)

    train_info = f"{len(train_periods)} periods"
    test_info = f"{len(test_periods)} periods"
    return train_prices, test_prices, train_info, test_info


def run_regime(regime_name: str, train_prices: np.ndarray,
               test_prices: np.ndarray, seeds: list[int],
               pop_size: int, generations: int, parsimony: float) -> dict:
    """Run GP on train, evaluate best on test, for multiple seeds."""
    np.random.seed(42)
    bh_test = buy_and_hold(test_prices)

    seed_results = []
    for seed in seeds:
        gp = GP(pop_size=pop_size, generations=generations, seed=seed,
                parsimony_penalty=parsimony)
        gp.max_depth = MAX_DEPTH

        res = gp.optimize(
            lambda t: backtest(train_prices, gp.evaluate(t, train_prices))["final_cash"],
            n_workers=N_WORKERS,
        )

        best_tree = res["best"]
        try:
            test_cash = backtest(test_prices,
                                 gp.evaluate(best_tree, test_prices))["final_cash"]
        except (ValueError, IndexError) as e:
            # Indicator computation failed on tiny test set — fall back to BH
            test_cash = bh_test
            print(f"  {regime_name} seed={seed:3d} WARNING: eval failed ({e}), "
                  f"using BH=${bh_test:,.0f}")

        tree_size = gp._tree_size(best_tree)
        tree_repr = repr(best_tree)
        train_fitness = res["fitness"]
        history = res["history"]

        seed_results.append({
            "seed": seed,
            "test_cash": float(test_cash),
            "train_fitness": float(train_fitness),
            "tree_size": tree_size,
            "tree_repr": tree_repr,
            "history": history,
        })
        print(f"  {regime_name} seed={seed:3d} test=${test_cash:>10,.0f} "
              f"size={tree_size} bh=${bh_test:>10,.0f}")

    beats = sum(1 for r in seed_results if r["test_cash"] > bh_test)
    test_cashes = [r["test_cash"] for r in seed_results]

    result = {
        "regime": regime_name,
        "train_days": int(len(train_prices)),
        "test_days": int(len(test_prices)),
        "bh_test": float(bh_test),
        "seeds": seed_results,
        "summary": {
            "mean_test": float(np.mean(test_cashes)),
            "std_test": float(np.std(test_cashes)),
            "median_test": float(np.median(test_cashes)),
            "max_test": float(np.max(test_cashes)),
            "beats_bh": beats,
            "n_seeds": len(seeds),
        },
    }
    return result


def main():
    data = load_regimes()
    close = data["close"]
    regime_periods = data["regime_periods"]

    results = {}
    for regime in ["bull", "bear", "sideways"]:
        periods = regime_periods[regime]
        if not periods:
            print(f"No {regime} periods found; skipping.")
            continue

        train_prices, test_prices, train_info, test_info = \
            split_train_test(close, periods)
        if train_prices is None:
            print(f"Skipping {regime}: insufficient data.")
            continue

        print(f"\n=== {regime.upper()} ===")
        print(f"  Train: {train_info}, {len(train_prices)} days")
        print(f"  Test:  {test_info}, {len(test_prices)} days")

        result = run_regime(regime, train_prices, test_prices, SEEDS,
                           POP_SIZE, GENERATIONS, PARSIMONY)
        results[regime] = result

        with open(f"{regime}_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {regime}_results.json")

    summary = {}
    for regime, r in results.items():
        summary[regime] = r["summary"]
    with open("results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
