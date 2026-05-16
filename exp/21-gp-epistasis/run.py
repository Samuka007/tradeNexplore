"""GP epistasis experiment runner.

Runs instrumented GP across 10 seeds, logging parent-offspring fitness pairs.
"""

import sys
import json
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from gp_epistasis import GP, GPNode
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest
from tiny_bot.filters import wma, sma_filter

# ── config ───────────────────────────────────────────────────────────────────

SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
POP_SIZE = 75
GENERATIONS = 20
LAMBDA = 500.0          # parsimony penalty
MAX_DEPTH = 5
N_WORKERS = 4           # parallel fitness eval

OUT_DIR = Path(__file__).resolve().parent

# ── data ─────────────────────────────────────────────────────────────────────

def load_data():
    """Load BTC train/test split."""
    train, test, _ = load_btc_data()
    return train, test


# ── fitness ──────────────────────────────────────────────────────────────────

def make_fitness_fn(prices: np.ndarray):
    """Return a fitness function that evaluates a GP tree on the given prices.

    Fitness = final_cash from backtest (with continuous position sizing).
    """
    def fitness(tree: GPNode) -> float:
        gp = GP()  # lightweight throwaway for evaluate only
        sig = gp.evaluate(tree, prices, continuous=True)
        result = backtest(prices, sig)
        return result["final_cash"]
    return fitness


# ── train / test evaluation ──────────────────────────────────────────────────

def evaluate_best(gp: GP, tree: GPNode, train_prices: np.ndarray,
                  test_prices: np.ndarray) -> dict:
    """Return train and test metrics for a tree."""
    train_sig = gp.evaluate(tree, train_prices, continuous=True)
    test_sig = gp.evaluate(tree, test_prices, continuous=True)
    train_res = backtest(train_prices, train_sig)
    test_res = backtest(test_prices, test_sig)
    return {
        "train_final_cash": train_res["final_cash"],
        "train_sharpe": train_res["sharpe_ratio"],
        "test_final_cash": test_res["final_cash"],
        "test_sharpe": test_res["sharpe_ratio"],
        "tree_repr": repr(tree),
        "tree_size": gp._tree_size(tree),
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    train, test = load_data()
    print(f"Train prices: {len(train)}, Test prices: {len(test)}")

    results = {"seeds": {}, "aggregate": {}}
    all_pair_counts = []

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # GP with extended function set, parsimony pressure, target depth
        gp = GP(
            pop_size=POP_SIZE,
            generations=GENERATIONS,
            seed=seed,
            parsimony_penalty=LAMBDA,
        )
        gp.max_depth = MAX_DEPTH

        fitness_fn = make_fitness_fn(train)

        print(f"  Optimizing ({POP_SIZE} pop × {GENERATIONS} gen)...")
        result = gp.optimize(fitness_fn, n_workers=N_WORKERS)

        best = result["best"]
        best_fit = result["fitness"]
        history = result["history"]

        # Evaluate on train and test
        best_eval = evaluate_best(gp, best, train, test)

        # Save per-seed pairs
        pairs_file = OUT_DIR / f"seed_{seed:02d}_pairs.json"
        with open(pairs_file, "w") as f:
            json.dump(gp.parent_offspring_log, f, indent=2)

        n_pairs = len(gp.parent_offspring_log)
        all_pair_counts.append(n_pairs)

        seed_result = {
            "seed": seed,
            "best_fitness": best_fit,
            "history": history,
            "n_pairs": n_pairs,
            **best_eval,
        }
        results["seeds"][str(seed)] = seed_result

        print(f"  Best fitness (raw): {best_fit:.2f}")
        print(f"  Train final cash:   {best_eval['train_final_cash']:.2f}")
        print(f"  Test  final cash:   {best_eval['test_final_cash']:.2f}")
        print(f"  Pairs logged:       {n_pairs}")

    # Aggregate stats
    train_cashes = [s["train_final_cash"] for s in results["seeds"].values()]
    test_cashes = [s["test_final_cash"] for s in results["seeds"].values()]
    best_fits = [s["best_fitness"] for s in results["seeds"].values()]

    results["aggregate"] = {
        "n_seeds": len(SEEDS),
        "mean_train_cash": float(np.mean(train_cashes)),
        "std_train_cash": float(np.std(train_cashes)),
        "mean_test_cash": float(np.mean(test_cashes)),
        "std_test_cash": float(np.std(test_cashes)),
        "mean_best_fitness": float(np.mean(best_fits)),
        "std_best_fitness": float(np.std(best_fits)),
        "mean_n_pairs": float(np.mean(all_pair_counts)),
        "total_pairs": int(sum(all_pair_counts)),
    }

    results_file = OUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Aggregate:")
    print(f"  Train cash: {results['aggregate']['mean_train_cash']:.1f} ± {results['aggregate']['std_train_cash']:.1f}")
    print(f"  Test  cash: {results['aggregate']['mean_test_cash']:.1f} ± {results['aggregate']['std_test_cash']:.1f}")
    print(f"  Total pairs logged: {results['aggregate']['total_pairs']}")
    print(f"\nSaved to {results_file}")


if __name__ == "__main__":
    main()
