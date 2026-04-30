"""Demo script: PSO and GP algorithm evaluation with baseline comparison.

Run with:
    python -m trading_bot.demo_algorithms

Shows:
- Baseline strategies (Golden Cross, Death Cross, Buy-and-Hold)
- PSO optimizing dual_crossover parameters
- GP evolving tree-structured rules
- Fitness comparison across all approaches
"""

from __future__ import annotations

import logging

import numpy as np

from trading_bot.algorithms.genetic_programming import GeneticProgramming
from trading_bot.algorithms.pso import PSO
from trading_bot.backtester import Backtester, buy_and_hold
from trading_bot.data_loader import generate_synthetic_data
from trading_bot.strategy import VectorStrategy, GoldenCross, DeathCross

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def evaluate_strategy(strategy, prices, label):
    """Evaluate a strategy and print results."""
    signals = strategy.generate_signals(prices)
    bt = Backtester(prices)
    result = bt.evaluate(signals)
    print(f"  {label:20s}: fitness=${result.fitness:10.2f} | trades={result.n_trades:3d} | win_rate={result.win_rate:5.1f}% | sharpe={result.sharpe_ratio:6.3f}")
    return result


def main():
    print("=" * 80)
    print("TRADING BOT ALGORITHM DEMO")
    print("=" * 80)

    np.random.seed(42)
    dataset = generate_synthetic_data(n_train=500, n_val=100, n_test=200, seed=42)
    train_prices = dataset.train_prices
    test_prices = dataset.test_prices

    print(f"\nDataset: {len(train_prices)} train, {len(test_prices)} test points")
    print(f"Price range: ${train_prices.min():.2f} - ${train_prices.max():.2f}")

    print("\n" + "-" * 80)
    print("BASELINE STRATEGIES (Train Set)")
    print("-" * 80)

    gc = GoldenCross(fast_window=10, slow_window=30)
    dc = DeathCross(fast_window=10, slow_window=30)

    gc_train = evaluate_strategy(gc, train_prices, "GoldenCross(10,30)")
    dc_train = evaluate_strategy(dc, train_prices, "DeathCross(10,30)")

    bh_fitness = buy_and_hold(train_prices)
    print(f"  {'Buy-and-Hold':20s}: fitness=${bh_fitness:10.2f}")

    print("\n" + "-" * 80)
    print("PSO OPTIMIZATION (Dual-Crossover, 14D)")
    print("-" * 80)

    bounds = (
        [(0.0, 1.0)] * 3
        + [(2.0, 50.0)] * 3
        + [(0.01, 0.99)]
        + [(0.0, 1.0)] * 3
        + [(2.0, 50.0)] * 3
        + [(0.01, 0.99)]
    )

    def pso_fitness(params):
        strategy = VectorStrategy(params, "dual_crossover")
        signals = strategy.generate_signals(train_prices)
        return Backtester(train_prices).evaluate(signals).fitness

    print("Running PSO (30 particles, 50 iterations)...")
    pso = PSO(n_particles=30, max_iter=50, adaptive_inertia=True)
    pso_result = pso.optimize(pso_fitness, bounds)

    best_pso_strategy = VectorStrategy(pso_result.best, "dual_crossover")
    pso_train = evaluate_strategy(best_pso_strategy, train_prices, "PSO-optimized")
    pso_test = evaluate_strategy(best_pso_strategy, test_prices, "PSO-optimized (test)")

    print(f"  PSO convergence: {len(pso_result.history)} iterations")
    print(f"  PSO best fitness: ${pso_result.best_fitness:.2f}")
    print(f"  PSO best params: {np.round(pso_result.best, 2).tolist()}")

    print("\n" + "-" * 80)
    print("GP OPTIMIZATION (Tree-structured rules)")
    print("-" * 80)

    gp = GeneticProgramming(
        population_size=50,
        generations=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        max_depth=4,
        tournament_size=3,
    )

    def gp_fitness(tree):
        signals = gp.evaluate(tree, train_prices)
        return Backtester(train_prices).evaluate(signals).fitness

    print("Running GP (50 pop, 20 gen)...")
    gp_result = gp.optimize(gp_fitness, n_generations=20)

    best_gp_signals = gp.evaluate(gp_result.best, train_prices)
    gp_train_bt = Backtester(train_prices).evaluate(best_gp_signals)
    print(f"  {'GP-evolved':20s}: fitness=${gp_train_bt.fitness:10.2f} | trades={gp_train_bt.n_trades:3d} | win_rate={gp_train_bt.win_rate:5.1f}% | sharpe={gp_train_bt.sharpe_ratio:6.3f}")

    best_gp_signals_test = gp.evaluate(gp_result.best, test_prices)
    gp_test_bt = Backtester(test_prices).evaluate(best_gp_signals_test)
    print(f"  {'GP-evolved (test)':20s}: fitness=${gp_test_bt.fitness:10.2f} | trades={gp_test_bt.n_trades:3d} | win_rate={gp_test_bt.win_rate:5.1f}% | sharpe={gp_test_bt.sharpe_ratio:6.3f}")

    print(f"  GP convergence: {len(gp_result.history)} generations")
    print(f"  GP best fitness: ${gp_result.best_fitness:.2f}")
    print(f"  GP best tree: {gp_result.best}")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':25s} {'Train Fitness':>12s} {'Test Fitness':>12s} {'Trades':>8s}")
    print("-" * 80)
    print(f"{'GoldenCross':25s} {gc_train.fitness:12.2f} {'N/A':>12s} {gc_train.n_trades:8d}")
    print(f"{'Buy-and-Hold':25s} {bh_fitness:12.2f} {'N/A':>12s} {'0':>8s}")
    print(f"{'PSO-optimized':25s} {pso_train.fitness:12.2f} {pso_test.fitness:12.2f} {pso_train.n_trades:8d}")
    print(f"{'GP-evolved':25s} {gp_train_bt.fitness:12.2f} {gp_test_bt.fitness:12.2f} {gp_train_bt.n_trades:8d}")
    print("=" * 80)

    improvement_over_bh = ((pso_result.best_fitness / bh_fitness - 1) * 100) if bh_fitness > 0 else 0
    print(f"\nPSO improvement over Buy-and-Hold: {improvement_over_bh:+.1f}%")
    print(f"PSO improvement over GoldenCross: {((pso_result.best_fitness / gc_train.fitness - 1) * 100) if gc_train.fitness > 0 else 0:+.1f}%")


if __name__ == "__main__":
    main()
