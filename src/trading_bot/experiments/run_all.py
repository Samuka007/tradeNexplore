"""Batch experiment runner — PSO vs GP comparison on BTC data.

Usage:
    python -m trading_bot.experiments.run_all

See docs/EXPERIMENT_PLAN.md for full design.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

import numpy as np

from trading_bot.algorithms.genetic_programming import GeneticProgramming
from trading_bot.algorithms.pso import PSO
from trading_bot.backtester import Backtester, buy_and_hold
from trading_bot.data_loader import Dataset
from trading_bot.experiments.adapters import GPAdapter
from trading_bot.experiments.runner import ExperimentRunner
from trading_bot.strategy import DeathCross, GoldenCross, VectorStrategy

logger = logging.getLogger(__name__)


def compute_baselines(dataset: Dataset) -> dict:
    """Compute baseline strategy fitnesses on train and test sets.

    Returns:
        dict with keys 'buy_and_hold', 'golden_cross', 'death_cross'.
        Each value: dict with 'train_fitness', 'test_fitness',
        'n_trades', 'win_rate'.
    """
    train = dataset.train_prices
    test = dataset.test_prices

    results = {}

    # Buy-and-Hold
    bh_train = buy_and_hold(train)
    bh_test = buy_and_hold(test)
    results["buy_and_hold"] = {
        "train_fitness": bh_train,
        "test_fitness": bh_test,
        "n_trades": 0,
        "win_rate": 0.0,
    }

    # GoldenCross
    for name, strategy_cls in [("golden_cross", GoldenCross), ("death_cross", DeathCross)]:
        s = strategy_cls(50, 200)
        sig_train = s.generate_signals(train)
        sig_test = s.generate_signals(test)
        bt_train = Backtester(train).evaluate(sig_train)
        bt_test = Backtester(test).evaluate(sig_test)
        results[name] = {
            "train_fitness": bt_train.fitness,
            "test_fitness": bt_test.fitness,
            "n_trades": bt_test.n_trades,
            "win_rate": bt_test.win_rate,
        }

    return results


def run_experiment_1(
    dataset: Dataset,
    n_runs: int = 10,
    pso_particles: int = 30,
    pso_iterations: int = 50,
) -> dict:
    """Experiment 1: PSO parameter optimization on dual_crossover and MACD.

    Args:
        dataset: Dataset with train/val/test split.
        n_runs: Number of independent PSO runs per strategy.
        pso_particles: PSO swarm size.
        pso_iterations: PSO generation count.

    Returns:
        dict: {
            'dual_crossover': {
                'train_fitness': [N floats],
                'test_fitness': [N floats],
                'convergence': [N lists],
                'best_params': np.ndarray,
            },
            'macd': { ... same ... },
        }
    """
    runner = ExperimentRunner(dataset)
    results: dict = {}

    configs = {
        "dual_crossover": {
            "bounds": (
                [(0.0, 1.0)] * 3
                + [(2.0, 200.0)] * 3
                + [(0.01, 0.99)]
                + [(0.0, 1.0)] * 3
                + [(2.0, 200.0)] * 3
                + [(0.01, 0.99)]
            ),
            "strategy_type": "dual_crossover",
        },
        "macd": {
            "bounds": (
                [(2.0, 200.0)]  # d1
                + [(0.01, 0.99)]  # a1
                + [(2.0, 200.0)]  # d2
                + [(0.01, 0.99)]  # a2
                + [(2.0, 200.0)]  # d3
                + [(0.01, 0.99)]  # a3
                + [(0.0, 1.0)]  # threshold
            ),
            "strategy_type": "macd",
        },
    }

    for name, cfg in configs.items():
        train_fits = []
        test_fits = []
        convergences = []
        best_runs = []

        for run_i in range(n_runs):
            pso = PSO(
                n_particles=pso_particles,
                max_iter=pso_iterations,
                adaptive_inertia=True,
            )
            r = runner.run_continuous(
                pso,
                cfg["bounds"],
                strategy_type=cfg["strategy_type"],
                evaluate_on_test=True,
            )
            train_fits.append(r.train_fitness)
            test_fits.append(r.test_fitness)
            convergences.append(r.opt_result.history)
            best_runs.append(r)

        best_idx = int(np.argmax(test_fits))
        best_run = best_runs[best_idx]
        bt = best_run.backtest_result
        test_bt = best_run.metadata.get("test_backtest")

        tfs = np.array(test_fits, dtype=np.float64)

        results[name] = {
            "train_fitness": train_fits,
            "test_fitness": test_fits,
            "convergence": convergences,
            "best_params": best_run.opt_result.best,
            "best_train_fitness": best_run.train_fitness,
            "best_test_fitness": best_run.test_fitness,
            "best_metrics": {
                "n_trades": bt.n_trades,
                "win_rate": bt.win_rate,
                "sharpe_ratio": bt.sharpe_ratio,
                "max_drawdown": bt.max_drawdown,
                "total_return_pct": bt.total_return_pct,
            },
            "best_equity_curve": test_bt.equity_curve if test_bt is not None else [],
            "best_trades": [dataclasses.asdict(t) for t in test_bt.trades] if test_bt is not None else [],
            "summary": {
                "train_mean": float(np.mean(train_fits)),
                "train_std": float(np.std(train_fits)),
                "test_mean": float(np.mean(tfs)),
                "test_std": float(np.std(tfs)),
                "test_min": float(np.min(tfs)),
                "test_max": float(np.max(tfs)),
            },
        }

    return results


def run_experiment_2(
    dataset: Dataset,
    n_runs: int = 10,
    gp_population: int = 100,
    gp_generations: int = 20,
    gp_max_depth: int = 3,
    gp_mutation: float = 0.2,
) -> dict:
    """Experiment 2: GP structure discovery.

    Args:
        dataset: Dataset with train/val/test split.
        n_runs: Number of independent GP runs.
        gp_population: GP population size.
        gp_generations: GP generation count.
        gp_max_depth: Maximum tree depth.
        gp_mutation: Mutation rate (higher = more exploration diversity).
    """
    runner = ExperimentRunner(dataset)
    train_fits = []
    test_fits = []
    convergences = []
    best_runs = []

    for run_i in range(n_runs):
        gp = GeneticProgramming(
            population_size=gp_population,
            generations=gp_generations,
            max_depth=gp_max_depth,
            mutation_rate=gp_mutation,
        )

        def strategy_factory(tree):
            return GPAdapter(gp, tree)

        r = runner.run_structural(gp, strategy_factory, evaluate_on_test=True)
        train_fits.append(r.train_fitness)
        test_fits.append(r.test_fitness)
        convergences.append(r.opt_result.history)
        best_runs.append(r)

    best_idx = int(np.argmax(test_fits))
    best_run = best_runs[best_idx]
    bt = best_run.backtest_result
    test_bt = best_run.metadata.get("test_backtest")

    tfs = np.array(test_fits, dtype=np.float64)

    return {
        "train_fitness": train_fits,
        "test_fitness": test_fits,
        "convergence": convergences,
        "best_tree": best_run.opt_result.best,
        "best_tree_repr": repr(best_run.opt_result.best),
        "best_train_fitness": best_run.train_fitness,
        "best_test_fitness": best_run.test_fitness,
        "best_metrics": {
            "n_trades": bt.n_trades,
            "win_rate": bt.win_rate,
            "sharpe_ratio": bt.sharpe_ratio,
            "max_drawdown": bt.max_drawdown,
            "total_return_pct": bt.total_return_pct,
        },
        "best_equity_curve": test_bt.equity_curve if test_bt is not None else [],
        "best_trades": [dataclasses.asdict(t) for t in test_bt.trades] if test_bt is not None else [],
        "summary": {
            "train_mean": float(np.mean(train_fits)),
            "train_std": float(np.std(train_fits)),
            "test_mean": float(np.mean(tfs)),
            "test_std": float(np.std(tfs)),
            "test_min": float(np.min(tfs)),
            "test_max": float(np.max(tfs)),
        },
    }


def run_experiment_3(
    dataset: Dataset,
    gp: GeneticProgramming,
    best_tree: object,
    n_runs: int = 30,
    pso_particles: int = 30,
    pso_iterations: int = 50,
) -> dict:
    """Experiment 3: PSO refines the best GP-discovered structure.

    Takes the single best tree from Experiment 2 and runs PSO
    n_runs times on the SAME extracted template. This ensures
    statistical significance while maintaining context continuity.

    Args:
        dataset: Dataset with train/val/test split.
        gp: GP instance (used only for _collect_terminals).
        best_tree: The single best tree from Experiment 2.
        n_runs: Number of independent PSO refinement runs.
        pso_particles, pso_iterations: PSO config.

    Returns:
        dict with gp_test_fitness, refined test fitness stats,
        per-run improvements, and tree representation.
    """
    from trading_bot.experiments.template_extract import extract_dual_crossover_template

    # Extract the SAME template once (durations locked from GP tree)
    template_strategy, bounds = extract_dual_crossover_template(gp, best_tree, dataset.train_prices)

    # GP baseline on test set (computed once)
    gp_test_sig = gp.evaluate(best_tree, dataset.test_prices)
    gp_test_bt = Backtester(dataset.test_prices).evaluate(gp_test_sig)
    gp_test_fitness = gp_test_bt.fitness

    improvements = []
    refined_fitnesses = []
    best_refined_equity = None
    best_refined_trades = None
    best_refined_bt = None
    best_refined_fitness = -float("inf")

    for run_i in range(n_runs):
        pso = PSO(n_particles=pso_particles, max_iter=pso_iterations, adaptive_inertia=True)

        def refined_fitness(params):
            s = VectorStrategy(params, template_strategy.type)
            sig = s.generate_signals(dataset.train_prices)
            return Backtester(dataset.train_prices).evaluate(sig).fitness

        pso_result = pso.optimize(refined_fitness, bounds)

        # Evaluate refined strategy on test set
        refined_strategy = VectorStrategy(pso_result.best, template_strategy.type)
        refined_test_sig = refined_strategy.generate_signals(dataset.test_prices)
        refined_test_bt = Backtester(dataset.test_prices).evaluate(refined_test_sig)

        imp = refined_test_bt.fitness - gp_test_fitness
        improvements.append(imp)
        refined_fitnesses.append(refined_test_bt.fitness)

        if refined_test_bt.fitness > best_refined_fitness:
            best_refined_fitness = refined_test_bt.fitness
            best_refined_equity = refined_test_bt.equity_curve
            best_refined_trades = refined_test_bt.trades
            best_refined_bt = refined_test_bt

    imps = np.array(improvements, dtype=np.float64)
    rfs = np.array(refined_fitnesses, dtype=np.float64)

    return {
        "tree_repr": repr(best_tree),
        "gp_test_fitness": gp_test_fitness,
        "gp_equity_curve": gp_test_bt.equity_curve,
        "gp_trades": [dataclasses.asdict(t) for t in gp_test_bt.trades],
        "refined_test_fitness": {
            "mean": float(rfs.mean()),
            "std": float(rfs.std()),
            "min": float(rfs.min()),
            "max": float(rfs.max()),
        },
        "improvement": {
            "mean": float(imps.mean()),
            "std": float(imps.std()),
            "min": float(imps.min()),
            "max": float(imps.max()),
            "mean_pct": float(imps.mean() / gp_test_fitness * 100) if gp_test_fitness != 0 else 0.0,
        },
        "improvements": [float(i) for i in improvements],
        "best_refined_equity_curve": best_refined_equity,
        "best_refined_trades": [dataclasses.asdict(t) for t in best_refined_trades] if best_refined_trades else [],
        "best_refined_metrics": {
            "n_trades": best_refined_bt.n_trades,
            "win_rate": best_refined_bt.win_rate,
            "sharpe_ratio": best_refined_bt.sharpe_ratio,
        } if best_refined_bt is not None else {},
        "gp_metrics": {
            "n_trades": gp_test_bt.n_trades,
            "win_rate": gp_test_bt.win_rate,
            "sharpe_ratio": gp_test_bt.sharpe_ratio,
        },
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------
def save_results(
    results: dict,
    baselines: dict,
    experiment: int,
    output_dir: str = "results",
) -> str:
    """Save experiment results and baselines to a JSON file.

    Args:
        results: Results dict from run_experiment_1/2/3.
        baselines: Baselines dict from compute_baselines.
        experiment: Experiment number (1, 2, or 3).
        output_dir: Directory to save results.

    Returns:
        Path to the saved file.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays and non-serializable objects for JSON
    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        # GPNode / non-serializable objects → repr string
        return repr(obj)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment,
        "baselines": baselines,
        "results": results,
    }

    fname = out / f"experiment_{experiment}_results.json"
    fname.write_text(json.dumps(payload, indent=2, default=_serialize))

    logger.info("Results saved to %s", fname)
    return str(fname)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Run experiments and save results to JSON files.

    Usage:
        python -m trading_bot.experiments.run_all              # all experiments
        python -m trading_bot.experiments.run_all --exp=1      # Exp 1 only
        python -m trading_bot.experiments.run_all --exp=1 --runs-pso=30 --runs-gp=10
    """
    import sys

    from trading_bot.data_loader import load_or_generate_data

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    use_synthetic = "--synthetic" in sys.argv
    exp_filter = None
    runs_pso = 30
    runs_gp = 10
    runs_pso_refine = 30
    for arg in sys.argv:
        if arg.startswith("--exp="):
            exp_filter = arg.split("=")[1]
        elif arg.startswith("--experiment="):
            exp_filter = arg.split("=")[1]
        elif arg.startswith("--runs-pso="):
            runs_pso = int(arg.split("=")[1])
        elif arg.startswith("--runs-gp="):
            runs_gp = int(arg.split("=")[1])
        elif arg.startswith("--runs-refine="):
            runs_pso_refine = int(arg.split("=")[1])

    ds = load_or_generate_data(
        "data/btc_daily_2014_2022.csv",
        fallback_to_synthetic=use_synthetic,
    )

    print("=" * 60)
    print(f"EXPERIMENTS: PSO ({runs_pso} runs) vs GP ({runs_gp} runs)")
    print(f"Data: {len(ds.train_prices)} train / {len(ds.val_prices)} val / {len(ds.test_prices)} test")
    print("=" * 60)

    baselines = compute_baselines(ds)

    # Experiment 1: PSO
    if exp_filter is None or exp_filter == "1":
        print("\n--- Experiment 1: PSO ---")
        results = run_experiment_1(ds, n_runs=runs_pso)
        save_results(results, baselines, experiment=1)
        for name, r in results.items():
            s = r["summary"]
            print(f"  {name}: test ${s['test_mean']:,.0f} +- ${s['test_std']:,.0f}  "
                  f"[{s['test_min']:,.0f}..{s['test_max']:,.0f}]")

    # Experiment 2: GP
    exp2_cache = None  # (results_dict, best_tree) — cached for Exp3 reuse
    if exp_filter is None or exp_filter == "2":
        print("\n--- Experiment 2: GP ---")
        results = run_experiment_2(ds, n_runs=runs_gp)
        save_results(results, baselines, experiment=2)
        s = results["summary"]
        print(f"  GP: test ${s['test_mean']:,.0f} +- ${s['test_std']:,.0f}  "
              f"[{s['test_min']:,.0f}..{s['test_max']:,.0f}]")
        print(f"  best tree: {results['best_tree_repr']}")
        exp2_cache = (results, results["best_tree"])

    # Experiment 3: Hierarchical (needs Exp 2 best tree)
    if exp_filter is None or exp_filter == "3":
        print("\n--- Experiment 3: PSO Refinement of Best GP Structure ---")

        if exp2_cache is None:
            # Exp2 was NOT run in this session (e.g. --exp=3 alone).
            # Run it now with the configured params to get a tree.
            print("  (Running Exp 2 first to obtain best GP structure...)")
            exp2_results = run_experiment_2(ds, n_runs=runs_gp)
            save_results(exp2_results, baselines, experiment=2)
            best_tree = exp2_results["best_tree"]
            s2 = exp2_results["summary"]
            print(f"  GP baseline: test ${s2['test_mean']:,.0f} +- ${s2['test_std']:,.0f}")
            print(f"  best tree: {exp2_results['best_tree_repr']}")
        else:
            # Reuse the tree from the Exp2 block above — no re-run.
            exp2_results, best_tree = exp2_cache
            print(f"  Using cached GP structure: {exp2_results['best_tree_repr']}")

        gp = GeneticProgramming(population_size=100, max_depth=3)

        print(f"\n  Running PSO refinement ({runs_pso_refine} runs on same template)...")
        results = run_experiment_3(
            dataset=ds,
            gp=gp,
            best_tree=best_tree,
            n_runs=runs_pso_refine,
            pso_particles=30,
            pso_iterations=50,
        )
        save_results(results, baselines, experiment=3)

        imp = results["improvement"]
        print(f"  GP test fitness:          ${results['gp_test_fitness']:,.0f}")
        print(f"  PSO refined test:  mean=${results['refined_test_fitness']['mean']:,.0f}  "
              f"std=${results['refined_test_fitness']['std']:,.0f}")
        print(f"  Improvement:        mean=${imp['mean']:,.0f}  "
              f"std=${imp['std']:,.0f}  ({imp['mean_pct']:+.1f}%)")
        print(f"  Per-run:            {[round(i, 0) for i in results['improvements']]}")

    print(f"\nResults saved to results/experiment_*_results.json")


if __name__ == "__main__":
    main()
