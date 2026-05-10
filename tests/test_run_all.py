"""Tests for experiments/run_all.py — batch runner with file output (TDD)."""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestRunAllV2:
    """TDD: tests must fail before run_all.py is updated."""

    @pytest.fixture
    def dataset(self):
        from trading_bot.data_loader import generate_synthetic_data
        return generate_synthetic_data(n_train=80, n_val=20, n_test=40, seed=42)

    def test_module_importable(self):
        from trading_bot.experiments import run_all
        assert run_all is not None

    def test_run_experiment_1(self, dataset):
        from trading_bot.experiments.run_all import run_experiment_1
        results = run_experiment_1(dataset=dataset, n_runs=2, pso_particles=5, pso_iterations=3)
        for key in ["dual_crossover", "macd"]:
            r = results[key]
            assert isinstance(r["train_fitness"], list)
            assert len(r["train_fitness"]) == 2
            assert isinstance(r["test_fitness"], list)
            assert isinstance(r["convergence"], list)
            assert "best_params" in r
            assert "best_metrics" in r
            assert "best_equity_curve" in r
            assert len(r["best_equity_curve"]) > 0
            assert "best_trades" in r
            assert isinstance(r["best_trades"], list)

    def test_run_experiment_2(self, dataset):
        from trading_bot.experiments.run_all import run_experiment_2
        results = run_experiment_2(dataset=dataset, n_runs=2, gp_population=10, gp_generations=3, gp_max_depth=3)
        assert isinstance(results["train_fitness"], list)
        assert len(results["train_fitness"]) == 2
        assert isinstance(results["test_fitness"], list)
        assert isinstance(results["convergence"], list)
        assert "best_tree_repr" in results
        assert "best_metrics" in results
        assert "best_equity_curve" in results
        assert len(results["best_equity_curve"]) > 0
        assert "best_trades" in results
        assert isinstance(results["best_trades"], list)

    def test_run_experiment_3(self, dataset):
        """Exp 3: takes GP best tree from Exp 2, runs PSO refinement 30x on same template."""
        from trading_bot.algorithms.genetic_programming import GeneticProgramming
        from trading_bot.experiments.run_all import run_experiment_2, run_experiment_3

        # First run a quick GP to get a best tree (simulating Exp 2 output)
        exp2 = run_experiment_2(dataset=dataset, n_runs=1, gp_population=10, gp_generations=3, gp_max_depth=3)
        best_tree = exp2["best_tree"]

        # Re-create GP instance for template extraction
        gp = GeneticProgramming(population_size=10, max_depth=3)

        # Now run Exp 3: PSO refinement on the SAME tree, n=2 runs for speed
        results = run_experiment_3(
            dataset=dataset,
            gp=gp,
            best_tree=best_tree,
            n_runs=2,
            pso_particles=5,
            pso_iterations=3,
        )
        assert "gp_test_fitness" in results
        assert "gp_equity_curve" in results
        assert len(results["gp_equity_curve"]) > 0
        assert "gp_trades" in results
        assert isinstance(results["gp_trades"], list)
        assert "best_refined_equity_curve" in results
        assert "refined_test_fitness" in results
        assert "improvement" in results
        assert "improvements" in results
        assert len(results["improvements"]) == 2  # n_runs=2
        assert "tree_repr" in results

    def test_compute_baselines(self, dataset):
        from trading_bot.experiments.run_all import compute_baselines
        baselines = compute_baselines(dataset)
        for key in ["buy_and_hold", "golden_cross", "death_cross"]:
            assert key in baselines

    def test_save_and_load_results(self, dataset):
        from trading_bot.experiments.run_all import run_experiment_1, save_results, compute_baselines
        results = run_experiment_1(dataset=dataset, n_runs=2, pso_particles=5, pso_iterations=3)
        baselines = compute_baselines(dataset)
        with tempfile.TemporaryDirectory() as tmp:
            save_results(results, baselines, experiment=1, output_dir=tmp)
            fname = Path(tmp) / "experiment_1_results.json"
            assert fname.exists()
            loaded = json.loads(fname.read_text())
            assert loaded["experiment"] == 1
            assert "dual_crossover" in loaded["results"]
            assert "baselines" in loaded

    def test_full_results_structure(self, dataset):
        from trading_bot.experiments.run_all import run_experiment_1
        results = run_experiment_1(dataset=dataset, n_runs=2, pso_particles=5, pso_iterations=3)
        dc = results["dual_crossover"]
        for key in ["train_fitness", "test_fitness", "summary", "convergence"]:
            assert key in dc
        for key in ["n_trades", "win_rate", "sharpe_ratio"]:
            assert key in dc["best_metrics"]
