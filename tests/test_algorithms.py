import numpy as np
import pytest

from trading_bot.algorithms.genetic_programming import GeneticProgramming, GPNode
from trading_bot.algorithms.pso import PSO
from trading_bot.backtester import Backtester
from trading_bot.data_loader import generate_synthetic_data
from trading_bot.experiments.runner import ExperimentRunner
from trading_bot.strategy import VectorStrategy, GoldenCross, DeathCross


class TestGoldenCross:
    def test_generates_signals(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        gc = GoldenCross(fast_window=10, slow_window=30)
        signals = gc.generate_signals(prices)
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_describe(self):
        gc = GoldenCross(10, 30)
        assert "GoldenCross" in gc.describe()


class TestDeathCross:
    def test_generates_signals(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        dc = DeathCross(fast_window=10, slow_window=30)
        signals = dc.generate_signals(prices)
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))


class TestPSOIntegration:
    def test_pso_optimizes_dual_crossover(self):
        np.random.seed(42)
        dataset = generate_synthetic_data(n_train=100, n_val=20, n_test=50, seed=42)
        runner = ExperimentRunner(dataset)
        pso = PSO(n_particles=10, max_iter=10)
        bounds = [(0.0, 1.0)] * 3 + [(2.0, 20.0)] * 3 + [(0.01, 0.99)] + [(0.0, 1.0)] * 3 + [(2.0, 20.0)] * 3 + [(0.01, 0.99)]

        result = runner.run_continuous(pso, bounds, strategy_type="dual_crossover", evaluate_on_test=True)

        assert result.algorithm_name == "PSO"
        assert result.train_fitness > 0
        assert result.test_fitness is not None
        assert len(result.opt_result.history) > 0

    def test_pso_result_better_than_random(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        pso = PSO(n_particles=10, max_iter=10)
        bounds = [(0.0, 1.0)] * 3 + [(2.0, 20.0)] * 3 + [(0.01, 0.99)] + [(0.0, 1.0)] * 3 + [(2.0, 20.0)] * 3 + [(0.01, 0.99)]

        def fitness(params):
            strategy = VectorStrategy(params, "dual_crossover")
            signals = strategy.generate_signals(prices)
            return Backtester(prices).evaluate(signals).fitness

        pso_result = pso.optimize(fitness, bounds)

        random_fitness = []
        for _ in range(10):
            random_params = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
            random_fitness.append(fitness(random_params))

        assert pso_result.best_fitness >= np.mean(random_fitness)


class TestGPIntegration:
    def test_gp_evolution(self):
        np.random.seed(42)
        dataset = generate_synthetic_data(n_train=100, n_val=20, n_test=50, seed=42)
        gp = GeneticProgramming(population_size=20, generations=5, max_depth=3)

        def fitness_fn(tree):
            signals = gp.evaluate(tree, dataset.train_prices)
            return Backtester(dataset.train_prices).evaluate(signals).fitness

        result = gp.optimize(fitness_fn, n_generations=5)

        assert result.best_fitness > 0
        assert isinstance(result.best, GPNode)
        assert len(result.history) == 5

    def test_gp_generates_valid_signals(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        gp = GeneticProgramming(population_size=10, generations=3, max_depth=3)

        tree = gp._create_random_tree()
        signals = gp.evaluate(tree, prices)

        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_gp_better_than_hold(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        gp = GeneticProgramming(population_size=20, generations=5, max_depth=3)

        def fitness_fn(tree):
            signals = gp.evaluate(tree, prices)
            return Backtester(prices).evaluate(signals).fitness

        result = gp.optimize(fitness_fn, n_generations=5)

        hold_fitness = Backtester(prices).evaluate(np.zeros(len(prices))).fitness
        assert result.best_fitness >= hold_fitness
