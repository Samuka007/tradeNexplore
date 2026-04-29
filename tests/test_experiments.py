import numpy as np
import pytest

from trading_bot.algorithms.base import StubABC, StubGP, StubHarmonySearch, StubPSO
from trading_bot.data_loader import generate_synthetic_data
from trading_bot.experiments.runner import ExperimentRunner
from trading_bot.strategy import TreeStrategy


class TestStubOptimizers:
    def test_stub_pso_returns_result(self):
        pso = StubPSO(n_particles=10, max_iter=5)
        bounds = [(0.0, 1.0), (10.0, 20.0)]

        def fitness_fn(params):
            return -np.sum(params ** 2)

        result = pso.optimize(fitness_fn, bounds)
        assert result.best_fitness is not None
        assert len(result.best) == 2
        assert result.name == "StubPSO"

    def test_stub_abc_returns_result(self):
        abc = StubABC(n_bees=10, max_cycles=5)
        bounds = [(0.0, 1.0), (10.0, 20.0)]

        def fitness_fn(params):
            return -np.sum(params ** 2)

        result = abc.optimize(fitness_fn, bounds)
        assert result.best_fitness is not None
        assert len(result.best) == 2
        assert result.name == "StubABC"

    def test_stub_hs_returns_result(self):
        hs = StubHarmonySearch(hms=10, max_iter=5)
        bounds = [(0.0, 1.0), (10.0, 20.0)]

        def fitness_fn(params):
            return -np.sum(params ** 2)

        result = hs.optimize(fitness_fn, bounds)
        assert result.best_fitness is not None
        assert len(result.best) == 2
        assert result.name == "StubHarmonySearch"

    def test_stub_gp_returns_result(self):
        gp = StubGP(population_size=10, generations=5)

        def fitness_fn(tree):
            return 100.0 if tree.get("type") == "sma_crossover" else 0.0

        result = gp.optimize(fitness_fn, n_generations=5)
        assert result.best_fitness is not None
        assert result.name == "StubGP"


class TestExperimentRunner:
    @pytest.fixture
    def dataset(self):
        return generate_synthetic_data(n_train=200, n_val=50, n_test=100, seed=42)

    def test_run_continuous(self, dataset):
        runner = ExperimentRunner(dataset)
        pso = StubPSO()
        bounds = [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)] + [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]

        result = runner.run_continuous(pso, bounds, strategy_type="dual_crossover")

        assert result.algorithm_name == "StubPSO"
        assert result.strategy_type == "dual_crossover"
        assert result.train_fitness is not None
        assert result.test_fitness is not None
        assert result.backtest_result.n_trades >= 0

    def test_run_structural_requires_factory(self, dataset):
        runner = ExperimentRunner(dataset)
        gp = StubGP()

        with pytest.raises(ValueError, match="strategy_factory"):
            runner.run_structural(gp, strategy_factory=None)

    def test_run_structural_with_factory(self, dataset):
        runner = ExperimentRunner(dataset)
        gp = StubGP()

        def factory(tree):
            return TreeStrategy(tree)

        result = runner.run_structural(gp, strategy_factory=factory)

        assert result.algorithm_name == "StubGP"
        assert result.strategy_type == "gp_tree"
        assert result.train_fitness is not None

    def test_run_hierarchical_requires_extractor(self, dataset):
        runner = ExperimentRunner(dataset)
        gp = StubGP()
        pso = StubPSO()

        with pytest.raises(ValueError, match="template_extractor"):
            runner.run_hierarchical(
                gp,
                pso,
                strategy_factory=lambda t: TreeStrategy(t),
                template_extractor=None,
            )

    def test_run_hierarchical_with_extractor(self, dataset):
        runner = ExperimentRunner(dataset)
        gp = StubGP()
        pso = StubPSO()

        def factory(tree):
            return TreeStrategy(tree)

        def extractor(tree, prices):
            from trading_bot.strategy import VectorStrategy
            params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
            bounds = [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)] + [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
            return VectorStrategy(params, "dual_crossover"), bounds

        result = runner.run_hierarchical(gp, pso, strategy_factory=factory, template_extractor=extractor)

        assert "gp_result" in result
        assert "refined_result" in result
        assert "improvement" in result
