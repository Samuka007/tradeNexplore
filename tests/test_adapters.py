"""Tests for GPAdapter — wraps GPNode tree for ExperimentRunner compatibility."""
import numpy as np
import pytest

from trading_bot.algorithms.genetic_programming import GeneticProgramming, GPNode


class TestGPAdapter:
    """Test the GPAdapter wrapper before it exists (TDD: RED phase)."""

    @pytest.fixture
    def gp(self):
        return GeneticProgramming(population_size=10, generations=3, max_depth=3)

    @pytest.fixture
    def tree(self, gp):
        return gp._create_random_tree()

    @pytest.fixture
    def adapter(self, gp, tree):
        from trading_bot.experiments.adapters import GPAdapter
        return GPAdapter(gp, tree)

    def test_has_generate_signals(self, adapter):
        """Adapter must expose generate_signals(prices) -> np.ndarray."""
        assert hasattr(adapter, "generate_signals")
        assert callable(adapter.generate_signals)

    def test_has_describe(self, adapter):
        """Adapter must expose describe() -> str."""
        assert hasattr(adapter, "describe")
        assert callable(adapter.describe)

    def test_generates_valid_signals(self, adapter):
        """Signals must be valid: only +1, -1, 0."""
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        signals = adapter.generate_signals(prices)
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_describe_returns_string(self, adapter):
        """describe() must return a non-empty string."""
        desc = adapter.describe()
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_works_with_run_structural(self, gp, tree):
        """Adapter must integrate with ExperimentRunner.run_structural()."""
        from trading_bot.experiments.adapters import GPAdapter
        from trading_bot.experiments.runner import ExperimentRunner
        from trading_bot.data_loader import generate_synthetic_data

        ds = generate_synthetic_data(n_train=50, n_val=20, n_test=30, seed=42)
        runner = ExperimentRunner(ds)

        adapter_instance = GPAdapter(gp, tree)
        strategy_factory = lambda t: GPAdapter(gp, t)

        r = runner.run_structural(gp, strategy_factory, evaluate_on_test=True)
        assert r.train_fitness > 0
        assert r.test_fitness is not None
        assert len(r.opt_result.history) > 0
