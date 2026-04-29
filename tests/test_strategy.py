import numpy as np
import pytest

from trading_bot.strategy import VectorStrategy


@pytest.fixture
def prices():
    np.random.seed(42)
    return np.cumsum(np.random.randn(200) * 10 + 100)


class TestVectorStrategy:
    def test_dual_crossover_signal_shape(self, prices):
        params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
        strategy = VectorStrategy(params, "dual_crossover")
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, np.ndarray)
        assert signals.dtype == np.int64
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_sma_crossover_generates_signals(self, prices):
        params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
        strategy = VectorStrategy(params, "dual_crossover")
        signals = strategy.generate_signals(prices)
        assert len(signals) > 0

    def test_param_clamping(self, prices):
        params = np.array([1, 0, 0, 1, 500, 1, 2.0, 1, 0, 0, 1, 500, 1, 2.0])
        strategy = VectorStrategy(params, "dual_crossover")
        assert np.all(strategy.high_d >= 2)
        assert np.all(strategy.high_d <= 200)
        assert np.all(strategy.low_d >= 2)
        assert np.all(strategy.low_d <= 200)
        assert 0.01 <= strategy.high_a <= 0.99
        assert 0.01 <= strategy.low_a <= 0.99

    def test_weight_normalization(self, prices):
        params = np.array([2, 4, 6, 10, 20, 30, 0.3, 1, 1, 1, 15, 25, 35, 0.3])
        strategy = VectorStrategy(params, "dual_crossover")
        assert np.isclose(strategy.high_w.sum(), 1.0)
        assert np.isclose(strategy.low_w.sum(), 1.0)

    def test_describe_returns_string(self):
        params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
        strategy = VectorStrategy(params, "dual_crossover")
        desc = strategy.describe()
        assert isinstance(desc, str)
        assert "DualCrossover" in desc
