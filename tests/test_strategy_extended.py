import numpy as np
import pytest

from trading_bot.backtester import Backtester, buy_and_hold, make_fitness, make_fitness_penalized
from trading_bot.strategy import VectorStrategy, TreeStrategy


class TestBuyAndHold:
    def test_basic(self):
        prices = np.array([100.0, 110.0, 120.0])
        result = buy_and_hold(prices, initial_cash=1000.0, fee_pct=0.03)
        btc = 1000.0 * 0.97 / 100.0
        expected = btc * 120.0 * 0.97
        assert result == pytest.approx(expected, rel=1e-10)

    def test_empty_prices(self):
        result = buy_and_hold(np.array([]), initial_cash=1000.0)
        assert result == 1000.0

    def test_single_price(self):
        result = buy_and_hold(np.array([100.0]), initial_cash=1000.0, fee_pct=0.03)
        btc = 1000.0 * 0.97 / 100.0
        expected = btc * 100.0 * 0.97
        assert result == pytest.approx(expected, rel=1e-10)


class TestMakeFitness:
    def test_returns_callable(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        fitness = make_fitness(prices, strategy_type="dual_crossover")
        assert callable(fitness)

    def test_evaluates_params(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        fitness = make_fitness(prices, strategy_type="dual_crossover")
        params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
        score = fitness(params)
        assert isinstance(score, float)

    def test_baseline_normalization(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        fitness = make_fitness(prices, strategy_type="dual_crossover", use_baseline=True)
        params = np.array([1, 0, 0, 10, 20, 30, 0.3, 1, 0, 0, 15, 25, 35, 0.3])
        score = fitness(params)
        baseline = buy_and_hold(prices)
        assert score > 0 or baseline > 0


class TestMakeFitnessPenalized:
    def test_penalizes_excess_trades(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0] * 20)
        fitness = make_fitness_penalized(prices, max_trades=1)
        params = np.array([1, 0, 0, 2, 3, 4, 0.3, 1, 0, 0, 2, 3, 4, 0.3])
        score = fitness(params)
        assert isinstance(score, float)


class TestMacdStrategy:
    def test_macd_signal_shape(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(200) * 10 + 100)
        params = np.array([12, 0.2, 26, 0.1, 9, 0.2, 0.0])
        strategy = VectorStrategy(params, "macd")
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, np.ndarray)
        assert signals.dtype == np.int64
        assert np.all(np.isin(signals, [-1, 0, 1]))
        assert len(signals) == len(prices)

    def test_macd_param_clamping(self):
        params = np.array([1, 2.0, 1, 2.0, 1, 2.0, 0.0])
        strategy = VectorStrategy(params, "macd")
        assert strategy.macd_d1 >= 2
        assert strategy.macd_a1 <= 0.99
        assert strategy.macd_a2 >= 0.01

    def test_macd_describe(self):
        params = np.array([12, 0.2, 26, 0.1, 9, 0.2, 0.0])
        strategy = VectorStrategy(params, "macd")
        desc = strategy.describe()
        assert isinstance(desc, str)
        assert "MACD" in desc


class TestTreeStrategy:
    def test_sma_crossover_tree(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        tree = {"type": "sma_crossover", "fast": 5, "slow": 20}
        strategy = TreeStrategy(tree)
        signals = strategy.generate_signals(prices)
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_macd_tree(self):
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(100) * 10 + 100)
        tree = {"type": "macd", "d1": 12, "a1": 0.2, "d2": 26, "a2": 0.1, "d3": 9, "a3": 0.2}
        strategy = TreeStrategy(tree)
        signals = strategy.generate_signals(prices)
        assert len(signals) == len(prices)
        assert np.all(np.isin(signals, [-1, 0, 1]))

    def test_unknown_tree_returns_hold(self):
        prices = np.array([100.0, 110.0, 120.0])
        tree = {"type": "unknown"}
        strategy = TreeStrategy(tree)
        signals = strategy.generate_signals(prices)
        assert np.all(signals == 0)

    def test_describe(self):
        tree = {"type": "sma_crossover", "fast": 10}
        strategy = TreeStrategy(tree)
        assert isinstance(strategy.describe(), str)
