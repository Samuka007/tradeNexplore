import numpy as np
import pytest

from trading_bot.backtester import Backtester, BacktestResult, Trade


class TestBacktesterBasics:
    def test_no_signals_no_trades(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = np.array([0, 0, 0, 0, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        assert result.final_cash == 1000.0
        assert result.n_trades == 0
        assert result.total_return_pct == 0.0

    def test_single_buy_and_hold(self):
        prices = np.array([100.0, 110.0, 120.0])
        signals = np.array([1, 0, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        expected_btc = 1000.0 * 0.97 / 100.0
        expected_cash = expected_btc * 120.0 * 0.97
        assert result.final_cash == pytest.approx(expected_cash, rel=1e-10)
        assert result.n_trades == 1

    def test_buy_sell_cycle(self):
        prices = np.array([100.0, 110.0, 105.0])
        signals = np.array([1, -1, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        buy_btc = 1000.0 * 0.97 / 100.0
        expected_cash = buy_btc * 110.0 * 0.97
        assert result.final_cash == pytest.approx(expected_cash, rel=1e-10)
        assert result.n_trades == 1

    def test_multiple_trades(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        signals = np.array([1, -1, 1, -1, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        assert result.n_trades == 2

    def test_forced_liquidation(self):
        prices = np.array([100.0, 110.0, 120.0])
        signals = np.array([1, 0, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        assert result.n_trades == 1
        assert result.final_cash > 0

    def test_signal_length_mismatch(self):
        prices = np.array([100.0, 101.0, 102.0])
        signals = np.array([1, -1])
        bt = Backtester(prices)
        with pytest.raises(ValueError, match="length"):
            bt.evaluate(signals)


class TestBacktesterMetrics:
    def test_sharpe_ratio_zero_for_flat(self):
        prices = np.array([100.0, 100.0, 100.0, 100.0])
        signals = np.array([0, 0, 0, 0])
        bt = Backtester(prices)
        result = bt.evaluate(signals)

        assert result.sharpe_ratio == 0.0

    def test_max_drawdown_negative(self):
        prices = np.array([100.0, 90.0, 80.0, 95.0, 85.0])
        signals = np.array([0, 0, 0, 0, 0])
        bt = Backtester(prices)
        result = bt.evaluate(signals)

        assert result.max_drawdown <= 0.0

    def test_win_rate_calculation(self):
        prices = np.array([100.0, 110.0, 105.0, 115.0, 108.0])
        signals = np.array([1, -1, 1, -1, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        assert 0.0 <= result.win_rate <= 100.0
        assert result.n_wins <= result.n_trades

    def test_equity_curve_length(self):
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = np.array([1, 0, 0, -1, 0])
        bt = Backtester(prices)
        result = bt.evaluate(signals)

        assert len(result.equity_curve) == len(prices)

    def test_trade_records_populated(self):
        prices = np.array([100.0, 110.0, 105.0])
        signals = np.array([1, -1, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_idx == 0
        assert trade.exit_idx == 1
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.pnl != 0.0


class TestBacktesterEdgeCases:
    def test_sell_without_buy_ignored(self):
        prices = np.array([100.0, 110.0, 120.0])
        signals = np.array([-1, -1, -1])
        bt = Backtester(prices, initial_cash=1000.0)
        result = bt.evaluate(signals)

        assert result.final_cash == 1000.0
        assert result.n_trades == 0

    def test_buy_after_buy_ignored(self):
        prices = np.array([100.0, 110.0, 120.0])
        signals = np.array([1, 1, 0])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        expected_btc = 1000.0 * 0.97 / 100.0
        expected_cash = expected_btc * 120.0 * 0.97
        assert result.final_cash == pytest.approx(expected_cash, rel=1e-10)
        assert result.n_trades == 1

    def test_zero_length_prices(self):
        prices = np.array([])
        signals = np.array([])
        bt = Backtester(prices, initial_cash=1000.0)
        result = bt.evaluate(signals)

        assert result.final_cash == 1000.0

    def test_single_price_point(self):
        prices = np.array([100.0])
        signals = np.array([1])
        bt = Backtester(prices, initial_cash=1000.0, fee_pct=0.03)
        result = bt.evaluate(signals)

        expected_btc = 1000.0 * 0.97 / 100.0
        expected_cash = expected_btc * 100.0 * 0.97
        assert result.final_cash == pytest.approx(expected_cash, rel=1e-10)
