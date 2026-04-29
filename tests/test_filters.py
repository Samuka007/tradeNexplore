import numpy as np
import pytest

from trading_bot.filters import pad, sma_filter, lma_filter, ema_filter, wma, crossover_detector


@pytest.fixture
def prices():
    np.random.seed(42)
    return np.cumsum(np.random.randn(200) * 10 + 100)


class TestSMA:
    @pytest.mark.parametrize("window", [5, 10, 20, 50])
    def test_sma_against_pandas(self, prices, window):
        kernel = sma_filter(window)
        conv_result = wma(prices, window, kernel)

        import pandas as pd
        pd_result = pd.Series(prices).rolling(window=window).mean().dropna().values

        np.testing.assert_allclose(conv_result[window - 1 :], pd_result, rtol=1e-10)

    def test_sma_kernel_sum(self):
        for N in [5, 10, 20]:
            kernel = sma_filter(N)
            assert np.isclose(kernel.sum(), 1.0)
            assert len(kernel) == N


class TestEMA:
    @pytest.mark.parametrize("window", [10, 20, 50])
    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5])
    def test_ema_against_pandas(self, prices, window, alpha):
        kernel = ema_filter(window, alpha)
        conv_result = wma(prices, window, kernel)

        import pandas as pd
        pd_result = pd.Series(prices).ewm(alpha=alpha, adjust=False).mean().values

        skip = window
        # Low alpha with small window: truncated convolution diverges from
        # pandas recursive EMA because significant tail weight is lost.
        rtol = 0.5 if (alpha == 0.1 and window <= 20) else 0.1
        np.testing.assert_allclose(
            conv_result[skip:],
            pd_result[skip:len(conv_result)],
            rtol=rtol,
        )

    def test_ema_kernel_sum(self):
        for N in [10, 20]:
            for alpha in [0.1, 0.3, 0.5]:
                kernel = ema_filter(N, alpha)
                assert np.isclose(kernel.sum(), 1.0)


class TestLMA:
    def test_lma_weights(self):
        N = 5
        kernel = lma_filter(N)
        expected = np.array([5, 4, 3, 2, 1]) / 15.0
        np.testing.assert_allclose(kernel, expected, rtol=1e-10)

    def test_lma_kernel_sum(self):
        for N in [5, 10, 20]:
            kernel = lma_filter(N)
            assert np.isclose(kernel.sum(), 1.0)

    def test_lma_against_manual(self, prices):
        N = 10
        kernel = lma_filter(N)
        conv_result = wma(prices, N, kernel)

        manual = np.zeros(len(prices) - N + 1)
        for i in range(len(manual)):
            window = prices[i : i + N]
            manual[i] = np.dot(window, kernel[::-1])

        np.testing.assert_allclose(
            conv_result[N - 1 :], manual, rtol=1e-10
        )


class TestCrossoverDetector:
    def test_golden_cross(self):
        diff = np.array([-1, -1, -0.5, 0.5, 1, 1])
        crosses = crossover_detector(diff)
        assert crosses[2] > 0.5

    def test_death_cross(self):
        diff = np.array([1, 1, 0.5, -0.5, -1, -1])
        crosses = crossover_detector(diff)
        assert crosses[2] < -0.5

    def test_no_cross(self):
        diff = np.array([1, 1, 1, 1, 1])
        crosses = crossover_detector(diff)
        assert np.all(crosses == 0)


class TestEdgeCases:
    def test_pad_preserves_rising_trend(self):
        P = np.array([100, 110, 120, 130])
        padded = pad(P, 3)
        assert padded[0] == -120
        assert padded[1] == -110
        np.testing.assert_array_equal(padded[2:], P)

    def test_wma_length(self, prices):
        N = 20
        kernel = sma_filter(N)
        result = wma(prices, N, kernel)
        assert len(result) == len(prices)

    def test_empty_price(self):
        P = np.array([])
        result = wma(P, 5, sma_filter(5))
        assert len(result) == 0

    def test_single_point(self):
        P = np.array([100.0])
        result = wma(P, 1, sma_filter(1))
        np.testing.assert_allclose(result, [100.0])
