"""Trading strategy implementations."""

import numpy as np
from typing import Protocol

from trading_bot.filters import wma, sma_filter, lma_filter, ema_filter, crossover_detector


class Strategy(Protocol):
    """Any object that can generate trading signals from price data."""

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """Return signal array: +1=buy, -1=sell, 0=hold."""
        ...

    def describe(self) -> str:
        ...


class VectorStrategy:
    """Fixed-structure strategy parameterised by a continuous vector.

    Supports:
    - dual_crossover (14D): [w1,w2,w3,d1,d2,d3,a3,w4,w5,w6,d4,d5,d6,a6]
    - macd (7D): [d1,a1,d2,a2,d3,a3,threshold]
    """

    def __init__(self, params: np.ndarray, strategy_type: str = "dual_crossover"):
        self.params = params
        self.type = strategy_type
        self._unpack()

    def _unpack(self):
        p = self.params
        if self.type == "dual_crossover":
            self.high_w = self._normalize_weights(p[0:3])
            self.high_d = np.clip(np.round(p[3:6]), 2, 200).astype(int)
            self.high_a = np.clip(p[6], 0.01, 0.99)
            self.low_w = self._normalize_weights(p[7:10])
            self.low_d = np.clip(np.round(p[10:13]), 2, 200).astype(int)
            self.low_a = np.clip(p[13], 0.01, 0.99)
        elif self.type == "macd":
            self.macd_d1 = max(2, min(int(np.round(p[0])), 200))
            self.macd_a1 = np.clip(p[1], 0.01, 0.99)
            self.macd_d2 = max(2, min(int(np.round(p[2])), 200))
            self.macd_a2 = np.clip(p[3], 0.01, 0.99)
            self.signal_d = max(2, min(int(np.round(p[4])), 200))
            self.signal_a = np.clip(p[5], 0.01, 0.99)
            self.threshold = p[6] if len(p) > 6 else 0.0
        else:
            raise ValueError(f"Unknown strategy type: {self.type}")

    @staticmethod
    def _normalize_weights(weights: np.ndarray) -> np.ndarray:
        total = weights.sum()
        return weights / (total + 1e-10)

    def _component_signal(self, prices: np.ndarray, weights, durations, alpha) -> np.ndarray:
        """Compute a weighted sum of SMA/LMA/EMA signals."""
        signals = []
        for i, (wi, di) in enumerate(zip(weights, durations)):
            di = min(int(di), len(prices) - 1)
            di = max(di, 2)
            if i == 0:
                sig = wma(prices, di, sma_filter(di))
            elif i == 1:
                sig = wma(prices, di, lma_filter(di))
            else:
                sig = wma(prices, di, ema_filter(di, alpha))
            signals.append(wi * sig)
        min_len = min(len(s) for s in signals)
        aligned = [s[-min_len:] for s in signals]
        return np.sum(aligned, axis=0)

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        if self.type == "dual_crossover":
            return self._generate_dual_crossover(prices)
        elif self.type == "macd":
            return self._generate_macd(prices)
        else:
            raise ValueError(f"Unknown strategy type: {self.type}")

    def _generate_dual_crossover(self, prices: np.ndarray) -> np.ndarray:
        high = self._component_signal(prices, self.high_w, self.high_d, self.high_a)
        low = self._component_signal(prices, self.low_w, self.low_d, self.low_a)
        diff = high - low
        crosses = crossover_detector(diff)

        signals = np.zeros(len(crosses), dtype=int)
        signals[crosses > 0.5] = 1
        signals[crosses < -0.5] = -1

        if len(signals) < len(prices):
            length_diff = len(prices) - len(signals)
            signals = np.pad(signals, (length_diff, 0), mode="constant", constant_values=0)

        return signals

    def _generate_macd(self, prices: np.ndarray) -> np.ndarray:
        """Generate signals using MACD crossover strategy.

        MACD = EMA(d1, a1) - EMA(d2, a2)
        Signal = EMA(MACD, signal_d, signal_a)
        Trigger = Sign(MACD - Signal) when |MACD - Signal| > threshold
        """
        ema_fast = wma(prices, self.macd_d1, ema_filter(self.macd_d1, self.macd_a1))
        ema_slow = wma(prices, self.macd_d2, ema_filter(self.macd_d2, self.macd_a2))
        macd_line = ema_fast - ema_slow
        signal_line = wma(macd_line, self.signal_d, ema_filter(self.signal_d, self.signal_a))

        min_len = min(len(macd_line), len(signal_line))
        macd_aligned = macd_line[-min_len:]
        signal_aligned = signal_line[-min_len:]
        diff = macd_aligned - signal_aligned

        if self.threshold > 0:
            diff = np.where(np.abs(diff) > self.threshold, diff, 0.0)

        crosses = crossover_detector(diff)
        signals = np.zeros(len(crosses), dtype=int)
        signals[crosses > 0.5] = 1
        signals[crosses < -0.5] = -1

        if len(signals) < len(prices):
            length_diff = len(prices) - len(signals)
            signals = np.pad(signals, (length_diff, 0), mode="constant", constant_values=0)

        return signals

    def describe(self) -> str:
        if self.type == "dual_crossover":
            return (
                f"DualCrossover("
                f"high_d={self.high_d.tolist()}, low_d={self.low_d.tolist()}"
                f")"
            )
        elif self.type == "macd":
            return (
                f"MACD("
                f"fast_d={self.macd_d1}, fast_a={self.macd_a1:.2f}, "
                f"slow_d={self.macd_d2}, slow_a={self.macd_a2:.2f}, "
                f"sig_d={self.signal_d}, sig_a={self.signal_a:.2f}"
                f")"
            )
        return f"VectorStrategy(type={self.type})"


class TreeStrategy:
    """GP tree-based strategy.

    Wraps a tree structure (dict/list/Node) and evaluates it against
    price data to generate trading signals.
    """

    def __init__(self, tree: object):
        self.tree = tree

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        """Evaluate tree on prices and return signals.

        For now, implements a simple rule-based fallback:
        - If tree is dict with 'type'=='sma_crossover': SMA crossover
        - Otherwise: return all-hold (placeholder until full GP eval)
        """
        if isinstance(self.tree, dict):
            if self.tree.get("type") == "sma_crossover":
                return self._sma_crossover(
                    prices,
                    self.tree.get("fast", 10),
                    self.tree.get("slow", 30),
                )
            elif self.tree.get("type") == "macd":
                params = np.array([
                    self.tree.get("d1", 12),
                    self.tree.get("a1", 0.2),
                    self.tree.get("d2", 26),
                    self.tree.get("a2", 0.1),
                    self.tree.get("d3", 9),
                    self.tree.get("a3", 0.2),
                    0.0,
                ], dtype=np.float64)
                return VectorStrategy(params, "macd").generate_signals(prices)
        return np.zeros(len(prices), dtype=int)

    def _sma_crossover(self, prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
        fast = max(2, min(int(fast), len(prices) - 1))
        slow = max(fast + 1, min(int(slow), len(prices) - 1))
        sma_fast = wma(prices, fast, sma_filter(fast))
        sma_slow = wma(prices, slow, sma_filter(slow))
        diff = sma_fast[-len(sma_slow):] - sma_slow
        crosses = crossover_detector(diff)
        signals = np.zeros(len(crosses), dtype=int)
        signals[crosses > 0.5] = 1
        signals[crosses < -0.5] = -1
        if len(signals) < len(prices):
            length_diff = len(prices) - len(signals)
            signals = np.pad(signals, (length_diff, 0), mode="constant", constant_values=0)
        return signals

    def describe(self) -> str:
        return f"TreeStrategy(tree={self.tree})"


class GoldenCross:
    """Classic golden cross strategy — buy when fast SMA crosses above slow SMA.

    Args:
        fast_window: Fast moving average window (default 50).
        slow_window: Slow moving average window (default 200).
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        fast = max(2, min(self.fast_window, len(prices) - 1))
        slow = max(fast + 1, min(self.slow_window, len(prices) - 1))
        sma_fast = wma(prices, fast, sma_filter(fast))
        sma_slow = wma(prices, slow, sma_filter(slow))
        diff = sma_fast[-len(sma_slow):] - sma_slow
        crosses = crossover_detector(diff)
        signals = np.zeros(len(crosses), dtype=int)
        signals[crosses > 0.5] = 1
        signals[crosses < -0.5] = -1
        if len(signals) < len(prices):
            pad = len(prices) - len(signals)
            signals = np.pad(signals, (pad, 0), mode="constant", constant_values=0)
        return signals

    def describe(self) -> str:
        return f"GoldenCross(fast={self.fast_window}, slow={self.slow_window})"


class DeathCross:
    """Classic death cross strategy — sell when fast SMA crosses below slow SMA.

    This is the inverse of GoldenCross: it goes short on death cross.
    For simplicity in this framework (no shorting), we treat it as:
    - Buy when slow SMA is above fast (uptrend after reversal)
    - Sell when fast drops below slow

    Args:
        fast_window: Fast moving average window (default 50).
        slow_window: Slow moving average window (default 200).
    """

    def __init__(self, fast_window: int = 50, slow_window: int = 200):
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        return GoldenCross(self.fast_window, self.slow_window).generate_signals(prices)

    def describe(self) -> str:
        return f"DeathCross(fast={self.fast_window}, slow={self.slow_window})"
