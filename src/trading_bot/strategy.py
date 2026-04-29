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

    Parameters (14D for dual_crossover):
        [w1, w2, w3, d1, d2, d3, a3, w4, w5, w6, d4, d5, d6, a6]
    where w=weights, d=durations (int), a=EMA alpha.
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
        high = self._component_signal(prices, self.high_w, self.high_d, self.high_a)
        low = self._component_signal(prices, self.low_w, self.low_d, self.low_a)
        diff = high - low
        crosses = crossover_detector(diff)

        # Map crossover events to buy/sell signals
        signals = np.zeros(len(crosses), dtype=int)
        signals[crosses > 0.5] = 1
        signals[crosses < -0.5] = -1
        return signals

    def describe(self) -> str:
        return (
            f"DualCrossover("
            f"high_d={self.high_d.tolist()}, low_d={self.low_d.tolist()}"
            f")"
        )
