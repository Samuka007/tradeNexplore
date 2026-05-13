"""Parametrised trading strategies."""

import numpy as np
from tiny_bot.filters import wma, sma_filter, lma_filter, ema_filter, crossover_detector


class VectorStrategy:
    """Fixed-structure strategy parameterised by a continuous vector.

    dual_crossover (14D): [w1,w2,w3,d1,d2,d3,a3, w4,w5,w6,d4,d5,d6,a6]
    macd (7D): [d1,a1,d2,a2,d3,a3,threshold]
    trivial_sma (2D): [d_fast, d_slow]  -- simple dual-SMA crossover, no weighting
    position_sma (3D): [d_fast, d_slow, scale] -- dual-SMA with sigmoid position sizing
    """
    def __init__(self, params: np.ndarray, stype: str = "dual_crossover"):
        self.params = params
        self.type = stype
        self._unpack()

    def _unpack(self):
        p = self.params
        if self.type == "dual_crossover":
            self.hw = self._norm(p[0:3])
            self.hd = np.clip(np.round(p[3:6]), 2, 200).astype(int)
            self.ha = np.clip(p[6], 0.01, 0.99)
            self.lw = self._norm(p[7:10])
            self.ld = np.clip(np.round(p[10:13]), 2, 200).astype(int)
            self.la = np.clip(p[13], 0.01, 0.99)
        elif self.type == "macd":
            self.d1 = max(2, min(int(np.round(p[0])), 200))
            self.a1 = np.clip(p[1], 0.01, 0.99)
            self.d2 = max(2, min(int(np.round(p[2])), 200))
            self.a2 = np.clip(p[3], 0.01, 0.99)
            self.sd = max(2, min(int(np.round(p[4])), 200))
            self.sa = np.clip(p[5], 0.01, 0.99)
            self.thresh = p[6] if len(p) > 6 else 0.0
        elif self.type == "trivial_sma":
            self.d_fast = max(2, min(int(np.round(p[0])), 200))
            self.d_slow = max(2, min(int(np.round(p[1])), 200))
        elif self.type == "position_sma":
            self.d_fast = max(2, min(int(np.round(p[0])), 200))
            self.d_slow = max(2, min(int(np.round(p[1])), 200))
            self.scale = max(1e-6, float(p[2]))
        else:
            raise ValueError(f"unknown strategy type: {self.type}")

    @staticmethod
    def _norm(weights: np.ndarray) -> np.ndarray:
        return weights / (weights.sum() + 1e-10)

    def _component(self, prices: np.ndarray, weights, durations, alpha):
        """Compute weighted sum of SMA/LMA/EMA signals."""
        sigs = []
        for i, (wi, di) in enumerate(zip(weights, durations)):
            di = min(max(int(di), 2), len(prices) - 1)
            if i == 0:
                sig = wma(prices, di, sma_filter(di))
            elif i == 1:
                sig = wma(prices, di, lma_filter(di))
            else:
                sig = wma(prices, di, ema_filter(di, alpha))
            sigs.append(wi * sig)
        m = min(len(s) for s in sigs)
        return np.sum([s[-m:] for s in sigs], axis=0)

    def signals(self, prices: np.ndarray) -> np.ndarray:
        if self.type == "dual_crossover":
            high = self._component(prices, self.hw, self.hd, self.ha)
            low = self._component(prices, self.lw, self.ld, self.la)
            diff = high - low
        elif self.type == "macd":
            d1 = min(max(self.d1, 2), len(prices) - 1)
            d2 = min(max(self.d2, 2), len(prices) - 1)
            sd = min(max(self.sd, 2), len(prices) - 1)
            ema_fast = wma(prices, d1, ema_filter(d1, self.a1))
            ema_slow = wma(prices, d2, ema_filter(d2, self.a2))
            macd_line = ema_fast - ema_slow
            signal_line = wma(macd_line, sd, ema_filter(sd, self.sa))
            m = min(len(macd_line), len(signal_line))
            diff = macd_line[-m:] - signal_line[-m:]
            if self.thresh > 0:
                diff = np.where(np.abs(diff) > self.thresh, diff, 0.0)
        elif self.type == "trivial_sma":
            d_fast = min(max(self.d_fast, 2), len(prices) - 1)
            d_slow = min(max(self.d_slow, 2), len(prices) - 1)
            sma_fast = wma(prices, d_fast, sma_filter(d_fast))
            sma_slow = wma(prices, d_slow, sma_filter(d_slow))
            m = min(len(sma_fast), len(sma_slow))
            diff = sma_fast[-m:] - sma_slow[-m:]
        elif self.type == "position_sma":
            d_fast = min(max(self.d_fast, 2), len(prices) - 1)
            d_slow = min(max(self.d_slow, 2), len(prices) - 1)
            sma_fast = wma(prices, d_fast, sma_filter(d_fast))
            sma_slow = wma(prices, d_slow, sma_filter(d_slow))
            diff = np.nan_to_num(sma_fast - sma_slow, nan=0.0)
            # Sigmoid position: 0 = empty, 1 = full long
            return 1.0 / (1.0 + np.exp(-np.clip(diff / self.scale, -500, 500)))
        else:
            raise ValueError(f"unknown strategy type: {self.type}")

        crosses = crossover_detector(diff)
        sig = np.zeros(len(crosses), dtype=int)
        sig[crosses > 0.5] = 1
        sig[crosses < -0.5] = -1
        if len(sig) < len(prices):
            sig = np.pad(sig, (len(prices) - len(sig), 0), mode="constant", constant_values=0)
        return sig
