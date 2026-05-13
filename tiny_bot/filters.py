"""Convolution-based WMA filters."""

import numpy as np


def pad(P: np.ndarray, N: int) -> np.ndarray:
    """Edge condition: flip first N-1 points."""
    padding = -np.flip(P[1:N])
    return np.append(padding, P)


def sma_filter(N: int) -> np.ndarray:
    """Simple Moving Average kernel — equal weights, normalised."""
    return np.ones(N) / N


def lma_filter(N: int) -> np.ndarray:
    """Linear-Weighted Moving Average kernel — triangular weights."""
    k = np.arange(N)
    return (2 / (N + 1)) * (1 - k / N)


def ema_filter(N: int, alpha: float) -> np.ndarray:
    """Exponential Moving Average kernel — exponential decay.

    Raw formula from spec:  α * (1-α)^k  for k = 0..N-1.
    """
    k = np.arange(N)
    return alpha * (1 - alpha) ** k


def wma(P: np.ndarray, N: int, kernel: np.ndarray) -> np.ndarray:
    """Weighted Moving Average via convolution."""
    if N > len(P):
        raise ValueError(f"window {N} larger than price length {len(P)}")
    return np.convolve(pad(P, N), kernel, mode="valid")


def crossover_detector(diff: np.ndarray) -> np.ndarray:
    """Detect sign changes in the difference signal.

    Returns array of the same length as diff (with one fewer valid point).
    +1 = golden cross (diff crosses from negative to positive).
    -1 = death cross (diff crosses from positive to negative).
    """
    golden = (diff[:-1] < 0) & (diff[1:] > 0)
    death = (diff[:-1] > 0) & (diff[1:] < 0)
    crosses = np.zeros(len(diff) - 1, dtype=int)
    crosses[golden] = 1
    crosses[death] = -1
    return crosses
