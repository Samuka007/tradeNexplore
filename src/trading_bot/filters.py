"""Convolution-based WMA filters for trading signals."""

import numpy as np


def pad(P: np.ndarray, N: int) -> np.ndarray:
    """Pad price sequence so WMA starts at time zero.

    The spec says: "flip the first part of the sequence over (rotate 180),
    so that if for example the series is rising from p0, it will also be
    rising into p0."
    """
    if N <= 1:
        return P
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
    """Exponential Moving Average kernel — exponential decay, normalised."""
    k = np.arange(N)
    kernel = alpha * (1 - alpha) ** k
    return kernel / kernel.sum()


def wma(P: np.ndarray, N: int, kernel: np.ndarray) -> np.ndarray:
    """Weighted Moving Average via convolution.

    Uses 'valid' mode so output length equals input length
    after padding. The spec's exact code:
        return np.convolve(pad(P, N), kernel, 'valid')
    """
    if len(P) == 0:
        return np.array([])
    if N > len(P):
        raise ValueError(
            f"Window size N={N} cannot exceed price sequence length {len(P)}"
        )
    return np.convolve(pad(P, N), kernel, mode='valid')


def crossover_detector(diff: np.ndarray) -> np.ndarray:
    """Detect sign changes in the difference signal.

    Kernel [1, -1]/2 responds to transitions:
      negative→positive  →  +1  (buy signal)
      positive→negative  →  -1  (sell signal)
      no change           →   0  (hold)
    """
    kernel = np.array([1, -1]) / 2
    sign_signal = np.sign(diff)
    return np.convolve(sign_signal, kernel, mode='valid')
