"""Adapters to bridge algorithm outputs with ExperimentRunner interfaces."""

import numpy as np


class GPAdapter:
    """Wrap a GP tree so it satisfies the Strategy protocol.

    TreeStrategy.generate_signals() falls back to all-hold for non-dict
    tree types (e.g. GPNode). This adapter delegates signal generation
    directly to GeneticProgramming.evaluate().

    Usage:
        gp = GeneticProgramming(...)
        strategy_factory = lambda tree: GPAdapter(gp, tree)
        runner.run_structural(gp, strategy_factory)
    """

    def __init__(self, gp, tree):
        self._gp = gp
        self._tree = tree

    def generate_signals(self, prices: np.ndarray) -> np.ndarray:
        return self._gp.evaluate(self._tree, prices)

    def describe(self) -> str:
        return f"GP({self._tree})"
