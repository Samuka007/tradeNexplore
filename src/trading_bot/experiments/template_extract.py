"""Extract parametric templates from GP trees for hierarchical optimization.

The GP tree structure is converted into a dual_crossover template where
durations are fixed (from the tree's terminal nodes) and weights/alpha
are free for PSO to optimize.
"""

from __future__ import annotations

import numpy as np

from trading_bot.algorithms.genetic_programming import GeneticProgramming
from trading_bot.strategy import VectorStrategy


def extract_dual_crossover_template(
    gp: GeneticProgramming,
    tree: object,
    prices: np.ndarray,
) -> tuple[VectorStrategy, list[tuple[float, float]]]:
    """Convert a GP tree into a dual_crossover template for PSO refinement.

    Strategy:
    - Collect all terminal nodes with duration parameters from the GP tree.
    - Assign shorter durations to the HIGH component, longer to LOW.
    - Fix durations; free weights and alphas for PSO to optimize.

    Args:
        gp: Trained GP instance (used to collect terminals).
        tree: Best GP tree from structural optimization.
        prices: Training prices (unused, kept for interface compatibility).

    Returns:
        (template_strategy, bounds) — strategy with initial params
        and bounds where durations are fixed to single values.
    """
    terminals = gp._collect_terminals(tree)

    # Collect all unique (filter_type, duration) pairs
    sma_durs = []
    lma_durs = []
    ema_durs = []
    ema_alphas = []

    for t in terminals:
        if t[0] == "sma":
            sma_durs.append(t[1])
        elif t[0] == "lma":
            lma_durs.append(t[1])
        elif t[0] == "ema":
            ema_durs.append(t[1])
            ema_alphas.append(t[2])

    # Sort durations: shorter → HIGH, longer → LOW
    all_durs = sma_durs + lma_durs + ema_durs
    if not all_durs:
        # Fallback: default template
        all_durs = [10, 30, 50, 80, 120, 200]

    all_durs = sorted(set(all_durs))
    if len(all_durs) < 3:
        # Pad with default durations
        defaults = [10, 30, 50, 80, 120, 200]
        for d in defaults:
            if d not in all_durs:
                all_durs.append(d)
        all_durs = sorted(all_durs)
    n = len(all_durs)

    # Split into HIGH (first 3) and LOW (last 3)
    # Pad/clone if fewer than needed
    high_durs = (all_durs[:3] * 2)[:3]   # take up to 3, pad with repeats
    low_durs = (all_durs[-3:] * 2)[:3]   # take up to 3, pad with repeats

    # Fixed alpha from EMA terminals, or default
    fixed_alpha = ema_alphas[0] if ema_alphas else 0.3

    # Build initial params: weights=1/3 each, fixed durations, fixed alpha
    initial = np.array(
        [1.0, 1.0, 1.0,                # w1-w3 HIGH
         high_durs[0], high_durs[1], high_durs[2],  # d1-d3 HIGH
         fixed_alpha,                   # a3
         1.0, 1.0, 1.0,                # w4-w6 LOW
         low_durs[0], low_durs[1], low_durs[2],     # d4-d6 LOW
         fixed_alpha],                  # a6
        dtype=np.float64,
    )

    # Bounds: weights free, durations fixed, alphas free
    bounds = [
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (float(high_durs[0]), float(high_durs[0])),
        (float(high_durs[1]), float(high_durs[1])),
        (float(high_durs[2]), float(high_durs[2])),
        (0.01, 0.99),
        (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        (float(low_durs[0]), float(low_durs[0])),
        (float(low_durs[1]), float(low_durs[1])),
        (float(low_durs[2]), float(low_durs[2])),
        (0.01, 0.99),
    ]

    template = VectorStrategy(initial, "dual_crossover")
    return template, bounds
