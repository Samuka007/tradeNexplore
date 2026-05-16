"""
Exp 19: PSO position_sma landscape grid visualization.

Grid search over fast/slow windows to reveal basin structure.
"""
import sys, json, os
sys.path.insert(0, '../..')

import numpy as np
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest
from tiny_bot.strategy import VectorStrategy

train, test, df = load_btc_data()

# Grid over fast (20-200, step 10) and slow (fast+10 to 200, step 10)
# Fix scale = 0.1 (optimal from PSO results)
grid = []
fast_range = list(range(20, 201, 10))
for fast in fast_range:
    for slow in range(fast + 10, 201, 10):
        params = np.array([float(fast), float(slow), 0.1])
        sig = VectorStrategy(params, 'position_sma').signals(train)
        ret = backtest(train, sig)['final_cash']
        grid.append({'fast': fast, 'slow': slow, 'train': float(ret)})

# Also evaluate test set for the best grid points
for g in grid:
    params = np.array([float(g['fast']), float(g['slow']), 0.1])
    sig = VectorStrategy(params, 'position_sma').signals(test)
    g['test'] = float(backtest(test, sig)['final_cash'])

# Find basins
from collections import defaultdict
basins = defaultdict(list)
for g in grid:
    # round to nearest basin center
    f, s = g['fast'], g['slow']
    if 35 <= f <= 50 and 90 <= s <= 120:
        basins['basin_b'].append(g)
    elif 100 <= f <= 130 and 160 <= s <= 190:
        basins['basin_a'].append(g)
    else:
        basins['other'].append(g)

with open('results.json', 'w') as f:
    json.dump({
        'experiment': 'landscape_grid',
        'grid': grid,
        'basin_a_mean_train': float(np.mean([g['train'] for g in basins['basin_a']])) if basins['basin_a'] else None,
        'basin_b_mean_train': float(np.mean([g['train'] for g in basins['basin_b']])) if basins['basin_b'] else None,
        'basin_a_mean_test': float(np.mean([g['test'] for g in basins['basin_a']])) if basins['basin_a'] else None,
        'basin_b_mean_test': float(np.mean([g['test'] for g in basins['basin_b']])) if basins['basin_b'] else None,
        'global_best_train': max(grid, key=lambda x: x['train']),
        'global_best_test': max(grid, key=lambda x: x['test']),
    }, f, indent=2)

print(f"Grid points: {len(grid)}")
print(f"Basin A mean train: {np.mean([g['train'] for g in basins['basin_a']]):.0f}" if basins['basin_a'] else "Basin A empty")
print(f"Basin B mean train: {np.mean([g['train'] for g in basins['basin_b']]):.0f}" if basins['basin_b'] else "Basin B empty")
print(f"Global best train: {max(grid, key=lambda x: x['train'])}")
print(f"Global best test: {max(grid, key=lambda x: x['test'])}")
