"""10-seed lambda sweep for statistical validity.

Extends exp/17 from 3 seeds to 10 seeds per lambda value.
Provides proper standard errors and confidence for the lambda sweet spot claim.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import json
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP
from tiny_bot.data import load_btc_data

SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
LAMBDAS = [100, 250, 500, 750, 1000, 2000, 5000]
DEPTHS = [5, 7]

train, test = load_btc_data('data/btc_daily_2014_2022.csv')
bh = buy_and_hold(test)

results = []
for seed in SEEDS:
    for lam in LAMBDAS:
        for depth in DEPTHS:
            np.random.seed(seed)
            gp = GP(pop_size=75, generations=20, seed=seed, parsimony_penalty=lam)
            # Set max_depth via attribute
            gp.max_depth = depth
            res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
            test_cash = backtest(test, gp.evaluate(res['best'], test))['final_cash']
            tree_size = gp._tree_size(res['best'])
            results.append({
                'seed': seed,
                'lambda': lam,
                'depth': depth,
                'test': float(test_cash),
                'tree_size': tree_size,
                'tree': str(res['best']),
            })
            print(f"seed={seed:3d} lambda={lam:5d} depth={depth} test=${test_cash:>10,.0f} size={tree_size}")

out = {'bh': float(bh), 'n_runs': len(results), 'results': results}
with open('results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nDone. {len(results)} runs. BH=${bh:,.2f}")