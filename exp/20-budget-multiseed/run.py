"""Budget paradox: multi-seed validation of GP pop_size x generations trade-off.

Tests (50,30), (75,20), (100,15) across 10 seeds to check if the budget
paradox holds robustly or is a single-seed artifact.
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
CONFIGS = [(50, 30), (75, 20), (100, 15)]
LAMBDA = 1000  # Same as budget experiment

train, test = load_btc_data('data/btc_daily_2014_2022.csv')
bh = buy_and_hold(test)

results = []
for seed in SEEDS:
    for pop, gen in CONFIGS:
        np.random.seed(seed)
        gp = GP(pop_size=pop, generations=gen, seed=seed, parsimony_penalty=LAMBDA)
        res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
        test_cash = backtest(test, gp.evaluate(res['best'], test))['final_cash']
        tree_size = gp._tree_size(res['best'])
        results.append({
            'seed': seed,
            'pop': pop,
            'gen': gen,
            'evals': pop * gen,
            'test': float(test_cash),
            'tree_size': tree_size,
        })
        print(f"seed={seed:3d} pop={pop:3d} gen={gen:2d} evals={pop*gen:4d} test=${test_cash:>10,.0f} size={tree_size}")

# Summary per config
print("\n=== SUMMARY ===")
for pop, gen in CONFIGS:
    tests = [r['test'] for r in results if r['pop'] == pop and r['gen'] == gen]
    beats = sum(1 for t in tests if t > bh)
    print(f"({pop:3d},{gen:2d}): mean=${np.mean(tests):>10,.0f} std=${np.std(tests):>7,.0f} median=${np.median(tests):>10,.0f} beats={beats}/10")

out = {'bh': float(bh), 'results': results}
with open('results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved to results.json")