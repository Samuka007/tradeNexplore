"""
Exp 17: Systematic GP Hyperparameter Grid Search

Grid:
- seeds: [0, 42, 88] (3 seeds to verify robustness)
- λ: [100, 250, 500, 750, 1000, 2000, 5000] (7 values, finer granularity)
- max_depth: [5, 7] (2 values)
- pop/gen: fixed at 75x20 (best from exp11)
- function_set: extended (best from exp12)

Total: 3 x 7 x 2 = 42 GP runs
Each run evaluates: train cash, test cash, tree size, tree string

Expected runtime: ~42 x 3min = ~2 hours (with parallel subagents)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
import os
sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP

SEEDS = [0, 42, 88]
LAMBDAS = [100, 250, 500, 750, 1000, 2000, 5000]
DEPTHS = [5, 7]
POP_SIZE = 75
GENERATIONS = 20

# Load data
df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)

results = []
total = len(SEEDS) * len(LAMBDAS) * len(DEPTHS)
count = 0

print(f"Grid search: {len(SEEDS)} seeds x {len(LAMBDAS)} lambdas x {len(DEPTHS)} depths = {total} runs")
print(f"Buy-and-Hold: ${bh:.0f}\n")

for seed in SEEDS:
    for lam in LAMBDAS:
        for depth in DEPTHS:
            count += 1
            np.random.seed(seed)
            
            gp = GP(pop_size=POP_SIZE, generations=GENERATIONS, seed=seed, parsimony_penalty=lam)
            gp.max_depth = depth
            
            def fitness_fn(tree):
                return backtest(train, gp.evaluate(tree, train))['final_cash']
            
            r = gp.optimize(fitness_fn)
            best = r['best']
            
            train_cash = backtest(train, gp.evaluate(best, train))['final_cash']
            test_cash = backtest(test, gp.evaluate(best, test))['final_cash']
            tree_size = gp._tree_size(best)
            tree_str = repr(best)
            
            results.append({
                "seed": seed,
                "lambda": lam,
                "depth": depth,
                "pop": POP_SIZE,
                "gen": GENERATIONS,
                "train": train_cash,
                "test": test_cash,
                "tree_size": tree_size,
                "tree": tree_str,
            })
            
            status = "BEATS BH" if test_cash > bh else ""
            print(f"[{count:3d}/{total}] seed={seed:3d} λ={lam:5d} depth={depth} | train=${train_cash:.0f} test=${test_cash:.0f} tree={tree_size:2d} {status}")
            
            # Save intermediate results
            with open('results.json', 'w') as f:
                json.dump({"bh": bh, "config": {"seeds": SEEDS, "lambdas": LAMBDAS, "depths": DEPTHS, "pop": POP_SIZE, "gen": GENERATIONS}, "results": results}, f, indent=2)

# Summary statistics
print(f"\n{'='*70}")
print("SUMMARY BY LAMBDA (averaged over seeds and depths)")
print(f"{'='*70}")
print(f"{'λ':>6} {'Avg Train':>10} {'Avg Test':>10} {'Std Test':>10} {'Beats BH':>8} {'Avg Tree':>8}")
print("-" * 60)

for lam in LAMBDAS:
    subset = [r for r in results if r['lambda'] == lam]
    tests = [r['test'] for r in subset]
    trains = [r['train'] for r in subset]
    sizes = [r['tree_size'] for r in subset]
    beats = sum(1 for t in tests if t > bh)
    print(f"{lam:>6} ${np.mean(trains):>8,.0f} ${np.mean(tests):>8,.0f} ${np.std(tests):>8,.0f} {beats:>6}/{len(subset)} {np.mean(sizes):>7.1f}")

print(f"\n{'='*70}")
print("SUMMARY BY DEPTH (averaged over seeds and lambdas)")
print(f"{'='*70}")
print(f"{'Depth':>6} {'Avg Train':>10} {'Avg Test':>10} {'Std Test':>10} {'Beats BH':>8} {'Avg Tree':>8}")
print("-" * 60)

for depth in DEPTHS:
    subset = [r for r in results if r['depth'] == depth]
    tests = [r['test'] for r in subset]
    trains = [r['train'] for r in subset]
    sizes = [r['tree_size'] for r in subset]
    beats = sum(1 for t in tests if t > bh)
    print(f"{depth:>6} ${np.mean(trains):>8,.0f} ${np.mean(tests):>8,.0f} ${np.std(tests):>8,.0f} {beats:>6}/{len(subset)} {np.mean(sizes):>7.1f}")

print(f"\n{'='*70}")
print("TOP 10 CONFIGURATIONS (by test)")
print(f"{'='*70}")
print(f"{'Rank':>5} {'Seed':>5} {'λ':>6} {'Depth':>6} {'Train':>10} {'Test':>10} {'Tree':>5}")
print("-" * 55)

sorted_results = sorted(results, key=lambda r: r['test'], reverse=True)
for i, r in enumerate(sorted_results[:10]):
    print(f"{i+1:>5} {r['seed']:>5} {r['lambda']:>6} {r['depth']:>6} ${r['train']:>8,.0f} ${r['test']:>8,.0f} {r['tree_size']:>5}")

print(f"\nBuy-and-Hold: ${bh:.0f}")
print("Saved to results.json")
