"""GP: population/generation trade-off under fixed evaluation budget.

Intuition: GP has two main hyperparameters: population size (pop) and
number of generations (gen). Their product ≈ total evaluations. Does a
large population with few generations outperform a small population with
many generations?

Setup:
- Fixed total evaluations ≈ 1500 (50 pop × 30 gen is baseline)
- Variants: (25,60), (50,30), (75,20), (100,15)
- Strategy: extended function set, discrete, λ=500, max_depth=5
- Data: train 2014-2019, test 2020-2022
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import json
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP

SEED = 42
np.random.seed(SEED)


def load_data():
    df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    return train, test


def run(train, test, pop, gen):
    gp = GP(pop_size=pop, generations=gen, seed=SEED, parsimony_penalty=500.0)
    gp.max_depth = 5
    gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
    train_cash = backtest(train, gp.evaluate(gp_res['best'], train))['final_cash']
    test_cash = backtest(test, gp.evaluate(gp_res['best'], test))['final_cash']
    return {
        'pop': int(pop),
        'gen': int(gen),
        'evals': int(pop * gen),
        'train_cash': float(train_cash),
        'test_cash': float(test_cash),
        'tree_size': gp._tree_size(gp_res['best']),
        'tree_repr': repr(gp_res['best']),
    }


def main():
    train, test = load_data()
    bh = buy_and_hold(test)
    print(f"Buy-and-Hold test: ${bh:,.0f}")
    print("=" * 70)

    configs = [
        (25, 60),
        (50, 30),
        (75, 20),
        (100, 15),
    ]

    results = []
    for pop, gen in configs:
        label = f"{pop}p×{gen}g"
        print(f"\nRunning {label}...")
        r = run(train, test, pop, gen)
        print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Config':<15} {'Evals':>6} {'Train':>10} {'Test':>10} {'Size':>6} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        print(f"{r['pop']}p×{r['gen']}g{r['evals']:>8} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {r['tree_size']:>6} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
