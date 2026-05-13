"""GP: parsimony penalty and max_depth sensitivity.

Intuition: Parsimony pressure λ controls the trade-off between raw
performance and tree complexity. Too low → overfitting; too high →
underfitting (tree collapses to 1 node). Similarly, max_depth limits
the expressive power of the hypothesis space.

Setup:
- Strategy: extended function set, discrete signals
- Pop=50, Gen=30, cross=0.9, mut=0.1
- λ values: 0, 100, 500, 1000, 5000
- max_depth values: 3, 5, 7
- Full grid search (5 × 3 = 15 runs)
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


def run(train, test, penalty, depth):
    gp = GP(pop_size=50, generations=30, seed=SEED, parsimony_penalty=penalty)
    gp.max_depth = depth
    gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
    train_cash = backtest(train, gp.evaluate(gp_res['best'], train))['final_cash']
    test_cash = backtest(test, gp.evaluate(gp_res['best'], test))['final_cash']
    return {
        'penalty': float(penalty),
        'max_depth': int(depth),
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

    penalties = [0, 100, 500, 1000, 5000]
    depths = [3, 5, 7]

    results = []
    for penalty in penalties:
        for depth in depths:
            label = f"λ={penalty}, d={depth}"
            print(f"\nRunning {label}...")
            r = run(train, test, penalty, depth)
            print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
            results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Config':<20} {'Train':>10} {'Test':>10} {'Size':>6} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        label = f"λ={r['penalty']:.0f}, d={r['max_depth']}"
        print(f"{label:<20} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {r['tree_size']:>6} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
