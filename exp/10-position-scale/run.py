"""Position SMA: scale parameter sensitivity.

Intuition: The scale parameter in sigmoid position sizing controls how
steeply the position transitions from 0 to 1 around the zero-crossing.
- scale → 0: nearly discrete (step function)
- scale → ∞: nearly flat (always ~0.5)
- Intermediate values provide a "soft threshold" that may reduce churn.

Setup:
- Strategy: position_sma with d_fast, d_slow from PSO baseline
- Fix d_fast=120, d_slow=179 (from best position_sma run)
- Scan scale ∈ [1e-4, 1e-2, 0.1, 1, 10, 100, 1000]
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
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO

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

def run(train, test, scale):
    pso = PSO(n_particles=20, max_iter=20, seed=SEED)
    bounds = [(2, 200), (2, 200), (scale, scale)]  # fix scale, optimize windows
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
        bounds
    )
    sig_train = VectorStrategy(pso_res['best'], 'position_sma').signals(train)
    sig_test = VectorStrategy(pso_res['best'], 'position_sma').signals(test)
    return {
        'scale': float(scale),
        'train_cash': float(backtest(train, sig_train)['final_cash']),
        'test_cash': float(backtest(test, sig_test)['final_cash']),
        'best_params': [float(x) for x in pso_res['best']],
    }


def main():
    train, test = load_data()
    bh = buy_and_hold(test)
    print(f"Buy-and-Hold test: ${bh:,.0f}")
    print("=" * 70)

    scales = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]

    results = []
    for scale in scales:
        print(f"\nRunning scale={scale}...")
        r = run(train, test, scale)
        print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Scale':<12} {'Train':>10} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        print(f"{r['scale']:<12.4f} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
