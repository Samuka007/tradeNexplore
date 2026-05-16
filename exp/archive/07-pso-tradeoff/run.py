"""PSO: particle/iteration trade-off under fixed evaluation budget.

Intuition: With a fixed budget of fitness evaluations, should we use
many particles × few iterations (broad search, shallow refinement) or
few particles × many iterations (narrow search, deep refinement)?

Setup:
- Strategy: position_sma (3D)
- Fixed total evaluations = 1500 (30 particles × 50 iterations is baseline)
- Variants: (15,100), (30,50), (60,25), (100,15)
- Data: train 2014-2019, test 2020-2022
- Seed: 42
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


def run(config, train, test):
    n, it = config['particles'], config['iterations']
    pso = PSO(n_particles=n, max_iter=it, seed=SEED)
    bounds = [(2, 200), (2, 200), (0.1, 100)]
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
        bounds
    )
    sig = VectorStrategy(pso_res['best'], 'position_sma').signals(test)
    test_cash = backtest(test, sig)['final_cash']
    return {
        'config': config,
        'train_cash': float(backtest(train, VectorStrategy(pso_res['best'], 'position_sma').signals(train))['final_cash']),
        'test_cash': float(test_cash),
        'best_params': [float(x) for x in pso_res['best']],
    }


def main():
    train, test = load_data()
    bh = buy_and_hold(test)
    print(f"Buy-and-Hold test: ${bh:,.0f}")
    print("=" * 70)

    configs = [
        {'particles': 15, 'iterations': 100, 'label': '15p×100i'},
        {'particles': 30, 'iterations': 50,  'label': '30p×50i (baseline)'},
        {'particles': 60, 'iterations': 25,  'label': '60p×25i'},
        {'particles': 100, 'iterations': 15, 'label': '100p×15i'},
    ]

    results = []
    for cfg in configs:
        print(f"\nRunning {cfg['label']}...")
        r = run(cfg, train, test)
        print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Config':<25} {'Train':>10} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        print(f"{r['config']['label']:<25} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
