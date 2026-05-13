"""PSO: walk-forward vs single-split evaluation.

Intuition: Single train/test split may overfit to the specific regime
alignment. Walk-forward re-trains at each step, giving a more honest
estimate of out-of-sample performance.

Setup:
- Strategy: position_sma (3D)
- Single split: train 2014-2019, test 2020-2022
- Walk-forward: 3y train / 1y test, yearly step, re-train each window
- Both use PSO 30×50
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
    return df


def run_single(train, test):
    pso = PSO(n_particles=30, max_iter=50, seed=SEED)
    bounds = [(2, 200), (2, 200), (0.1, 100)]
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
        bounds
    )
    sig = VectorStrategy(pso_res['best'], 'position_sma').signals(test)
    return {
        'method': 'single_split',
        'train_cash': float(backtest(train, VectorStrategy(pso_res['best'], 'position_sma').signals(train))['final_cash']),
        'test_cash': float(backtest(test, sig)['final_cash']),
        'best_params': [float(x) for x in pso_res['best']],
    }


def run_walkforward(df):
    results = []
    start = df['Date'].iloc[0]
    while True:
        train_end = start + pd.DateOffset(years=3)
        test_end = train_end + pd.DateOffset(years=1)
        if test_end > df['Date'].iloc[-1]:
            break
        train_mask = (df['Date'] >= start) & (df['Date'] < train_end)
        test_mask = (df['Date'] >= train_end) & (df['Date'] < test_end)
        train = df[train_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        test = df[test_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        if len(train) < 100 or len(test) < 30:
            start = train_end
            continue

        label = f"{train_end.year}-{test_end.year}"
        pso = PSO(n_particles=30, max_iter=50, seed=SEED)
        bounds = [(2, 200), (2, 200), (0.1, 100)]
        pso_res = pso.optimize(
            lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
            bounds
        )
        sig = VectorStrategy(pso_res['best'], 'position_sma').signals(test)
        test_cash = backtest(test, sig)['final_cash']
        bh = buy_and_hold(test)
        results.append({
            'window': label,
            'test_cash': float(test_cash),
            'bh_cash': float(bh),
            'wins': test_cash > bh,
            'best_params': [float(x) for x in pso_res['best']],
        })
        print(f"  {label:12s}  PSO=${test_cash:>10,.0f}  BH=${bh:>10,.0f}  {'Y' if test_cash > bh else 'N'}")
        start = train_end

    avg_test = np.mean([r['test_cash'] for r in results])
    win_rate = sum(r['wins'] for r in results) / len(results) if results else 0
    return {
        'method': 'walk_forward',
        'windows': results,
        'avg_test_cash': float(avg_test),
        'win_rate_vs_bh': float(win_rate),
    }


def main():
    df = load_data()
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    bh = buy_and_hold(test)

    print(f"Buy-and-Hold test: ${bh:,.0f}")
    print("=" * 70)

    print("\n[Single split]")
    r1 = run_single(train, test)
    print(f"  train=${r1['train_cash']:>10,.0f}  test=${r1['test_cash']:>10,.0f}")

    print("\n[Walk-forward (3y train / 1y test)]")
    r2 = run_walkforward(df)
    print(f"\n  Average test cash: ${r2['avg_test_cash']:>10,.0f}")
    print(f"  Win rate vs BH:    {r2['win_rate_vs_bh']:.1%}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Method':<25} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    print(f"{'single_split':<25} ${r1['test_cash']:>8,.0f} {'YES' if r1['test_cash'] > bh else 'NO':>8}")
    print(f"{'walk_forward (avg)':<25} ${r2['avg_test_cash']:>8,.0f} {'YES' if r2['avg_test_cash'] > bh else 'NO':>8}")

    out = {
        'buy_and_hold_test': float(bh),
        'single_split': r1,
        'walk_forward': r2,
    }
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
