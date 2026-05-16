"""GP walk-forward validation.

Tests GP under walk-forward protocol (5 windows, 3y train / 1y test).
Uses seed=42 with parsimony_penalty=500, depth=5.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import json
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP

SEED = 42
LAMBDA = 500
DEPTH = 5

csv_path = Path(__file__).resolve().parent.parent.parent / 'data' / 'btc_daily_2014_2022.csv'
full_data = pd.read_csv(csv_path)
full_data['date'] = pd.to_datetime(full_data['date'])
full_data = full_data.sort_values('date').reset_index(drop=True)
prices_col = 'close'

results = []
start = full_data['date'].iloc[0]

while True:
    train_end = start + pd.DateOffset(years=3)
    test_end = train_end + pd.DateOffset(years=1)
    if test_end > full_data['date'].iloc[-1]:
        break

    train_mask = (full_data['date'] >= start) & (full_data['date'] < train_end)
    test_mask = (full_data['date'] >= train_end) & (full_data['date'] < test_end)
    train_prices = full_data[train_mask][prices_col].to_numpy(dtype=np.float64).flatten()
    test_prices = full_data[test_mask][prices_col].to_numpy(dtype=np.float64).flatten()

    if len(train_prices) < 100 or len(test_prices) < 30:
        start = start + pd.DateOffset(years=1)
        continue

    bh = buy_and_hold(test_prices)

    np.random.seed(SEED)
    gp = GP(pop_size=75, generations=20, seed=SEED, parsimony_penalty=LAMBDA)
    gp.max_depth = DEPTH
    res = gp.optimize(lambda t: backtest(train_prices, gp.evaluate(t, train_prices))['final_cash'])
    test_cash = backtest(test_prices, gp.evaluate(res['best'], test_prices))['final_cash']

    label = f"{train_end.year}-{test_end.year}"
    results.append({
        'window': label,
        'test_cash': float(test_cash),
        'bh_cash': float(bh),
        'wins': test_cash > bh,
        'tree_size': gp._tree_size(res['best']),
        'tree': str(res['best']),
    })
    print(f"  {label:12s}  GP=${test_cash:>10,.0f}  BH=${bh:>10,.0f}  {'Y' if test_cash > bh else 'N'}  size={gp._tree_size(res['best'])}")
    start = start + pd.DateOffset(years=1)

avg_test = np.mean([r['test_cash'] for r in results])
win_rate = sum(r['wins'] for r in results) / len(results) if results else 0
print(f"\nAverage test cash: ${avg_test:>10,.0f}")
print(f"Win rate vs BH:    {win_rate:.1%}")

out = {
    'method': f'GP walk-forward (lambda={LAMBDA}, depth={DEPTH}, seed={SEED})',
    'bh': float(buy_and_hold(test_prices)),
    'windows': results,
    'avg_test_cash': float(avg_test),
    'win_rate_vs_bh': float(win_rate),
}

with open('results.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved to results.json")