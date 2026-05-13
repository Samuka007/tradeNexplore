"""Controlled experiments: causal-WMA baseline + ablations.

The original pad()-based WMA is a BUG, not a valid baseline.
It injects future prices (P[1:N]) into the series head, creating
an impossible look-ahead that makes any backtest worthless.

This script therefore uses the CORRECT causal WMA as the sole
baseline and tests three remaining hypotheses:
  H1: Transaction-cost drag
  H2: Regime shift (reverse chronological split)
  H3: Overfitting to noise (shuffled prices)

All results are written to experiment_results.json.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO
from tiny_bot.gp import GP

SEED = 42
np.random.seed(SEED)


def load_btc_data(ticker='BTC-USD', start='2014-01-01', end='2022-12-31'):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    return train, test


train, test = load_btc_data()

print("=" * 60)
print("Baseline: Causal WMA + 3% fee")
print("=" * 60)

bounds = (
    [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
  + [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
)

pso = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_res = pso.optimize(
    lambda p: backtest(train, VectorStrategy(p, 'dual_crossover').signals(train))['final_cash'],
    bounds
)
pso_test = backtest(test, VectorStrategy(pso_res['best'], 'dual_crossover').signals(test))

gp = GP(pop_size=50, generations=30, seed=SEED)
gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
gp_test = backtest(test, gp.evaluate(gp_res['best'], test))

baseline = {
    'buy_and_hold_train': float(buy_and_hold(train)),
    'buy_and_hold_test': float(buy_and_hold(test)),
    'pso_train': float(pso_res['fitness']),
    'pso_test': float(pso_test['final_cash']),
    'gp_train': float(gp_res['fitness']),
    'gp_test': float(gp_test['final_cash']),
}
for k, v in baseline.items():
    print(f"  {k}: ${v:,.2f}")

print()
print("=" * 60)
print("H1: Zero transaction cost")
print("=" * 60)

pso_h1 = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_h1_res = pso_h1.optimize(
    lambda p: backtest(train, VectorStrategy(p, 'dual_crossover').signals(train), fee=0.0)['final_cash'],
    bounds
)
pso_h1_test = backtest(test, VectorStrategy(pso_h1_res['best'], 'dual_crossover').signals(test), fee=0.0)

gp_h1 = GP(pop_size=50, generations=30, seed=SEED)
gp_h1_res = gp_h1.optimize(lambda t: backtest(train, gp_h1.evaluate(t, train), fee=0.0)['final_cash'])
gp_h1_test = backtest(test, gp_h1.evaluate(gp_h1_res['best'], test), fee=0.0)

h1_results = {
    'pso_train': float(pso_h1_res['fitness']),
    'pso_test': float(pso_h1_test['final_cash']),
    'gp_train': float(gp_h1_res['fitness']),
    'gp_test': float(gp_h1_test['final_cash']),
}
for k, v in h1_results.items():
    print(f"  {k}: ${v:,.2f}")

print()
print("=" * 60)
print("H2: Reverse-chronological train/test")
print("=" * 60)

pso_h2 = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_h2_res = pso_h2.optimize(
    lambda p: backtest(test, VectorStrategy(p, 'dual_crossover').signals(test))['final_cash'],
    bounds
)
pso_h2_test = backtest(train, VectorStrategy(pso_h2_res['best'], 'dual_crossover').signals(train))

gp_h2 = GP(pop_size=50, generations=30, seed=SEED)
gp_h2_res = gp_h2.optimize(lambda t: backtest(test, gp_h2.evaluate(t, test))['final_cash'])
gp_h2_test = backtest(train, gp_h2.evaluate(gp_h2_res['best'], train))

h2_results = {
    'pso_train': float(pso_h2_res['fitness']),
    'pso_test': float(pso_h2_test['final_cash']),
    'gp_train': float(gp_h2_res['fitness']),
    'gp_test': float(gp_h2_test['final_cash']),
}
for k, v in h2_results.items():
    print(f"  {k}: ${v:,.2f}")

print()
print("=" * 60)
print("H3: Train on random-shuffled prices")
print("=" * 60)

rng = np.random.default_rng(SEED)
shuffled_train = rng.permutation(train.copy())

pso_h3 = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_h3_res = pso_h3.optimize(
    lambda p: backtest(shuffled_train, VectorStrategy(p, 'dual_crossover').signals(shuffled_train))['final_cash'],
    bounds
)
pso_h3_test = backtest(test, VectorStrategy(pso_h3_res['best'], 'dual_crossover').signals(test))

gp_h3 = GP(pop_size=50, generations=30, seed=SEED)
gp_h3_res = gp_h3.optimize(lambda t: backtest(shuffled_train, gp_h3.evaluate(t, shuffled_train))['final_cash'])
gp_h3_test = backtest(test, gp_h3.evaluate(gp_h3_res['best'], test))

h3_results = {
    'pso_train': float(pso_h3_res['fitness']),
    'pso_test': float(pso_h3_test['final_cash']),
    'gp_train': float(gp_h3_res['fitness']),
    'gp_test': float(gp_h3_test['final_cash']),
}
for k, v in h3_results.items():
    print(f"  {k}: ${v:,.2f}")

with open('experiment_results.json', 'w') as f:
    json.dump({
        'baseline': baseline,
        'h1_zero_fee': h1_results,
        'h2_reverse_split': h2_results,
        'h3_shuffled_train': h3_results,
    }, f, indent=2)

print()
print("Saved to experiment_results.json")
