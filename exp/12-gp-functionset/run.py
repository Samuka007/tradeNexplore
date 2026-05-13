"""GP: function set comparison.

Intuition: The expressive power of GP depends on its terminal and function
sets. A richer set allows more complex strategies but also increases the
risk of overfitting. We compare:
- minimal: only SMA, +, -, > (barebones)
- original: SMA, LMA, EMA, +, -, *, >, <, AND, IF (course baseline)
- extended: original + RSI, momentum, volatility, /, ABS, MAX, MIN

Setup:
- Pop=50, Gen=30, λ=500, max_depth=5
- Discrete signals
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


def run(train, test, funs, terms, label):
    gp = GP(pop_size=50, generations=30, seed=SEED, parsimony_penalty=500.0)
    gp.max_depth = 5
    gp.funs = funs
    gp.terms = terms
    gp.arity = {"+": 2, "-": 2, "*": 2, "/": 2, "ABS": 1, "MAX": 2, "MIN": 2, ">": 2, "<": 2, "AND": 2, "IF": 3}
    gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
    train_cash = backtest(train, gp.evaluate(gp_res['best'], train))['final_cash']
    test_cash = backtest(test, gp.evaluate(gp_res['best'], test))['final_cash']
    return {
        'label': label,
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

    variants = [
        ("minimal", ["+", "-", ">"], ["price", "sma", "const"]),
        ("original", ["+", "-", "*", ">", "<", "AND", "IF"], ["price", "sma", "lma", "ema", "const"]),
        ("extended", ["+", "-", "*", "/", "ABS", "MAX", "MIN", ">", "<", "AND", "IF"],
         ["price", "sma", "lma", "ema", "rsi", "momentum", "volatility", "const"]),
    ]

    results = []
    for label, funs, terms in variants:
        print(f"\nRunning {label}...")
        r = run(train, test, funs, terms, label)
        print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Function Set':<20} {'Train':>10} {'Test':>10} {'Size':>6} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        print(f"{r['label']:<20} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {r['tree_size']:>6} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
