"""Extended comparison: PSO structures × GP configurations.

Experiments:
1. PSO + MACD        (7D, "validated" structure from literature)
2. PSO + trivial_sma (2D, simplest possible crossover)
3. Classic 50/200 SMA crossover (fixed, no optimisation — human baseline)
4. GP  + original function set + no parsimony penalty
5. GP  + extended function set + no parsimony penalty
6. GP  + extended function set + parsimony penalty

All evaluated on the same train/test split for fair comparison.
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
    return df


def run_pso(train, test, stype, bounds, n_particles=30, max_iter=50):
    pso = PSO(n_particles=n_particles, max_iter=max_iter, seed=SEED)
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, stype).signals(train))['final_cash'],
        bounds
    )
    pso_test = backtest(test, VectorStrategy(pso_res['best'], stype).signals(test))
    return {
        'train_cash': float(backtest(train, VectorStrategy(pso_res['best'], stype).signals(train))['final_cash']),
        'test_cash': float(pso_test['final_cash']),
        'best_params': [float(x) for x in pso_res['best']],
    }


def run_gp(train, test, pop_size=50, generations=30, parsimony_penalty=0.0, extended=False):
    gp = GP(pop_size=pop_size, generations=generations, seed=SEED, parsimony_penalty=parsimony_penalty)
    if not extended:
        # Restrict to original function set for fair baseline
        gp.funs = ["+", "-", "*", ">", "<", "AND", "IF"]
        gp.terms = ["price", "sma", "lma", "ema", "const"]
        gp.arity = {"+": 2, "-": 2, "*": 2, ">": 2, "<": 2, "AND": 2, "IF": 3}
    gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train))['final_cash'])
    gp_test = backtest(test, gp.evaluate(gp_res['best'], test))
    train_cash = backtest(train, gp.evaluate(gp_res['best'], train))['final_cash']
    return {
        'train_cash': float(train_cash),
        'test_cash': float(gp_test['final_cash']),
        'tree_size': gp._tree_size(gp_res['best']),
        'tree_repr': repr(gp_res['best']),
    }


def main():
    df = load_btc_data()
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()

    bh_train = buy_and_hold(train)
    bh_test = buy_and_hold(test)

    print(f"Buy-and-Hold  train=${bh_train:,.0f}  test=${bh_test:,.0f}")
    print("=" * 70)

    results = {
        'buy_and_hold': {'train': float(bh_train), 'test': float(bh_test)},
        'experiments': [],
    }

    # --- PSO experiments ---
    print("\n[PSO + MACD]")
    macd_bounds = [
        (2, 200), (0.01, 0.99),   # fast EMA
        (2, 200), (0.01, 0.99),   # slow EMA
        (2, 200), (0.01, 0.99),   # signal EMA
        (0, 50),                    # threshold
    ]
    r = run_pso(train, test, 'macd', macd_bounds)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
    results['experiments'].append({'name': 'PSO+MACD', **r})

    print("\n[PSO + trivial SMA]")
    sma_bounds = [(2, 200), (2, 200)]
    r = run_pso(train, test, 'trivial_sma', sma_bounds)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
    results['experiments'].append({'name': 'PSO+trivial_sma', **r})

    # --- Classic baseline (no optimisation) ---
    print("\n[Classic 50/200 SMA crossover]")
    classic_train = backtest(train, VectorStrategy(np.array([50.0, 200.0]), 'trivial_sma').signals(train))
    classic_test = backtest(test, VectorStrategy(np.array([50.0, 200.0]), 'trivial_sma').signals(test))
    r = {
        'train_cash': float(classic_train['final_cash']),
        'test_cash': float(classic_test['final_cash']),
    }
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
    results['experiments'].append({'name': 'Classic_50_200_SMA', **r})
    # --- GP experiments ---
    print("\n[GP + original set, no penalty]")
    r = run_gp(train, test, extended=False, parsimony_penalty=0.0)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
    results['experiments'].append({'name': 'GP_original_no_penalty', **r})

    print("\n[GP + extended set, no penalty]")
    r = run_gp(train, test, extended=True, parsimony_penalty=0.0)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
    results['experiments'].append({'name': 'GP_extended_no_penalty', **r})

    print("\n[GP + extended set, WITH penalty]")
    r = run_gp(train, test, extended=True, parsimony_penalty=500.0)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
    results['experiments'].append({'name': 'GP_extended_with_penalty', **r})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Experiment':<30} {'Train':>10} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    for e in results['experiments']:
        wins = 'YES' if e['test_cash'] > bh_test else 'NO'
        print(f"{e['name']:<30} ${e['train_cash']:>8,.0f} ${e['test_cash']:>8,.0f} {wins:>8}")

    with open('extended_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to extended_comparison_results.json")


if __name__ == '__main__':
    main()
