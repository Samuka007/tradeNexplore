"""Position sizing comparison: discrete vs continuous.

Compare:
1. trivial SMA (discrete {-1, 0, +1})
2. position SMA (continuous [0, 1] via sigmoid)
3. GP + extended set + penalty (discrete)
4. GP + extended set + penalty (continuous)
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


def load_btc_data():
    df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def run_pso(train, test, stype, bounds):
    pso = PSO(n_particles=30, max_iter=50, seed=SEED)
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, stype).signals(train))['final_cash'],
        bounds
    )
    sig_train = VectorStrategy(pso_res['best'], stype).signals(train)
    sig_test = VectorStrategy(pso_res['best'], stype).signals(test)
    train_cash = backtest(train, sig_train)['final_cash']
    test_cash = backtest(test, sig_test)['final_cash']
    return {
        'train_cash': float(train_cash),
        'test_cash': float(test_cash),
        'best_params': [float(x) for x in pso_res['best']],
    }


def run_gp(train, test, continuous=False, parsimony_penalty=500.0):
    gp = GP(pop_size=50, generations=30, seed=SEED, parsimony_penalty=parsimony_penalty)
    gp_res = gp.optimize(lambda t: backtest(train, gp.evaluate(t, train, continuous=continuous))['final_cash'])
    train_cash = backtest(train, gp.evaluate(gp_res['best'], train, continuous=continuous))['final_cash']
    test_cash = backtest(test, gp.evaluate(gp_res['best'], test, continuous=continuous))['final_cash']
    return {
        'train_cash': float(train_cash),
        'test_cash': float(test_cash),
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

    print("\n[PSO + trivial SMA (discrete)]")
    r = run_pso(train, test, 'trivial_sma', [(2, 200), (2, 200)])
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
    results['experiments'].append({'name': 'PSO_trivial_discrete', **r})

    print("\n[PSO + position SMA (continuous)]")
    r = run_pso(train, test, 'position_sma', [(2, 200), (2, 200), (1e-3, 100)])
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
    results['experiments'].append({'name': 'PSO_position_continuous', **r})

    print("\n[GP + extended + penalty (discrete)]")
    r = run_gp(train, test, continuous=False, parsimony_penalty=500.0)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
    results['experiments'].append({'name': 'GP_discrete', **r})

    print("\n[GP + extended + penalty (continuous)]")
    r = run_gp(train, test, continuous=True, parsimony_penalty=500.0)
    print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}  size={r['tree_size']}")
    results['experiments'].append({'name': 'GP_continuous', **r})

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Experiment':<35} {'Train':>10} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    for e in results['experiments']:
        wins = 'YES' if e['test_cash'] > bh_test else 'NO'
        print(f"{e['name']:<35} ${e['train_cash']:>8,.0f} ${e['test_cash']:>8,.0f} {wins:>8}")

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
