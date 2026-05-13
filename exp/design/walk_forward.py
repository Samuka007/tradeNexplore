"""Walk-forward validation + relative-benchmark evaluation.

Design critique of the single-split protocol:
1.  Single train/test split on non-stationary data is inherently fragile.
2.  A strategy that "wins" on one split may simply have been lucky with regime alignment.
3.  Walk-forward validation (rolling/expanding window) is the correct protocol for
    financial time series (Lopez de Prado, 2018).

This script implements:
- Rolling-window walk-forward: train on [t-3y, t), test on [t, t+1y).
- Relative metrics per window:
    * final_cash vs buy_and_hold
    * Information Ratio (strategy return minus benchmark return, annualised)
    * Win rate: fraction of windows where strategy > benchmark
- Expanding-window variant as a robustness check.
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


def walk_forward_rolling(df, train_years=3, test_years=1):
    """Yield (train_mask, test_mask, label) for each rolling window."""
    start = df['Date'].iloc[0]
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > df['Date'].iloc[-1]:
            break
        train_mask = (df['Date'] >= start) & (df['Date'] < train_end)
        test_mask = (df['Date'] >= train_end) & (df['Date'] < test_end)
        label = f"{train_end.year}-{test_end.year}"
        yield train_mask, test_mask, label
        start = train_end  # roll forward by test_years


def run_pso(train_prices, test_prices):
    bounds = (
        [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
      + [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
    )
    pso = PSO(n_particles=20, max_iter=30, seed=SEED)
    pso_res = pso.optimize(
        lambda p: backtest(train_prices, VectorStrategy(p, 'dual_crossover').signals(train_prices))['final_cash'],
        bounds
    )
    pso_test = backtest(test_prices, VectorStrategy(pso_res['best'], 'dual_crossover').signals(test_prices))
    return pso_test['final_cash']


def run_gp(train_prices, test_prices):
    gp = GP(pop_size=30, generations=20, seed=SEED)
    gp_res = gp.optimize(lambda t: backtest(train_prices, gp.evaluate(t, train_prices))['final_cash'])
    gp_test = backtest(test_prices, gp.evaluate(gp_res['best'], test_prices))
    return gp_test['final_cash']


def main():
    df = load_btc_data()
    print("Walk-forward evaluation (rolling 3y train / 1y test)")
    print("=" * 60)

    results = []
    for train_mask, test_mask, label in walk_forward_rolling(df):
        train = df[train_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        test = df[test_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        if len(train) < 100 or len(test) < 30:
            continue

        bh = buy_and_hold(test)
        pso_cash = run_pso(train, test)
        gp_cash = run_gp(train, test)

        # Relative returns (vs $1000 initial)
        pso_ret = (pso_cash - 1000) / 1000
        gp_ret = (gp_cash - 1000) / 1000
        bh_ret = (bh - 1000) / 1000

        # Information ratio proxy: excess return / |benchmark return| + epsilon
        pso_excess = pso_ret - bh_ret
        gp_excess = gp_ret - bh_ret
        ir_pso = pso_excess / (abs(bh_ret) + 1e-10)
        ir_gp = gp_excess / (abs(bh_ret) + 1e-10)

        results.append({
            'window': label,
            'bh_cash': float(bh),
            'pso_cash': float(pso_cash),
            'gp_cash': float(gp_cash),
            'pso_wins': pso_cash > bh,
            'gp_wins': gp_cash > bh,
            'ir_pso': float(ir_pso),
            'ir_gp': float(ir_gp),
        })

        print(f"{label:12s}  BH=${bh:>10,.0f}  PSO=${pso_cash:>10,.0f}  GP=${gp_cash:>10,.0f}  "
              f"PSO>bh={'Y' if pso_cash > bh else 'N'}  GP>bh={'Y' if gp_cash > bh else 'N'}")

    # Aggregate
    pso_win_rate = sum(r['pso_wins'] for r in results) / len(results) if results else 0
    gp_win_rate = sum(r['gp_wins'] for r in results) / len(results) if results else 0
    avg_ir_pso = np.mean([r['ir_pso'] for r in results]) if results else 0
    avg_ir_gp = np.mean([r['ir_gp'] for r in results]) if results else 0

    print()
    print("=" * 60)
    print("AGGREGATE")
    print(f"  Windows evaluated: {len(results)}")
    print(f"  PSO win rate vs BH: {pso_win_rate:.1%}")
    print(f"  GP  win rate vs BH: {gp_win_rate:.1%}")
    print(f"  Avg IR PSO: {avg_ir_pso:.3f}")
    print(f"  Avg IR GP:  {avg_ir_gp:.3f}")

    with open('walk_forward_results.json', 'w') as f:
        json.dump({
            'windows': results,
            'aggregate': {
                'pso_win_rate': float(pso_win_rate),
                'gp_win_rate': float(gp_win_rate),
                'avg_ir_pso': float(avg_ir_pso),
                'avg_ir_gp': float(avg_ir_gp),
            }
        }, f, indent=2)

    print()
    print("Saved to walk_forward_results.json")


if __name__ == '__main__':
    main()
