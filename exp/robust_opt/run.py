"""Robust optimization: optimise once, validate on *all* windows simultaneously.

Engineering notes
-----------------
* Standard hyper-parameters (PSO 30/50, GP 50/30).  Speed-up comes from
  parallelism, not from shrinking the search space.
* PSO and GP are launched in **parallel processes** because they are fully
  independent.  Within each algorithm ThreadPoolExecutor parallelises particle /
  population evaluation, and an additional inner ThreadPoolExecutor parallelises
  the K window backtests inside each fitness call.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO
from tiny_bot.gp import GP

SEED = 42
np.random.seed(SEED)

TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_MONTHS = 1
WINDOWS_PER_FITNESS = 12
N_WORKERS_INNER = 4          # windows inside one fitness call
N_WORKERS_ALGO = 4           # particles / population members


def load_btc_data(ticker='BTC-USD', start='2014-01-01', end='2022-12-31'):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def make_windows(df, train_years=TRAIN_YEARS, test_years=TEST_YEARS, step_months=STEP_MONTHS):
    windows = []
    start = df['Date'].iloc[0]
    while True:
        train_end = start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > df['Date'].iloc[-1]:
            break
        train_mask = (df['Date'] >= start) & (df['Date'] < train_end)
        test_mask = (df['Date'] >= train_end) & (df['Date'] < test_end)
        train = df[train_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        test = df[test_mask]['Close'].to_numpy(dtype=np.float64).flatten()
        if len(train) >= 100 and len(test) >= 30:
            windows.append((train, test))
        start = start + pd.DateOffset(months=step_months)
    return windows


# ---------------------------------------------------------------------------
# PSO worker
# ---------------------------------------------------------------------------
def _eval_one_window_pso(params, train):
    sig = VectorStrategy(params, 'dual_crossover').signals(train)
    return backtest(train, sig)['final_cash']


def robust_fitness_pso(params, windows, rng):
    if len(windows) <= WINDOWS_PER_FITNESS:
        sample = windows
    else:
        idx = rng.choice(len(windows), size=WINDOWS_PER_FITNESS, replace=False)
        sample = [windows[i] for i in idx]
    cash_list = []
    with ThreadPoolExecutor(max_workers=N_WORKERS_INNER) as ex:
        futures = [ex.submit(_eval_one_window_pso, params, train) for train, _ in sample]
        for f in futures:
            cash_list.append(f.result())
    return float(np.mean(cash_list))


def run_pso(windows, seed=SEED):
    bounds = (
        [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
      + [(0.0, 1.0)] * 3 + [(2.0, 200.0)] * 3 + [(0.01, 0.99)]
    )
    pso = PSO(n_particles=30, max_iter=50, seed=seed)
    rng = np.random.default_rng(seed)
    res = pso.optimize(
        lambda p: robust_fitness_pso(p, windows, rng),
        bounds,
        n_workers=N_WORKERS_ALGO,
    )
    return res


# ---------------------------------------------------------------------------
# GP worker
# ---------------------------------------------------------------------------
def _eval_one_window_gp(tree, gp_instance, train):
    sig = gp_instance.evaluate(tree, train)
    return backtest(train, sig)['final_cash']


def robust_fitness_gp(tree, windows, gp_instance, rng):
    if len(windows) <= WINDOWS_PER_FITNESS:
        sample = windows
    else:
        idx = rng.choice(len(windows), size=WINDOWS_PER_FITNESS, replace=False)
        sample = [windows[i] for i in idx]
    cash_list = []
    with ThreadPoolExecutor(max_workers=N_WORKERS_INNER) as ex:
        futures = [ex.submit(_eval_one_window_gp, tree, gp_instance, train) for train, _ in sample]
        for f in futures:
            cash_list.append(f.result())
    return float(np.mean(cash_list))


def run_gp(windows, seed=SEED):
    gp = GP(pop_size=50, generations=30, seed=seed)
    rng = np.random.default_rng(seed + 1)
    res = gp.optimize(
        lambda t: robust_fitness_gp(t, windows, gp, rng),
        n_workers=N_WORKERS_ALGO,
    )
    return res, gp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(params, windows, strategy_type='pso', gp_instance=None):
    results = []
    for train, test in windows:
        if strategy_type == 'pso':
            sig = VectorStrategy(params, 'dual_crossover').signals(test)
        else:
            sig = gp_instance.evaluate(params, test)
        cash = backtest(test, sig)['final_cash']
        bh = buy_and_hold(test)
        results.append({
            'strategy_cash': float(cash),
            'bh_cash': float(bh),
            'wins': cash > bh,
            'excess': float((cash - bh) / 1000),
        })
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_btc_data()
    windows = make_windows(df)
    print(f"Generated {len(windows)} walk-forward windows")
    print(f"Fitness evaluation samples {WINDOWS_PER_FITNESS} windows per call")
    print(f"Inner parallel workers: {N_WORKERS_INNER}, Algo workers: {N_WORKERS_ALGO}")
    print("=" * 60)

    # Run PSO and GP in parallel processes
    print("Launching PSO and GP in parallel...")
    with ProcessPoolExecutor(max_workers=2) as ex:
        pso_future = ex.submit(run_pso, windows, SEED)
        gp_future = ex.submit(run_gp, windows, SEED)
        pso_res = pso_future.result()
        gp_res, gp_instance = gp_future.result()

    # PSO results
    print(f"\nPSO best robust fitness: {pso_res['fitness']:.0f}")
    pso_val = validate(pso_res['best'], windows, strategy_type='pso')
    pso_win_rate = sum(r['wins'] for r in pso_val) / len(pso_val)
    avg_excess_pso = np.mean([r['excess'] for r in pso_val])
    print(f"  Validation win rate vs BH: {pso_win_rate:.1%}")
    print(f"  Avg excess return: {avg_excess_pso:.3f}")

    # GP results
    print(f"\nGP best robust fitness: {gp_res['fitness']:.0f}")
    gp_val = validate(gp_res['best'], windows, strategy_type='gp', gp_instance=gp_instance)
    gp_win_rate = sum(r['wins'] for r in gp_val) / len(gp_val)
    avg_excess_gp = np.mean([r['excess'] for r in gp_val])
    print(f"  Validation win rate vs BH: {gp_win_rate:.1%}")
    print(f"  Avg excess return: {avg_excess_gp:.3f}")

    out = {
        'n_windows': len(windows),
        'windows_per_fitness': WINDOWS_PER_FITNESS,
        'n_workers_inner': N_WORKERS_INNER,
        'n_workers_algo': N_WORKERS_ALGO,
        'pso': {
            'best_params': [float(x) for x in pso_res['best']],
            'robust_fitness': float(pso_res['fitness']),
            'win_rate': float(pso_win_rate),
            'avg_excess': float(avg_excess_pso),
            'per_window': pso_val,
        },
        'gp': {
            'robust_fitness': float(gp_res['fitness']),
            'win_rate': float(gp_win_rate),
            'avg_excess': float(avg_excess_gp),
            'per_window': gp_val,
        },
    }
    with open('robust_opt_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to robust_opt_results.json")


if __name__ == '__main__':
    main()
