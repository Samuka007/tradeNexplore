"""run_early_stop.py — Early stopping experiment comparing structural stop vs control.

Runs two conditions (control vs structural early stopping), 10 seeds each.
GP config: λ=500, depth=5, pop=75, gen=30.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tiny_bot.backtest import backtest, buy_and_hold


def run_experiment():
    # Import early-stopping GP
    from gp_early_stop import GP

    # Config
    SEEDS = list(range(10))
    POP_SIZE = 75
    GENERATIONS = 30
    PARSIMONY = 500  # λ=500
    MAX_DEPTH = 5

    # Load BTC data (same split as all experiments)
    df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31',
                     progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    bh = buy_and_hold(test)

    print(f"Buy-and-Hold: ${bh:.0f}")
    n_seeds = len(SEEDS)
    print(f"Seeds: {n_seeds} | Pop: {POP_SIZE} | Gen: {GENERATIONS} | λ: {PARSIMONY} | Depth: {MAX_DEPTH}\n")

    results = {'bh': bh, 'conditions': {}}

    for condition_name, early_stop in [('control', False), ('structural_stop', True)]:
        label = condition_name.upper()
        print(f"{'='*70}")
        print(f"CONDITION {label}: early_stop={early_stop}")
        print(f"{'='*70}")

        seeds_data = []
        for seed in SEEDS:
            t0 = time.time()
            gp = GP(pop_size=POP_SIZE, generations=GENERATIONS, seed=seed,
                    parsimony_penalty=PARSIMONY)
            gp.max_depth = MAX_DEPTH

            def fitness_fn(tree):
                return backtest(train, gp.evaluate(tree, train))['final_cash']

            r = gp.optimize(fitness_fn, early_stop=early_stop)
            best = r['best']

            train_cash = backtest(train, gp.evaluate(best, train))['final_cash']
            test_cash = backtest(test, gp.evaluate(best, test))['final_cash']
            tree_size = gp._tree_size(best)
            tree_str = repr(best)
            metrics = gp.structural_metrics(best)
            elapsed = time.time() - t0

            seeds_data.append({
                'seed': seed,
                'train_cash': train_cash,
                'test_cash': test_cash,
                'tree_size': tree_size,
                'tree': tree_str,
                'stop_gen': r['stop_gen'],
                'stop_reason': r['stop_reason'],
                'metrics': metrics,
                'elapsed_s': round(elapsed, 1),
            })

            status = "BEATS BH" if test_cash > bh else ""
            print(f"  seed={seed:2d} | train=${train_cash:,.0f} test=${test_cash:,.0f} "
                  f"tree={tree_size:2d} stop_gen={r['stop_gen']:2d} "
                  f"reason={r['stop_reason'][:30]:30s} {status}")

        test_vals = [d['test_cash'] for d in seeds_data]
        stop_gens = [d['stop_gen'] for d in seeds_data]
        beats = sum(1 for v in test_vals if v > bh)

        cond_summary = {
            'seeds': seeds_data,
            'mean_test': float(np.mean(test_vals)),
            'std_test': float(np.std(test_vals)),
            'min_test': float(np.min(test_vals)),
            'max_test': float(np.max(test_vals)),
            'beats_bh': beats,
            'win_rate_vs_bh': beats / n_seeds,
            'mean_stop_gen': float(np.mean(stop_gens)),
            'mean_elapsed_s': float(np.mean([d['elapsed_s'] for d in seeds_data])),
            'mean_train': float(np.mean([d['train_cash'] for d in seeds_data])),
            'mean_tree_size': float(np.mean([d['tree_size'] for d in seeds_data])),
        }
        results['conditions'][condition_name] = cond_summary

        print(f"\n  Mean test: ${cond_summary['mean_test']:,.0f} ± ${cond_summary['std_test']:,.0f}")
        print(f"  Beats BH: {beats}/{n_seeds} ({cond_summary['win_rate_vs_bh']:.1%})")
        print(f"  Mean stop gen: {cond_summary['mean_stop_gen']:.1f}")
        print(f"  Mean train: ${cond_summary['mean_train']:,.0f}")
        print(f"  Mean tree size: {cond_summary['mean_tree_size']:.1f}")

    # Compare conditions
    control = results['conditions']['control']
    stop = results['conditions']['structural_stop']

    print(f"\n{'='*70}")
    print("COMPARISON: Control vs Structural Early Stop")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Control':>15} {'Stop':>15} {'Δ':>15}")
    print("-" * 70)
    print(f"{'Mean Test $':<25} ${control['mean_test']:>13,.0f} ${stop['mean_test']:>13,.0f} ${stop['mean_test'] - control['mean_test']:>14,.0f}")
    print(f"{'Std Test $':<25} ${control['std_test']:>13,.0f} ${stop['std_test']:>13,.0f}")
    print(f"{'Win Rate vs BH':<25} {control['win_rate_vs_bh']:>14.1%} {stop['win_rate_vs_bh']:>14.1%}")
    print(f"{'Mean Stop Gen':<25} {control['mean_stop_gen']:>14.1f} {stop['mean_stop_gen']:>14.1f}")
    print(f"{'Mean Train $':<25} ${control['mean_train']:>13,.0f} ${stop['mean_train']:>13,.0f}")
    print(f"{'Mean Tree Size':<25} {control['mean_tree_size']:>14.1f} {stop['mean_tree_size']:>14.1f}")

    # Save
    out_path = Path(__file__).resolve().parent / 'early_stop_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    run_experiment()
