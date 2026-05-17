"""
Exp 24: Fair head-to-head comparison of PSO vs GP-restricted on position_sma.
Same representation, same budget (1,500 evals), same data split, same seeds.
"""
import sys, json
sys.path.insert(0, '../..')

import numpy as np
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO

train, test, df = load_btc_data()
bh = buy_and_hold(test)
print(f"BH: ${bh:,.2f}")

BOUNDS = [(5, 200), (5, 200), (0.1, 100)]

def fitness(params):
    sig = VectorStrategy(params, 'position_sma').signals(train)
    return -backtest(train, sig)['final_cash']

seeds = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
results = []

for seed in seeds:
    np.random.seed(seed)
    pso = PSO(dim=3, bounds=BOUNDS, n_particles=30, max_iter=50,
              w=0.7, c1=1.5, c2=1.5, objective=fitness, verbose=False)
    best = pso.optimize()
    test_cash = backtest(test, VectorStrategy(best, 'position_sma').signals(test))['final_cash']
    results.append({'seed': seed, 'params': [float(p) for p in best],
                    'test': float(test_cash), 'beat_bh': test_cash > bh})
    print(f"seed={seed}: test=${test_cash:,.2f} beat_bh={test_cash > bh}")

mean_test = float(np.mean([r['test'] for r in results]))
std_test = float(np.std([r['test'] for r in results]))
cv = round(100 * std_test / mean_test, 1) if mean_test != 0 else 0

print(f"\nMean: ${mean_test:,.2f} ± ${std_test:,.2f}")
print(f"Beat BH: {sum(r['beat_bh'] for r in results)}/{len(results)}")
print(f"CV: {cv}%")

with open('results.json', 'w') as f:
    json.dump({
        'experiment': 'fair_pso_comparison',
        'bh': float(bh),
        'seeds': results,
        'summary': {'mean_test': mean_test, 'std_test': std_test,
                    'beat_bh_count': sum(r['beat_bh'] for r in results),
                    'cv_percent': cv}
    }, f, indent=2)

with open('paper_data.json', 'w') as f:
    json.dump({
        "pso": {"mean": mean_test, "std": std_test, "cv": cv,
                "beat_bh": sum(r['beat_bh'] for r in results), "n": 10},
        "gp_restricted": {"mean": 2219.61, "std": 230.02, "cv": 10.4,
                          "beat_bh": 5, "n": 10},
        "buy_and_hold": float(bh),
    }, f, indent=2)
print("Done. results.json + paper_data.json written.")

if __name__ == '__main__':
    pass