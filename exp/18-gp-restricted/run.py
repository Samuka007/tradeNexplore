"""
Exp 18: GP restricted to position_sma 3-parameter space.

Causal control experiment: when GP searches the same parametric space as PSO,
does it converge to the same basin?
"""
import sys, json, os
sys.path.insert(0, '../..')

import numpy as np
from tiny_bot.data import load_btc_data
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy

# ---------------------------------------------------------------------------
# Load data (unified source)
# ---------------------------------------------------------------------------
train, test, df = load_btc_data()
bh = buy_and_hold(test)

print(f"Train: {len(train)} days, Test: {len(test)} days")
print(f"Buy-and-hold test: ${bh:,.2f}")

# ---------------------------------------------------------------------------
# Strategy bounds (same as PSO position_sma)
# ---------------------------------------------------------------------------
BOUNDS = [(5, 200), (5, 200), (0.1, 100)]

# ---------------------------------------------------------------------------
# GP with restricted tree = single param node
# ---------------------------------------------------------------------------
class GPParam:
    """Single-node GP individual: just 3 params [fast, slow, scale]."""
    def __init__(self, params=None):
        if params is None:
            self.params = np.array([
                np.random.uniform(*BOUNDS[0]),
                np.random.uniform(*BOUNDS[1]),
                np.random.uniform(*BOUNDS[2]),
            ])
        else:
            self.params = np.array(params, dtype=float)

    def copy(self):
        return GPParam(self.params.copy())

    def tree_size(self):
        return 1

class GPParamSearch:
    """GA-style search over 3D param space using tournament selection."""
    def __init__(self, pop_size=75, generations=20, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.pop_size = pop_size
        self.generations = generations

    def optimize(self, fitness_fn):
        pop = [GPParam() for _ in range(self.pop_size)]
        fit = [fitness_fn(p) for p in pop]
        history = []
        for _ in range(self.generations):
            new = []
            best_idx = int(np.argmax(fit))
            new.append(pop[best_idx].copy())
            while len(new) < self.pop_size:
                r = np.random.rand()
                if r < 0.9:          # crossover
                    p1 = self._select(pop, fit)
                    p2 = self._select(pop, fit)
                    child = self._crossover(p1, p2)
                    new.append(child)
                elif r < 0.95:       # mutation
                    p = self._select(pop, fit)
                    new.append(self._mutate(p))
                else:                # reproduction
                    new.append(self._select(pop, fit).copy())
            pop = new
            fit = [fitness_fn(p) for p in pop]
            history.append(float(max(fit)))
        best_idx = int(np.argmax(fit))
        return {"best": pop[best_idx], "fitness": fit[best_idx], "history": history}

    def _select(self, pop, fit):
        idxs = np.random.choice(len(pop), size=3, replace=False)
        return pop[idxs[np.argmax([fit[i] for i in idxs])]].copy()

    def _crossover(self, p1, p2):
        mask = np.random.rand(3) < 0.5
        child = GPParam(p1.params.copy())
        child.params[mask] = p2.params[mask]
        return child

    def _mutate(self, p):
        child = GPParam(p.params.copy())
        idx = np.random.randint(0, 3)
        sigma = [20, 20, 5][idx]
        child.params[idx] += np.random.normal(0, sigma)
        child.params[0] = np.clip(child.params[0], *BOUNDS[0])
        child.params[1] = np.clip(child.params[1], *BOUNDS[1])
        child.params[2] = np.clip(child.params[2], *BOUNDS[2])
        return child

# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------
def fitness(ind):
    sig = VectorStrategy(ind.params, 'position_sma').signals(train)
    return backtest(train, sig)['final_cash']

# ---------------------------------------------------------------------------
# Run 10 seeds
# ---------------------------------------------------------------------------
seeds = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
results = []

for seed in seeds:
    np.random.seed(seed)
    gp = GPParamSearch(pop_size=75, generations=20, seed=seed)
    out = gp.optimize(fitness)
    best = out['best']

    train_sig = VectorStrategy(best.params, 'position_sma').signals(train)
    test_sig  = VectorStrategy(best.params, 'position_sma').signals(test)
    train_ret = backtest(train, train_sig)['final_cash']
    test_ret  = backtest(test,  test_sig)['final_cash']

    results.append({
        'seed': seed,
        'params': best.params.tolist(),
        'train': float(train_ret),
        'test': float(test_ret),
        'beat_bh': bool(test_ret > bh),
    })
    print(f"seed={seed}: params={best.params.round(2).tolist()}, test=${test_ret:,.2f}, beat_bh={test_ret>bh}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open('results.json', 'w') as f:
    json.dump({
        'experiment': 'gp_restricted_position_sma',
        'seeds': results,
        'buy_and_hold': float(bh),
        'summary': {
            'mean_test': float(np.mean([r['test'] for r in results])),
            'std_test': float(np.std([r['test'] for r in results])),
            'beat_bh_count': sum(r['beat_bh'] for r in results),
        }
    }, f, indent=2)

print(f"\nMean test: ${np.mean([r['test'] for r in results]):,.2f} ± ${np.std([r['test'] for r in results]):,.2f}")
print(f"Beat BH: {sum(r['beat_bh'] for r in results)}/{len(results)}")
