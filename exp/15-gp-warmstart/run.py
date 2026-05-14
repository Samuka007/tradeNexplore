"""
Exp 15: GP Warm-start Initialization

Hypothesis: Injecting human-designed trading rules into GP's initial population
will improve convergence quality and reduce seed-sensitivity.

Compare: Random init vs Warm-start init (50% human rules + 50% random)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP, GPNode

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)

# ---------------------------------------------------------------------------
# Build human-designed seed trees
# ---------------------------------------------------------------------------
def make_terminal(name, **kwargs):
    n = GPNode(name, terminal=True)
    n.params = kwargs
    return n

def make_fun(name, *children):
    n = GPNode(name, terminal=False)
    n.children = list(children)
    return n

seed_trees = [
    # 1. Classic SMA crossover
    make_fun(">", make_terminal("sma", N=50), make_terminal("sma", N=200)),
    # 2. Price above long-term MA
    make_fun(">", make_terminal("price"), make_terminal("lma", N=100)),
    # 3. RSI oversold bounce
    make_fun(">", make_terminal("rsi", N=14), make_terminal("const", value=30)),
    # 4. Volatility regime shift
    make_fun(">", make_terminal("volatility", N=20), make_terminal("volatility", N=50)),
    # 5. Short-term momentum positive
    make_fun(">", make_terminal("momentum", N=10), make_terminal("const", value=0)),
    # 6. Price below short SMA (mean reversion)
    make_fun("<", make_terminal("price"), make_terminal("sma", N=20)),
    # 7. Trend + RSI combined
    make_fun("AND",
        make_fun(">", make_terminal("sma", N=50), make_terminal("sma", N=200)),
        make_fun(">", make_terminal("rsi", N=14), make_terminal("const", value=30))),
    # 8. High-volatility trend follow
    make_fun("IF",
        make_fun(">", make_terminal("volatility", N=20), make_terminal("const", value=0.05)),
        make_fun(">", make_terminal("price"), make_terminal("lma", N=100)),
        make_terminal("price")),
    # 9. Momentum + volatility filter
    make_fun("AND",
        make_fun(">", make_terminal("momentum", N=20), make_terminal("const", value=0)),
        make_fun(">", make_terminal("volatility", N=10), make_terminal("volatility", N=30))),
    # 10. EMA fast vs slow
    make_fun(">", make_terminal("ema", N=12, alpha=0.15), make_terminal("ema", N=26, alpha=0.07)),
]

print("=" * 60)
print("GP Warm-start Initialization Experiment")
print("=" * 60)
print(f"Buy-and-Hold: ${bh:,.0f}")

# ---------------------------------------------------------------------------
# Test multiple seeds with both init strategies
# ---------------------------------------------------------------------------
seeds = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
results = {"random": [], "warmstart": []}

for seed in seeds:
    np.random.seed(seed)
    
    # --- Random init ---
    gp_rand = GP(pop_size=50, generations=30, seed=seed, parsimony_penalty=1000)
    gp_rand.max_depth = 7
    r_rand = gp_rand.optimize(lambda tree: backtest(train, gp_rand.evaluate(tree, train))['final_cash'])
    test_rand = backtest(test, gp_rand.evaluate(r_rand['best'], test))['final_cash']
    results["random"].append({"seed": seed, "test": test_rand, "tree_size": gp_rand._tree_size(r_rand['best'])})
    
    # --- Warm-start init ---
    gp_warm = GP(pop_size=50, generations=30, seed=seed, parsimony_penalty=1000)
    gp_warm.max_depth = 7
    # Build population: 10 seeds + 40 random
    pop = [t.copy() for t in seed_trees]
    while len(pop) < gp_warm.pop_size:
        pop.append(gp_warm._random_tree())
    
    # Monkey-patch optimize to use custom initial population
    def optimize_with_pop(self, fitness_fn, initial_pop, n_workers=1):
        pop = initial_pop
        def _eval_all(population):
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return list(ex.map(fitness_fn, population))
            return [fitness_fn(ind) for ind in population]
        raw_fit = _eval_all(pop)
        fit = [f - self.parsimony_penalty * self._tree_size(ind) for f, ind in zip(raw_fit, pop)]
        history = []
        for _ in range(self.generations):
            new = []
            best_idx = int(np.argmax(fit))
            new.append(pop[best_idx].copy())
            while len(new) < self.pop_size:
                r = np.random.rand()
                if r < 0.9:
                    p1 = self._select(pop, fit)
                    p2 = self._select(pop, fit)
                    new.append(self._crossover(p1, p2))
                elif r < 0.9 + 0.1:
                    p = self._select(pop, fit)
                    new.append(self._mutate(p))
                else:
                    new.append(self._select(pop, fit).copy())
            pop = new
            raw_fit = _eval_all(pop)
            fit = [f - self.parsimony_penalty * self._tree_size(ind) for f, ind in zip(raw_fit, pop)]
            history.append(float(max(fit)))
        best_idx = int(np.argmax(fit))
        return {"best": pop[best_idx], "fitness": fit[best_idx], "history": history}
    
    r_warm = optimize_with_pop(gp_warm, lambda tree: backtest(train, gp_warm.evaluate(tree, train))['final_cash'], pop)
    test_warm = backtest(test, gp_warm.evaluate(r_warm['best'], test))['final_cash']
    results["warmstart"].append({"seed": seed, "test": test_warm, "tree_size": gp_warm._tree_size(r_warm['best'])})
    
    print(f"seed={seed}: random=${test_rand:.0f} warm=${test_warm:.0f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

rand_tests = [r['test'] for r in results['random']]
warm_tests = [r['test'] for r in results['warmstart']]
rand_beats = sum(1 for t in rand_tests if t > bh)
warm_beats = sum(1 for t in warm_tests if t > bh)

print(f"\nBuy-and-Hold: ${bh:.0f}")
print(f"Random init:   mean=${np.mean(rand_tests):.0f} std=${np.std(rand_tests):.0f} beats BH={rand_beats}/{len(seeds)}")
print(f"Warm-start:    mean=${np.mean(warm_tests):.0f} std=${np.std(warm_tests):.0f} beats BH={warm_beats}/{len(seeds)}")

out = {
    "bh": bh,
    "seeds": seeds,
    "random": results['random'],
    "warmstart": results['warmstart'],
    "summary": {
        "random_mean": float(np.mean(rand_tests)),
        "random_std": float(np.std(rand_tests)),
        "random_beats": rand_beats,
        "warm_mean": float(np.mean(warm_tests)),
        "warm_std": float(np.std(warm_tests)),
        "warm_beats": warm_beats,
    }
}

with open('results.json', 'w') as f:
    json.dump(out, f, indent=2)

print("\nSaved to results.json")
