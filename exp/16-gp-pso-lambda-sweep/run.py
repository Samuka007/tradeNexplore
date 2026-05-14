"""
Exp 16: GP+PSO Hybrid with varying parsimony penalty λ.

Hypothesis: Lower λ allows GP to discover more complex tree structures,
which provide more tunable parameters for PSO to optimize.

Compare λ ∈ {0, 100, 500, 1000}:
1. Run GP to get tree structure
2. Extract all tunable parameters
3. PSO optimizes parameters with fixed structure
4. Evaluate test performance
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import sys
sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP, GPNode
from tiny_bot.pso import PSO

SEED = 42
np.random.seed(SEED)

# Load data
df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)

def collect_params(node, path=""):
    params = []
    if node.terminal:
        if node.value in ("sma", "lma", "rsi", "momentum", "volatility"):
            params.append((path, "N", node.params["N"], (5, 200)))
        elif node.value == "ema":
            params.append((path, "N", node.params["N"], (5, 200)))
            params.append((path, "alpha", node.params["alpha"], (0.01, 0.99)))
        elif node.value == "const":
            params.append((path, "value", node.params["value"], (-1.0, 1.0)))
    else:
        for i, child in enumerate(node.children):
            params.extend(collect_params(child, f"{path}/{node.value}:{i}"))
    return params

def apply_params(tree, param_values):
    t = tree.copy()
    tunable_list = collect_params(t)
    for (path, name, _, bounds), new_val in zip(tunable_list, param_values):
        parts = path.strip("/").split("/")
        node = t
        for part in parts:
            if part:
                op, idx = part.split(":")
                node = node.children[int(idx)]
        if name == "N":
            new_val = int(round(float(new_val)))
        node.params[name] = new_val
    return t

print("=" * 70)
print("GP+PSO Hybrid: λ Sweep")
print("=" * 70)
print(f"Buy-and-Hold: ${bh:,.0f}\n")

results = []

for lam in [0, 100, 500, 1000]:
    print(f"--- λ = {lam} ---")
    
    # Step 1: GP structure
    gp = GP(pop_size=75, generations=20, seed=SEED, parsimony_penalty=lam)
    gp.max_depth = 7
    
    def gp_fitness(tree):
        return backtest(train, gp.evaluate(tree, train))['final_cash']
    
    gp_res = gp.optimize(gp_fitness)
    gp_best = gp_res['best']
    gp_train = backtest(train, gp.evaluate(gp_best, train))['final_cash']
    gp_test = backtest(test, gp.evaluate(gp_best, test))['final_cash']
    tree_size = gp._tree_size(gp_best)
    tree_str = repr(gp_best)
    
    print(f"GP tree ({tree_size} nodes): {tree_str}")
    print(f"GP-only: train=${gp_train:.0f} test=${gp_test:.0f}")
    
    # Step 2: Extract parameters
    tunable = collect_params(gp_best)
    print(f"Tunable params: {len(tunable)}")
    
    # Step 3: PSO refinement
    if len(tunable) > 0:
        bounds = [b for _, _, _, b in tunable]
        
        def pso_fitness(param_values):
            tree = apply_params(gp_best, param_values)
            try:
                sig = gp.evaluate(tree, train)
                return backtest(train, sig)['final_cash']
            except Exception:
                return 0.0
        
        pso = PSO(n_particles=30, max_iter=50, seed=SEED)
        pso_res = pso.optimize(pso_fitness, bounds)
        
        refined_tree = apply_params(gp_best, pso_res['best'])
        refined_train = backtest(train, gp.evaluate(refined_tree, train))['final_cash']
        refined_test = backtest(test, gp.evaluate(refined_tree, test))['final_cash']
        
        print(f"PSO refined: train=${refined_train:.0f} test=${refined_test:.0f}")
        print(f"Refined params: {[float(x) for x in pso_res['best']]}")
    else:
        refined_train = gp_train
        refined_test = gp_test
        print("No tunable params, skipping PSO")
    
    results.append({
        "lambda": lam,
        "tree_size": tree_size,
        "tree": tree_str,
        "gp_train": gp_train,
        "gp_test": gp_test,
        "refined_train": refined_train,
        "refined_test": refined_test,
        "n_params": len(tunable),
        "param_bounds": [str(b) for _, _, _, b in tunable],
    })
    print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'λ':>6} {'Tree':>5} {'Params':>7} {'GP-only':>10} {'GP+PSO':>10} {'Δ':>8}")
print("-" * 56)
for r in results:
    delta = r['refined_test'] - r['gp_test']
    print(f"{r['lambda']:>6} {r['tree_size']:>5} {r['n_params']:>7} ${r['gp_test']:>8,.0f} ${r['refined_test']:>8,.0f} ${delta:>+6,.0f}")

with open('results.json', 'w') as f:
    json.dump({"bh": bh, "results": results}, f, indent=2)

print(f"\nBuy-and-Hold: ${bh:.0f}")
print("Saved to results.json")
