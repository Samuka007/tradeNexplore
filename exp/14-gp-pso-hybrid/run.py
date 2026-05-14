"""
Exp 14: GP structure + PSO parameter refinement hybrid.

Idea: GP optimizes the tree structure (which functions/terminals to use),
      but the numerical parameters (N, alpha, const) are randomly sampled.
      PSO can refine these numerical parameters for a fixed structure.

Workflow:
1. Run GP to get best tree structure
2. Extract all tunable numerical parameters from the tree
3. Fix the structure, use PSO to optimize the numerical parameters
4. Compare: GP-only vs GP+PSO vs PSO-only
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import json
sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP
from tiny_bot.pso import PSO
from tiny_bot.strategy import VectorStrategy

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
# Step 1: Run GP to get best tree
# ---------------------------------------------------------------------------
print("=" * 60)
print("Step 1: GP structure optimization")
print("=" * 60)

gp = GP(pop_size=75, generations=20, seed=SEED, parsimony_penalty=1000)
gp.max_depth = 7

def gp_fitness(tree):
    return backtest(train, gp.evaluate(tree, train))['final_cash']

gp_res = gp.optimize(gp_fitness)
gp_best = gp_res['best']
gp_train_cash = backtest(train, gp.evaluate(gp_best, train))['final_cash']
gp_test_cash = backtest(test, gp.evaluate(gp_best, test))['final_cash']
gp_tree_str = repr(gp_best)
gp_tree_size = gp._tree_size(gp_best)

print(f"GP best tree: {gp_tree_str}")
print(f"GP tree size: {gp_tree_size}")
print(f"GP train: ${gp_train_cash:,.0f}  test: ${gp_test_cash:,.0f}")

# ---------------------------------------------------------------------------
# Step 2: Extract tunable parameters from tree
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: Extract tunable parameters")
print("=" * 60)

def collect_params(node, path=""):
    """Collect all tunable parameters from a GP tree.
    Returns list of (path, param_name, current_value, bounds)
    """
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

tunable = collect_params(gp_best)
print(f"Found {len(tunable)} tunable parameters:")
for path, name, val, bounds in tunable:
    print(f"  {path}/{name} = {val:.3f}  bounds={bounds}")

# ---------------------------------------------------------------------------
# Step 3: Fix structure, optimize parameters with PSO
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: PSO parameter refinement")
print("=" * 60)

def apply_params(tree, param_values):
    """Apply new parameter values to a tree copy."""
    t = tree.copy()
    tunable_list = collect_params(t)
    for (path, name, _, bounds), new_val in zip(tunable_list, param_values):
        # Navigate to the node
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

def pso_fitness(param_values):
    tree = apply_params(gp_best, param_values)
    try:
        sig = gp.evaluate(tree, train)
        return backtest(train, sig)['final_cash']
    except Exception:
        return 0.0

bounds = [b for _, _, _, b in tunable]
init_vals = [v for _, _, v, _ in tunable]

print(f"Optimizing {len(bounds)} parameters with PSO...")
pso = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_res = pso.optimize(pso_fitness, bounds)

# Evaluate refined tree
refined_tree = apply_params(gp_best, pso_res['best'])
refined_train = backtest(train, gp.evaluate(refined_tree, train))['final_cash']
refined_test = backtest(test, gp.evaluate(refined_tree, test))['final_cash']

print(f"PSO refined params: {[float(x) for x in pso_res['best']]}")
print(f"Refined train: ${refined_train:,.0f}  test: ${refined_test:,.0f}")

# ---------------------------------------------------------------------------
# Step 4: Compare with PSO-only baseline
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: PSO-only baseline")
print("=" * 60)

pso_base = PSO(n_particles=30, max_iter=50, seed=SEED)
pso_base_res = pso_base.optimize(
    lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
    [(2, 200), (2, 200), (0.1, 100)]
)
pso_base_test = backtest(test, VectorStrategy(pso_base_res['best'], 'position_sma').signals(test))['final_cash']

print(f"PSO-only params: {pso_base_res['best'].tolist()}")
print(f"PSO-only test: ${pso_base_test:,.0f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

results = {
    "buy_and_hold_test": bh,
    "gp_only": {
        "train": gp_train_cash,
        "test": gp_test_cash,
        "tree_size": gp_tree_size,
        "tree": gp_tree_str,
    },
    "gp_pso_hybrid": {
        "train": refined_train,
        "test": refined_test,
        "n_params_optimized": len(tunable),
        "refined_params": [float(x) for x in pso_res['best']],
        "original_params": [float(v) for _, _, v, _ in tunable],
    },
    "pso_only": {
        "params": [float(x) for x in pso_base_res['best']],
        "test": pso_base_test,
    }
}

print(f"\nBuy-and-Hold:       ${bh:,.0f}")
print(f"GP-only:            ${gp_test_cash:,.0f}")
print(f"GP+PSO hybrid:      ${refined_test:,.0f}")
print(f"PSO-only:           ${pso_base_test:,.0f}")

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nSaved to results.json")
