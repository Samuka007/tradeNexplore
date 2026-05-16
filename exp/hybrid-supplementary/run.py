"""
Exp: Hybrid supplementary — PSO refinement across GP tree sizes.
Tests whether PSO refinement effect depends on tree size.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import sys
import json
import re
import os

sys.path.insert(0, '../..')
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.gp import GP, GPNode
from tiny_bot.pso import PSO

PSO_SEED = 42
N_WORKERS = 8

# ---------------------------------------------------------------------------
# Tree parser: GPNode repr string → GPNode tree
# ---------------------------------------------------------------------------
def _parse_terminal(tok: str) -> GPNode:
    m = re.match(r'^(sma|lma)\((\d+)\)$', tok)
    if m:
        n = GPNode(m.group(1), True)
        n.params['N'] = int(m.group(2))
        return n
    m = re.match(r'^ema\((\d+),([\d.]+)\)$', tok)
    if m:
        n = GPNode('ema', True)
        n.params['N'] = int(m.group(1))
        n.params['alpha'] = float(m.group(2))
        return n
    m = re.match(r'^(rsi|momentum|volatility)\((\d+)\)$', tok)
    if m:
        n = GPNode(m.group(1), True)
        n.params['N'] = int(m.group(2))
        return n
    if tok == 'price':
        return GPNode('price', True)
    try:
        v = float(tok)
        n = GPNode('const', True)
        n.params['value'] = v
        return n
    except ValueError:
        raise ValueError(f"Unknown terminal token: '{tok}'")


def _parse_tokens(tokens: list[str], idx: int) -> tuple[GPNode, int]:
    tok = tokens[idx]
    if tok != '(':
        return _parse_terminal(tok), idx + 1
    op = tokens[idx + 1]
    node = GPNode(op, False)
    idx += 2
    while idx < len(tokens) and tokens[idx] != ')':
        child, idx = _parse_tokens(tokens, idx)
        node.children.append(child)
    return node, idx + 1


def parse_tree(s: str) -> GPNode:
    """Parse a GPNode repr string back into a GPNode tree."""
    # Protect terminal parentheses so the tokenizer doesn't split them.
    # sma(183) → sma⟨183⟩; ema(45,0.30) → ema⟨45,0.30⟩
    s = re.sub(r'\b(sma|lma|rsi|momentum|volatility)\((\d+)\)',
               r'\1⟨\2⟩', s)
    s = re.sub(r'\bema\((\d+),([\d.]+)\)',
               r'ema⟨\1,\2⟩', s)

    tokens = []
    i = 0
    while i < len(s):
        if s[i] in '()':
            tokens.append(s[i])
            i += 1
        elif s[i].isspace():
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in '()' and not s[j].isspace():
                j += 1
            tokens.append(s[i:j])
            i = j

    # Restore terminal markers
    tokens = [t.replace('⟨', '(').replace('⟩', ')') for t in tokens]
    tree, end = _parse_tokens(tokens, 0)
    assert end == len(tokens), f"trailing tokens: {tokens[end:]}"
    return tree


# ---------------------------------------------------------------------------
# Parameter collection (identical to exp/14)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Selected trees spanning sizes 1–10 from exp/17 results
# ---------------------------------------------------------------------------
TREES = [
    {"size": 1,  "gp_only": 2364.4470828734525, "tree_str": "sma(183)",                                                                          "seed": 0},
    {"size": 3,  "gp_only": 2364.4470828734525, "tree_str": "(> volatility(81) rsi(91))",                                                         "seed": 0},
    {"size": 4,  "gp_only": 1651.123175467201,  "tree_str": "(> (ABS momentum(84)) momentum(142))",                                               "seed": 88},
    {"size": 5,  "gp_only": 3142.6350990071487, "tree_str": "(< ema(53,0.91) (- volatility(15) momentum(30)))",                                   "seed": 88},
    {"size": 7,  "gp_only": 1720.081807072984,  "tree_str": "(> (- price (- price rsi(139))) rsi(139))",                                          "seed": 88},
    {"size": 9,  "gp_only": 3142.6350990071487, "tree_str": "(> (< momentum(44) lma(162)) (> (+ price volatility(63)) ema(45,0.30)))",            "seed": 42},
    {"size": 10, "gp_only": 1606.0507213394196, "tree_str": "(IF sma(195) (< (- volatility(143) (- sma(195) volatility(138))) sma(94)) momentum(31))", "seed": 0},
]


# ---------------------------------------------------------------------------
# Load data (same as all prior experiments)
# ---------------------------------------------------------------------------
print("Loading BTC data...")
df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
bh = buy_and_hold(test)
print(f"Buy-and-Hold test: ${bh:,.0f}\n")

gp = GP(pop_size=75, generations=20, seed=PSO_SEED, parsimony_penalty=1000)
results = []

for i, entry in enumerate(TREES):
    tree_str = entry["tree_str"]
    size = entry["size"]
    gp_only_recorded = entry["gp_only"]

    print(f"{'='*60}")
    print(f"Tree {i+1}/{len(TREES)}: size={size}")
    print(f"  {tree_str}")
    print(f"{'='*60}")

    # Parse tree
    tree = parse_tree(tree_str)
    parsed_size = gp._tree_size(tree)
    assert parsed_size == size, f"size mismatch: parsed {parsed_size} != expected {size}"

    # Verify GP-only test return
    gp_test = backtest(test, gp.evaluate(tree, test))['final_cash']
    diff = abs(gp_test - gp_only_recorded)
    if diff > 1.0:
        print(f"  ⚠ GP test differs from recorded: {gp_test:.0f} vs {gp_only_recorded:.0f} (Δ={diff:.0f})")
    else:
        print(f"  GP-only test: ${gp_test:,.0f} (verified)")

    # Collect tunable params
    tunable = collect_params(tree)
    n_params = len(tunable)
    print(f"  Tunable params: {n_params}")

    if n_params == 0:
        print(f"  ⚠ No tunable params — skipping PSO")
        results.append({
            "size": size,
            "gp_only": gp_only_recorded,
            "hybrid": gp_only_recorded,
            "tree_str": tree_str,
            "n_params": 0,
            "pso_improvement": 0.0,
        })
        continue

    # Run PSO parameter refinement
    bounds = [b for _, _, _, b in tunable]
    init_vals = [v for _, _, v, _ in tunable]
    print(f"  Initial params: {[round(v,2) for v in init_vals]}")

    def pso_fitness(param_values):
        t = apply_params(tree, param_values)
        try:
            sig = gp.evaluate(t, train)
            return backtest(train, sig)['final_cash']
        except Exception:
            return 0.0

    print(f"  Running PSO (30 particles × 50 iters)...")
    np.random.seed(PSO_SEED)
    pso = PSO(n_particles=30, max_iter=50, seed=PSO_SEED)
    pso_res = pso.optimize(pso_fitness, bounds, n_workers=N_WORKERS)

    # Evaluate refined tree on test
    refined_tree = apply_params(tree, pso_res['best'])
    hybrid_test = backtest(test, gp.evaluate(refined_tree, test))['final_cash']
    refined_params = [float(x) for x in pso_res['best']]
    improvement = hybrid_test - gp_only_recorded

    print(f"  Refined params: {[round(p,1) for p in refined_params]}")
    print(f"  Hybrid test: ${hybrid_test:,.0f}  (Δ={improvement:+.0f}, {improvement/gp_only_recorded*100:+.1f}%)")

    results.append({
        "size": size,
        "gp_only": gp_only_recorded,
        "hybrid": hybrid_test,
        "tree_str": tree_str,
        "n_params": n_params,
        "init_params": [float(v) for v in init_vals],
        "refined_params": refined_params,
        "pso_improvement": improvement,
        "pso_improvement_pct": improvement / gp_only_recorded * 100,
    })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SUMMARY: GP-only vs GP+PSO by tree size")
print(f"{'='*60}")
print(f"{'Size':>5} {'GP-only':>10} {'Hybrid':>10} {'Δ':>10} {'Δ%':>8} {'#Params':>8}")
print("-" * 56)
for r in results:
    imp = r["pso_improvement"]
    imp_pct = imp / r["gp_only"] * 100
    print(f"{r['size']:>5} ${r['gp_only']:>9,.0f} ${r['hybrid']:>9,.0f} {imp:>+10,.0f} {imp_pct:>+7.1f}% {r['n_params']:>8}")

print(f"\nBuy-and-Hold: ${bh:,.0f}")

# Save results
output = {"bh": bh, "trees": results}
os.makedirs(os.path.dirname('results.json') or '.', exist_ok=True)
with open('results.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to results.json")
