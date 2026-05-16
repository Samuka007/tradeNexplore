"""analyze_existing.py — Overfitting Geometry analysis of existing GP trees.

Parses tree representations from Exp 09, 11, 12, 17 and computes structural
metrics, then correlates them with train-test gaps.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr, spearmanr

# Import tiny_bot from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tiny_bot.gp import GP, GPNode
from tiny_bot.backtest import backtest

# ---------------------------------------------------------------------------
# Tree parser — reconstructs GPNode from repr() output
# ---------------------------------------------------------------------------

# Terminal patterns: name(params) or float literal or bare name
_RE_TERM_FUNC = re.compile(
    r'^(sma|lma|ema|rsi|momentum|volatility)'
    r'\((\d+)(?:,(\d+\.\d+))?\)$'  # ema has N,alpha; others just N
)
_RE_CONST = re.compile(r'^-?\d+\.\d+$')
_RE_BARE = re.compile(r'^price$')


def _tokenize(s: str) -> list[str]:
    """Tokenize a tree repr string into tokens.

    Top-level '(' and ')' are separate tokens. Everything else
    between spaces is a single token (including sma(195), ema(157,0.27), etc.).
    """
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '(' or ch == ')':
            tokens.append(ch)
            i += 1
        elif ch.isspace():
            i += 1
        else:
            # Read until space or top-level paren
            j = i
            depth = 0
            while j < len(s):
                c = s[j]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth < 0:
                        break
                elif c.isspace() and depth == 0:
                    break
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def parse_tree(repr_str: str) -> GPNode:
    """Parse a GP tree from its repr() string back to a GPNode."""
    tokens = _tokenize(repr_str)

    def _parse(idx: int) -> tuple[GPNode, int]:
        tok = tokens[idx]
        if tok != '(':
            # Terminal
            return _parse_terminal(tok), idx + 1

        # Function node: (op children...)
        idx += 1  # skip '('
        op = tokens[idx]
        idx += 1   # skip op

        node = GPNode(op, terminal=False)

        # Known arities for precise child parsing
        arity = {"+": 2, "-": 2, "*": 2, "/": 2, "ABS": 1, "MAX": 2,
                 "MIN": 2, ">": 2, "<": 2, "AND": 2, "IF": 3}.get(op)
        if arity is not None:
            for _ in range(arity):
                child, idx = _parse(idx)
                node.children.append(child)
        else:
            # Unknown — consume until ')'
            while idx < len(tokens) and tokens[idx] != ')':
                child, idx = _parse(idx)
                node.children.append(child)

        # skip ')'
        if idx < len(tokens) and tokens[idx] == ')':
            idx += 1
        return node, idx

    def _parse_terminal(tok: str) -> GPNode:
        # Check function-like terminals: sma(195), ema(157,0.27), etc.
        m = _RE_TERM_FUNC.match(tok)
        if m:
            name = m.group(1)
            node = GPNode(name, terminal=True)
            if name == 'ema':
                node.params['N'] = int(m.group(2))
                node.params['alpha'] = float(m.group(3))
            else:
                node.params['N'] = int(m.group(2))
            return node

        # Check const
        if _RE_CONST.match(tok):
            node = GPNode('const', terminal=True)
            node.params['value'] = float(tok)
            return node

        # Bare name (price)
        if _RE_BARE.match(tok):
            return GPNode(tok, terminal=True)

        raise ValueError(f"Unrecognised terminal token: {tok!r}")

    tree, idx = _parse(0)
    if idx != len(tokens):
        raise ValueError(f"Trailing tokens after parse: {tokens[idx:]}")
    return tree


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def compute_metrics(tree: GPNode) -> dict:
    """Compute structural metrics for a GP tree."""
    # Collect all nodes
    def _all_nodes(node):
        nodes = [node]
        for c in node.children:
            nodes.extend(_all_nodes(c))
        return nodes

    nodes = _all_nodes(tree)
    total = len(nodes)
    internal = sum(1 for n in nodes if not n.terminal)
    terminals = total - internal

    # Depth
    def _depth(node):
        if not node.children:
            return 0
        return 1 + max(_depth(c) for c in node.children)
    depth = _depth(tree)

    # Counts by value
    if_count = sum(1 for n in nodes if n.value == 'IF')
    and_count = sum(1 for n in nodes if n.value == 'AND')

    # Terminals by type
    terminal_types = set()
    const_count = 0
    for n in nodes:
        if n.terminal:
            terminal_types.add(n.value)
            if n.value == 'const':
                const_count += 1

    return {
        'tree_size': total,
        'depth': depth,
        'nesting_ratio': internal / total if total > 0 else 0,
        'constant_ratio': const_count / terminals if terminals > 0 else 0,
        'if_count': if_count,
        'and_count': and_count,
        'unique_terminals': len(terminal_types),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_btc_data():
    """Load BTC data with same split as Exp 17."""
    df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31',
                     progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    return train, test


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def load_all_trees():
    """Load all trees from existing experiments.

    Returns list of dicts with keys: train_cash, test_cash, tree, metrics, source.
    """
    base = Path(__file__).resolve().parent.parent
    gp = GP()
    records = []

    # --- Exp 09: 15 trees with train_cash, test_cash ---
    with open(base / '09-gp-parsimony' / 'results.json') as f:
        data09 = json.load(f)
    for ex in data09['experiments']:
        try:
            tree = parse_tree(ex['tree_repr'])
            metrics = compute_metrics(tree)
            records.append({
                'source': 'exp09',
                'train_cash': ex['train_cash'],
                'test_cash': ex['test_cash'],
                'tree_repr': ex['tree_repr'],
                'tree': tree,
                'metrics': metrics,
            })
        except Exception as e:
            print(f"  [skip exp09] {e}: {ex['tree_repr'][:80]}")

    # --- Exp 11: 4 trees with train_cash, test_cash ---
    with open(base / '11-gp-tradeoff' / 'results.json') as f:
        data11 = json.load(f)
    for ex in data11['experiments']:
        try:
            tree = parse_tree(ex['tree_repr'])
            metrics = compute_metrics(tree)
            records.append({
                'source': 'exp11',
                'train_cash': ex['train_cash'],
                'test_cash': ex['test_cash'],
                'tree_repr': ex['tree_repr'],
                'tree': tree,
                'metrics': metrics,
            })
        except Exception as e:
            print(f"  [skip exp11] {e}: {ex['tree_repr'][:80]}")

    # --- Exp 12: 3 trees with train_cash, test_cash ---
    with open(base / '12-gp-functionset' / 'results.json') as f:
        data12 = json.load(f)
    for ex in data12['experiments']:
        try:
            tree = parse_tree(ex['tree_repr'])
            metrics = compute_metrics(tree)
            records.append({
                'source': 'exp12',
                'train_cash': ex['train_cash'],
                'test_cash': ex['test_cash'],
                'tree_repr': ex['tree_repr'],
                'tree': tree,
                'metrics': metrics,
            })
        except Exception as e:
            print(f"  [skip exp12] {e}: {ex['tree_repr'][:80]}")

    # --- Exp 17: 42 trees; test only, must compute train ---
    with open(base / '17-systematic-hyperparam' / 'results.json') as f:
        data17 = json.load(f)

    # Load train/test split for backtesting
    print("Loading BTC data for Exp 17 train backtest...")
    train_prices, test_prices = load_btc_data()

    for ex in data17['results']:
        try:
            tree = parse_tree(ex['tree'])
            metrics = compute_metrics(tree)

            # Compute train_cash via backtest
            train_sig = gp.evaluate(tree, train_prices)
            train_cash = backtest(train_prices, train_sig)['final_cash']

            records.append({
                'source': 'exp17',
                'train_cash': train_cash,
                'test_cash': ex['test'],
                'tree_repr': ex['tree'],
                'tree': tree,
                'metrics': metrics,
            })
        except Exception as e:
            print(f"  [skip exp17] {e}: {ex['tree'][:80]}")

    return records


def analyze(records, output_path: str):
    """Compute correlations, regression, and save results."""
    n = len(records)
    train_test_gap = np.array([r['train_cash'] - r['test_cash'] for r in records],
                               dtype=np.float64)

    metric_names = ['tree_size', 'depth', 'nesting_ratio', 'constant_ratio',
                    'if_count', 'and_count', 'unique_terminals']
    metric_arrays = {}
    for m in metric_names:
        metric_arrays[m] = np.array([r['metrics'][m] for r in records],
                                     dtype=np.float64)

    # Correlations
    corr_results = {}
    for m in metric_names:
        vals = metric_arrays[m]
        mask = np.isfinite(vals) & np.isfinite(train_test_gap)
        if mask.sum() < 3:
            corr_results[m] = {'pearson_r': None, 'pearson_p': None,
                               'spearman_r': None, 'spearman_p': None}
            continue
        pr, pp = pearsonr(vals[mask], train_test_gap[mask])
        sr, sp = spearmanr(vals[mask], train_test_gap[mask])
        corr_results[m] = {
            'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp),
        }

    # Linear regression: gap ~ tree_size + depth + nesting_ratio + constant_ratio + if_count
    X_cols = ['tree_size', 'depth', 'nesting_ratio', 'constant_ratio', 'if_count']
    X = np.column_stack([metric_arrays[c] for c in X_cols])
    y = train_test_gap

    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]

    # OLS via normal equations (no sklearn dependency)
    X_design = np.column_stack([np.ones(len(X_clean)), X_clean])
    coef, _residuals, _rank, _sv = np.linalg.lstsq(X_design, y_clean, rcond=None)
    y_pred = X_design @ coef
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

    # Coefficient summary
    coef_summary = {col: {'coef': float(coef[i + 1])}
                    for i, col in enumerate(X_cols)}
    coef_summary['intercept'] = float(coef[0])

    regression = {
        'r2': r2,
        'n_samples': int(mask.sum()),
        'predictors': X_cols,
        'coefficients': coef_summary,
    }

    # Build per-tree records (serializable)
    tree_records = []
    for r in records:
        tree_records.append({
            'source': r['source'],
            'train_cash': r['train_cash'],
            'test_cash': r['test_cash'],
            'train_test_gap': r['train_cash'] - r['test_cash'],
            'tree_repr': r['tree_repr'],
            'metrics': r['metrics'],
        })

    output = {
        'n_trees': len(records),
        'correlations': corr_results,
        'regression': regression,
        'trees': tree_records,
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(records)} trees to {output_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("STRUCTURAL METRICS vs TRAIN-TEST GAP")
    print(f"{'='*70}")
    print(f"N = {n}")
    print(f"\n{'Metric':<20} {'Pearson r':>10} {'p-val':>10} {'Spearman ρ':>10} {'p-val':>10}")
    print("-" * 60)
    for m in metric_names:
        c = corr_results[m]
        if c['pearson_r'] is not None:
            print(f"{m:<20} {c['pearson_r']:>10.4f} {c['pearson_p']:>10.4f} "
                  f"{c['spearman_r']:>10.4f} {c['spearman_p']:>10.4f}")
        else:
            print(f"{m:<20} {'N/A':>10} {'':>10} {'N/A':>10} {'':>10}")

    print(f"\n{'='*70}")
    print("LINEAR REGRESSION: gap ~ tree_size + depth + nesting_ratio + constant_ratio + if_count")
    print(f"{'='*70}")
    print(f"R² = {r2:.4f}  (N = {mask.sum()})")
    for i, col in enumerate(X_cols):
        print(f"  {col:<20} coef = {coef[i + 1]:>12.6f}")
    print(f"  {'intercept':<20} coef = {coef[0]:>12.6f}")

    print(f"\nSummary stats for gap:")
    print(f"  Mean gap: ${np.mean(y_clean):,.0f}  Std: ${np.std(y_clean):,.0f}")
    print(f"  Min gap: ${np.min(y_clean):,.0f}  Max gap: ${np.max(y_clean):,.0f}")

def main():
    out_dir = Path(__file__).resolve().parent
    records = load_all_trees()
    analyze(records, str(out_dir / 'structural_analysis.json'))


if __name__ == '__main__':
    main()
