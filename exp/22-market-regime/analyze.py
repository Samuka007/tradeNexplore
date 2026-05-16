"""Analyze GP tree structure by market regime.

Reads per-regime result JSONs, categorizes winning trees (>BH) by:
  - Trend-following features: SMA, LMA, EMA
  - Mean-reversion features: RSI, volatility
  - Conditional logic: IF, AND
  - Tree size and depth

Outputs: structure_by_regime.json
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import re
from collections import Counter


REGIMES = ["bull", "bear", "sideways"]


def load_regime_results(regime: str) -> dict:
    with open(f"{regime}_results.json") as f:
        return json.load(f)


def count_features(tree_repr: str) -> dict:
    """Analyze tree string for structural features.

    Returns dict with counts of key feature groups.
    """
    features = {
        "sma": len(re.findall(r"\bsma\b", tree_repr)),
        "lma": len(re.findall(r"\blma\b", tree_repr)),
        "ema": len(re.findall(r"\bema\b", tree_repr)),
        "rsi": len(re.findall(r"\brsi\b", tree_repr)),
        "momentum": len(re.findall(r"\bmomentum\b", tree_repr)),
        "volatility": len(re.findall(r"\bvolatility\b", tree_repr)),
        "IF": len(re.findall(r"\bIF\b", tree_repr)),
        "AND": len(re.findall(r"\bAND\b", tree_repr)),
        "price": len(re.findall(r"\bprice\b", tree_repr)),
        "const": len(re.findall(r"\bconst\b", tree_repr)),
    }
    return features


def tree_depth_from_repr(tree_repr: str) -> int:
    """Estimate tree depth from nested parentheses."""
    max_depth = 0
    depth = 0
    for ch in tree_repr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def analyze_regime(regime: str, results: dict) -> dict:
    """Analyze all winning trees for a regime."""
    bh = results["bh_test"]
    seeds = results["seeds"]

    winning = [s for s in seeds if s["test_cash"] > bh]
    all_seeds = seeds

    def aggregate(seed_list):
        if not seed_list:
            return {
                "count": 0,
                "mean_size": 0.0,
                "mean_depth": 0.0,
                "feature_counts": {},
                "feature_trees": {},
            }

        sizes = [s["tree_size"] for s in seed_list]
        depths = [tree_depth_from_repr(s["tree_repr"]) for s in seed_list]
        all_features = Counter()
        trees_with_feature = Counter()
        for s in seed_list:
            feats = count_features(s["tree_repr"])
            all_features.update(feats)
            for k, v in feats.items():
                if v > 0:
                    trees_with_feature[k] += 1

        return {
            "count": len(seed_list),
            "mean_size": sum(sizes) / len(sizes) if sizes else 0.0,
            "mean_depth": sum(depths) / len(depths) if depths else 0.0,
            "feature_counts": dict(all_features),
            "feature_trees": dict(trees_with_feature),
        }

    win_agg = aggregate(winning)
    all_agg = aggregate(all_seeds)

    # Derived metrics
    trend_count = (win_agg["feature_counts"].get("sma", 0)
                   + win_agg["feature_counts"].get("lma", 0)
                   + win_agg["feature_counts"].get("ema", 0))
    mean_rev_count = (win_agg["feature_counts"].get("rsi", 0)
                      + win_agg["feature_counts"].get("volatility", 0))
    conditional_count = (win_agg["feature_counts"].get("IF", 0)
                         + win_agg["feature_counts"].get("AND", 0))

    return {
        "regime": regime,
        "n_seeds": len(all_seeds),
        "n_winning": win_agg["count"],
        "winning_pct": win_agg["count"] / len(all_seeds) * 100 if all_seeds else 0,
        "winning_trees": {
            "mean_size": win_agg["mean_size"],
            "mean_depth": win_agg["mean_depth"],
            "trend_following_nodes": trend_count,
            "mean_reversion_nodes": mean_rev_count,
            "conditional_nodes": conditional_count,
            "feature_counts": win_agg["feature_counts"],
            "trees_with_feature": win_agg["feature_trees"],
        },
        "all_trees": {
            "mean_size": all_agg["mean_size"],
            "mean_depth": all_agg["mean_depth"],
            "feature_counts": all_agg["feature_counts"],
            "trees_with_feature": all_agg["feature_trees"],
        },
    }


def compare_regimes(analyses: list[dict]) -> dict:
    """Cross-regime comparison."""
    comparison = {}
    metrics = ["mean_size", "mean_depth", "trend_following_nodes",
               "mean_reversion_nodes", "conditional_nodes"]

    for metric in metrics:
        values = {}
        for a in analyses:
            values[a["regime"]] = a["winning_trees"].get(metric, 0)
        comparison[metric] = values

    # Interpret
    interpretations = []
    sizes = {a["regime"]: a["winning_trees"]["mean_size"] for a in analyses}
    depths = {a["regime"]: a["winning_trees"]["mean_depth"] for a in analyses}
    conditionals = {a["regime"]: a["winning_trees"]["conditional_nodes"] for a in analyses}
    trends = {a["regime"]: a["winning_trees"]["trend_following_nodes"] for a in analyses}
    meanrevs = {a["regime"]: a["winning_trees"]["mean_reversion_nodes"] for a in analyses}

    # Which regime has simplest trees?
    simplest = min(sizes, key=sizes.get)
    interpretations.append(
        f"{simplest}-market winning trees are simplest "
        f"(mean size={sizes[simplest]:.1f} vs "
        f"{', '.join(f'{r}={sizes[r]:.1f}' for r in sizes if r != simplest)})"
    )

    # Which regime uses most conditional logic?
    most_conditional = max(conditionals, key=conditionals.get)
    interpretations.append(
        f"{most_conditional}-market trees use most conditional logic "
        f"({conditionals[most_conditional]:.1f} IF/AND nodes)"
    )

    # Which regime is most trend-following?
    most_trend = max(trends, key=trends.get)
    interpretations.append(
        f"{most_trend}-market trees use most trend-following features "
        f"({trends[most_trend]:.1f} SMA/LMA/EMA nodes)"
    )

    # Which regime is most mean-reverting?
    most_mr = max(meanrevs, key=meanrevs.get)
    interpretations.append(
        f"{most_mr}-market trees use most mean-reversion features "
        f"({meanrevs[most_mr]:.1f} RSI/volatility nodes)"
    )

    comparison["interpretations"] = interpretations
    return comparison


def main():
    regime_analyses = []
    for regime in REGIMES:
        try:
            results = load_regime_results(regime)
        except FileNotFoundError:
            print(f"Warning: {regime}_results.json not found, skipping.")
            continue

        analysis = analyze_regime(regime, results)
        regime_analyses.append(analysis)

        print(f"\n=== {regime.upper()} ===")
        w = analysis["winning_trees"]
        print(f"  Winning trees: {analysis['n_winning']}/{analysis['n_seeds']} "
              f"({analysis['winning_pct']:.0f}%)")
        print(f"  Mean size: {w['mean_size']:.1f}, Mean depth: {w['mean_depth']:.1f}")
        print(f"  Trend-following nodes (SMA/LMA/EMA): {w['trend_following_nodes']}")
        print(f"  Mean-reversion nodes (RSI/volatility): {w['mean_reversion_nodes']}")
        print(f"  Conditional nodes (IF/AND): {w['conditional_nodes']}")
        print(f"  Feature counts: {w['feature_counts']}")
        print(f"  Trees with feature: {w['trees_with_feature']}")

    comparison = compare_regimes(regime_analyses)

    output = {
        "regimes": {a["regime"]: a for a in regime_analyses},
        "comparison": comparison,
    }

    with open("structure_by_regime.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved structure_by_regime.json")

    print("\n=== CROSS-REGIME COMPARISON ===")
    for interp in comparison["interpretations"]:
        print(f"  → {interp}")


if __name__ == "__main__":
    main()
