"""Post-hoc analysis for the GP epistasis experiment.

Computes Pearson r between parent_mean_raw_fit and offspring_raw_fit:
- Per generation (across seeds)
- Overall (all pairs pooled)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from scipy.stats import pearsonr

OUT_DIR = Path(__file__).resolve().parent


def load_all_pairs() -> dict[str, list[dict]]:
    """Load all seed_XX_pairs.json files.

    Returns: {seed_str: [record, ...]}
    """
    pairs = {}
    for p in sorted(OUT_DIR.glob("seed_*_pairs.json")):
        seed_str = p.stem.replace("seed_", "").replace("_pairs", "")
        with open(p) as f:
            data = json.load(f)
        pairs[seed_str] = data
    return pairs


def compute_per_generation(all_pairs: dict[str, list[dict]]) -> dict:
    """Compute Pearson r per generation, aggregating across seeds.

    Returns dict with keys: generations (list of gen indices),
      r_mean, r_std, r_values, p_values (aligned lists).
    """
    # Group by generation
    gen_to_parent = {}
    gen_to_offspring = {}

    for seed_str, records in all_pairs.items():
        for rec in records:
            g = rec["generation"]
            if g not in gen_to_parent:
                gen_to_parent[g] = []
                gen_to_offspring[g] = []
            gen_to_parent[g].append(rec["parent_mean_raw_fit"])
            gen_to_offspring[g].append(rec["offspring_raw_fit"])

    generations = sorted(gen_to_parent.keys())
    r_mean = []
    r_std = []
    r_values = []
    p_values = []

    # Compute per-seed r for each generation, then aggregate
    for g in generations:
        # Group by seed for this generation
        seed_rs = []
        for seed_str, records in all_pairs.items():
            gen_recs = [r for r in records if r["generation"] == g]
            if len(gen_recs) < 3:
                continue  # too few points for correlation
            x = [r["parent_mean_raw_fit"] for r in gen_recs]
            y = [r["offspring_raw_fit"] for r in gen_recs]
            if np.std(x) < 1e-10 or np.std(y) < 1e-10:
                continue
            r_val, p_val = pearsonr(x, y)
            seed_rs.append(r_val)

        if seed_rs:
            r_mean.append(float(np.mean(seed_rs)))
            r_std.append(float(np.std(seed_rs)))
        else:
            r_mean.append(0.0)
            r_std.append(0.0)

        # Pooled across all seeds for this generation
        all_x = gen_to_parent[g]
        all_y = gen_to_offspring[g]
        if len(all_x) >= 3 and np.std(all_x) > 1e-10 and np.std(all_y) > 1e-10:
            pr, pp = pearsonr(all_x, all_y)
            r_values.append(float(pr))
            p_values.append(float(pp))
        else:
            r_values.append(0.0)
            p_values.append(1.0)

    return {
        "generations": generations,
        "r_mean_per_seed": r_mean,
        "r_std_per_seed": r_std,
        "r_pooled": r_values,
        "p_pooled": p_values,
    }


def main():
    # Load results.json for aggregate stats
    results_path = OUT_DIR / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    else:
        print("WARNING: results.json not found; skipping aggregate stats")
        results = {"aggregate": {}, "seeds": {}}

    # Load pair data
    all_pairs = load_all_pairs()
    if not all_pairs:
        print("ERROR: no seed_*_pairs.json files found")
        sys.exit(1)

    print(f"Loaded pairs from {len(all_pairs)} seeds")
    total_pairs = sum(len(v) for v in all_pairs.values())
    print(f"Total pairs: {total_pairs}")

    # Per-generation analysis
    per_gen = compute_per_generation(all_pairs)

    # Overall correlation (all pairs pooled)
    all_parent = []
    all_offspring = []
    for records in all_pairs.values():
        for rec in records:
            all_parent.append(rec["parent_mean_raw_fit"])
            all_offspring.append(rec["offspring_raw_fit"])

    overall_r, overall_p = pearsonr(all_parent, all_offspring) if len(all_parent) >= 3 else (0.0, 1.0)

    print(f"\nOverall Pearson r: {overall_r:.4f} (p={overall_p:.4e})")
    print(f"Per-generation r (pooled):")
    for i, g in enumerate(per_gen["generations"]):
        print(f"  Gen {g:2d}: r={per_gen['r_pooled'][i]:+.4f} "
              f"(mean per-seed: {per_gen['r_mean_per_seed'][i]:+.4f} ± {per_gen['r_std_per_seed'][i]:.4f})")

    # Save analysis
    analysis = {
        "overall": {
            "pearson_r": float(overall_r),
            "p_value": float(overall_p),
            "n_pairs": len(all_parent),
        },
        "per_generation": per_gen,
        "aggregate_stats": results.get("aggregate", {}),
    }

    analysis_path = OUT_DIR / "analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"\nSaved to {analysis_path}")


if __name__ == "__main__":
    main()
