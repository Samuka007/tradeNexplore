"""PSO parent-offspring baseline for the epistasis experiment.

Analog to GP epistasis: logs previous best position fitness (parent)
versus new position fitness (offspring) at each iteration.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import pearsonr

from tiny_bot.data import load_btc_data
from tiny_bot.strategy import VectorStrategy
from tiny_bot.backtest import backtest

# ── config ───────────────────────────────────────────────────────────────────

SEEDS = [0, 7, 13, 21, 42, 55, 77, 88, 99, 123]
N_PARTICLES = 30
MAX_ITER = 50
N_WORKERS = 4

# dual_crossover has 14 parameters
# [w1,w2,w3,d1,d2,d3,a3, w4,w5,w6,d4,d5,d6,a6]
BOUNDS: list[tuple[float, float]] = [
    (0.0, 1.0),  # w1
    (0.0, 1.0),  # w2
    (0.0, 1.0),  # w3
    (2.0, 200.0),  # d1
    (2.0, 200.0),  # d2
    (2.0, 200.0),  # d3
    (0.01, 0.99),  # a3
    (0.0, 1.0),  # w4
    (0.0, 1.0),  # w5
    (0.0, 1.0),  # w6
    (2.0, 200.0),  # d4
    (2.0, 200.0),  # d5
    (2.0, 200.0),  # d6
    (0.01, 0.99),  # a6
]

OUT_DIR = Path(__file__).resolve().parent


class InstrumentedPSO:
    """PSO that logs parent (pbest) → offspring (new position) fitness pairs."""

    def __init__(self, n_particles: int = 30, max_iter: int = 100,
                 seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.n = n_particles
        self.max_iter = max_iter
        self.parent_offspring_log: list[dict] = []

    def optimize(self, fitness_fn, bounds: list[tuple[float, float]],
                 n_workers: int = 1) -> dict:
        dim = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        particles = np.random.uniform(lo, hi, size=(self.n, dim))
        velocities = np.zeros((self.n, dim))
        pbest = particles.copy()

        def _eval_all(pop):
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return np.array(list(ex.map(fitness_fn, pop)))
            return np.array([fitness_fn(p) for p in pop])

        pfit = _eval_all(particles)
        gbest = pbest[int(np.argmax(pfit))].copy()
        gfit = float(pfit.max())
        history = [gfit]
        self.parent_offspring_log = []

        for it in range(self.max_iter):
            w = 0.9 - 0.5 * it / self.max_iter
            for i in range(self.n):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    w * velocities[i]
                    + 2.05 * r1 * (pbest[i] - particles[i])
                    + 2.05 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lo, hi)

            new_fit = _eval_all(particles)

            # Log parent (pbest fitness before update) vs offspring (new position fitness)
            for i in range(self.n):
                parent_fit = float(pfit[i])
                offspring_fit = float(new_fit[i])
                self.parent_offspring_log.append({
                    "iteration": it,
                    "particle": i,
                    "parent_fit": parent_fit,
                    "offspring_fit": offspring_fit,
                })

            # Update pbest and gbest
            for i in range(self.n):
                f = new_fit[i]
                if f > pfit[i]:
                    pbest[i] = particles[i].copy()
                    pfit[i] = f
                    if f > gfit:
                        gbest = particles[i].copy()
                        gfit = f

            history.append(gfit)

        return {"best": gbest, "fitness": gfit, "history": history}


# ── fitness ──────────────────────────────────────────────────────────────────

def make_fitness_fn(prices: np.ndarray):
    """Return fitness function for a 14D dual_crossover strategy vector."""
    def fitness(vec: np.ndarray) -> float:
        strat = VectorStrategy(vec, stype="dual_crossover")
        sig = strat.signals(prices)
        # Convert discrete {-1,0,1} to continuous [0,1] for backtest
        continuous_sig = np.where(sig == 1, 1.0, np.where(sig == -1, 0.0, 0.5))
        result = backtest(prices, continuous_sig)
        return result["final_cash"]
    return fitness


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    train, test, _ = load_btc_data()
    print(f"Train prices: {len(train)}, Test prices: {len(test)}")

    fitness_fn = make_fitness_fn(train)

    # We run all seeds independently, collecting all parent-offspring pairs
    all_pairs: list[dict] = []
    all_results = {}

    for seed in SEEDS:
        print(f"\nSeed {seed} ({N_PARTICLES} × {MAX_ITER})...")
        pso = InstrumentedPSO(
            n_particles=N_PARTICLES,
            max_iter=MAX_ITER,
            seed=seed,
        )
        result = pso.optimize(fitness_fn, BOUNDS, n_workers=N_WORKERS)

        best_vec = result["best"]
        best_fit = result["fitness"]

        # Evaluate on test
        strat = VectorStrategy(best_vec, stype="dual_crossover")
        test_sig = strat.signals(test)
        test_cont = np.where(test_sig == 1, 1.0, np.where(test_sig == -1, 0.0, 0.5))
        test_res = backtest(test, test_cont)

        all_results[str(seed)] = {
            "best_fitness": float(best_fit),
            "history": [float(h) for h in result["history"]],
            "test_final_cash": test_res["final_cash"],
            "test_sharpe": test_res["sharpe_ratio"],
            "n_pairs": len(pso.parent_offspring_log),
        }
        all_pairs.extend(pso.parent_offspring_log)

        print(f"  Best fitness: {best_fit:.2f}, Test cash: {test_res['final_cash']:.2f}, "
              f"Pairs: {len(pso.parent_offspring_log)}")

    # Save pairs
    pairs_path = OUT_DIR / "pso_pairs.json"
    with open(pairs_path, "w") as f:
        json.dump(all_pairs, f, indent=2)

    # Compute parent-offspring Pearson r
    parents = [p["parent_fit"] for p in all_pairs]
    offsprings = [p["offspring_fit"] for p in all_pairs]

    if len(parents) >= 3 and np.std(parents) > 1e-10 and np.std(offsprings) > 1e-10:
        overall_r, overall_p = pearsonr(parents, offsprings)
    else:
        overall_r, overall_p = 0.0, 1.0

    print(f"\nOverall PSO parent-offspring Pearson r: {overall_r:.4f} (p={overall_p:.4e})")

    # Also compute per-iteration
    iters = sorted(set(p["iteration"] for p in all_pairs))
    iter_rs = []
    for it in iters:
        it_pairs = [p for p in all_pairs if p["iteration"] == it]
        x = [p["parent_fit"] for p in it_pairs]
        y = [p["offspring_fit"] for p in it_pairs]
        if len(x) >= 3 and np.std(x) > 1e-10 and np.std(y) > 1e-10:
            rv, _ = pearsonr(x, y)
            iter_rs.append(float(rv))
        else:
            iter_rs.append(0.0)

    analysis = {
        "overall": {
            "pearson_r": float(overall_r),
            "p_value": float(overall_p),
            "n_pairs": len(all_pairs),
        },
        "per_iteration": {
            "iterations": iters,
            "pearson_r": iter_rs,
        },
        "seeds": all_results,
        "aggregate": {
            "n_seeds": len(SEEDS),
            "n_particles": N_PARTICLES,
            "max_iter": MAX_ITER,
            "mean_best_fitness": float(np.mean([r["best_fitness"] for r in all_results.values()])),
            "std_best_fitness": float(np.std([r["best_fitness"] for r in all_results.values()])),
            "mean_test_cash": float(np.mean([r["test_final_cash"] for r in all_results.values()])),
            "std_test_cash": float(np.std([r["test_final_cash"] for r in all_results.values()])),
        },
    }

    analysis_path = OUT_DIR / "pso_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)

    print(f"Saved pairs to {pairs_path}")
    print(f"Saved analysis to {analysis_path}")


if __name__ == "__main__":
    main()
