"""PSO: inertia weight strategy comparison.

Intuition: The inertia weight w controls the balance between global
exploration (large w) and local exploitation (small w). Common variants:
- linear: w decays from 0.9 to 0.4 (our baseline)
- fixed: w = 0.7 (constant)
- Clerc: constriction coefficient for guaranteed convergence
- random: w uniform in [0.5, 1.0] each iteration

We implement each by subclassing PSO and overriding the velocity update.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import yfinance as yf
import json
from tiny_bot.backtest import backtest, buy_and_hold
from tiny_bot.strategy import VectorStrategy
from tiny_bot.pso import PSO

SEED = 42
np.random.seed(SEED)


class PSO_Linear(PSO):
    """Baseline: linear decay 0.9 → 0.4."""
    pass  # already the default in PSO


class PSO_Fixed(PSO):
    """Fixed inertia w = 0.7."""
    def optimize(self, fitness_fn, bounds, n_workers=1):
        # Copy-paste PSO.optimize but replace w calculation
        dim = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        particles = np.random.uniform(lo, hi, size=(self.n, dim))
        velocities = np.zeros((self.n, dim))
        pbest = particles.copy()

        def _eval_all(pop):
            if n_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return np.array(list(ex.map(fitness_fn, pop)))
            return np.array([fitness_fn(p) for p in pop])

        pfit = _eval_all(particles)
        gbest = pbest[int(np.argmax(pfit))].copy()
        gfit = float(pfit.max())
        history = [gfit]

        for it in range(self.max_iter):
            w = 0.7  # FIXED
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


class PSO_Clerc(PSO):
    """Clerc constriction: φ = c1+c2 = 4.1, χ = 2/(φ-2+√(φ²-4φ))."""
    def optimize(self, fitness_fn, bounds, n_workers=1):
        phi = 4.1
        chi = 2.0 / abs(phi - 2.0 + np.sqrt(phi**2 - 4*phi))
        dim = len(bounds)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        particles = np.random.uniform(lo, hi, size=(self.n, dim))
        velocities = np.zeros((self.n, dim))
        pbest = particles.copy()

        def _eval_all(pop):
            if n_workers > 1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return np.array(list(ex.map(fitness_fn, pop)))
            return np.array([fitness_fn(p) for p in pop])

        pfit = _eval_all(particles)
        gbest = pbest[int(np.argmax(pfit))].copy()
        gfit = float(pfit.max())
        history = [gfit]

        for it in range(self.max_iter):
            for i in range(self.n):
                r1, r2 = np.random.rand(2)
                velocities[i] = chi * (
                    velocities[i]
                    + 2.05 * r1 * (pbest[i] - particles[i])
                    + 2.05 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lo, hi)
            new_fit = _eval_all(particles)
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


def load_data():
    df = yf.download('BTC-USD', start='2014-01-01', end='2022-12-31', progress=False, auto_adjust=True)
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    train = df[df['Date'] < '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    test = df[df['Date'] >= '2020-01-01']['Close'].to_numpy(dtype=np.float64).flatten()
    return train, test


def run(cls, label, train, test):
    pso = cls(n_particles=30, max_iter=50, seed=SEED)
    bounds = [(2, 200), (2, 200), (0.1, 100)]
    pso_res = pso.optimize(
        lambda p: backtest(train, VectorStrategy(p, 'position_sma').signals(train))['final_cash'],
        bounds
    )
    sig = VectorStrategy(pso_res['best'], 'position_sma').signals(test)
    test_cash = backtest(test, sig)['final_cash']
    return {
        'label': label,
        'train_cash': float(backtest(train, VectorStrategy(pso_res['best'], 'position_sma').signals(train))['final_cash']),
        'test_cash': float(test_cash),
        'best_params': [float(x) for x in pso_res['best']],
    }


def main():
    train, test = load_data()
    bh = buy_and_hold(test)
    print(f"Buy-and-Hold test: ${bh:,.0f}")
    print("=" * 70)

    variants = [
        (PSO_Linear, 'linear 0.9→0.4 (baseline)'),
        (PSO_Fixed, 'fixed 0.7'),
        (PSO_Clerc, 'Clerc constriction'),
    ]

    results = []
    for cls, label in variants:
        print(f"\nRunning {label}...")
        r = run(cls, label, train, test)
        print(f"  train=${r['train_cash']:>10,.0f}  test=${r['test_cash']:>10,.0f}")
        results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Strategy':<30} {'Train':>10} {'Test':>10} {'vs BH?':>8}")
    print("-" * 70)
    for r in results:
        wins = 'YES' if r['test_cash'] > bh else 'NO'
        print(f"{r['label']:<30} ${r['train_cash']:>8,.0f} ${r['test_cash']:>8,.0f} {wins:>8}")

    out = {'buy_and_hold_test': float(bh), 'experiments': results}
    with open('results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to results.json")


if __name__ == '__main__':
    main()
