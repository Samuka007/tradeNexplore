"""Particle Swarm Optimization."""

import numpy as np
from concurrent.futures import ThreadPoolExecutor


class PSO:
    """Maximise a fitness function within parameter bounds."""

    def __init__(self, n_particles: int = 30, max_iter: int = 100, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)
        self.n = n_particles
        self.max_iter = max_iter

    def optimize(self, fitness_fn, bounds: list[tuple[float, float]], n_workers: int = 1):
        """Return dict with keys: best, fitness, history.

        Args:
            fitness_fn: callable mapping np.ndarray -> float.
            bounds: list of (min, max) per dimension.
            n_workers: number of threads for parallel fitness evaluation.
                       numpy releases the GIL, so ThreadPoolExecutor helps.
        """
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