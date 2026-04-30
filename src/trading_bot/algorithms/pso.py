"""Particle Swarm Optimization (PSO) for continuous parameter tuning.

Implements the standard PSO velocity/position update with:
- Inertia weight (adaptive decay)
- Cognitive coefficient (pbest attraction)
- Social coefficient (gbest attraction)
- Boundary clipping
"""

from __future__ import annotations

import numpy as np

from trading_bot.algorithms.base import ContinuousOptimizer, OptResult

__all__ = ["PSO"]


class PSO(ContinuousOptimizer):
    """Particle Swarm Optimization for continuous parameter tuning.

    Maximizes a fitness function within specified bounds.

    Args:
        n_particles: Number of particles in the swarm.
        w: Inertia weight (default 0.729 from Clerc & Kennedy 2002).
        c1: Cognitive coefficient (attraction to personal best).
        c2: Social coefficient (attraction to global best).
        max_iter: Maximum number of iterations.
        adaptive_inertia: If True, linearly decay w from 0.9 to 0.4.
    """

    def __init__(
        self,
        n_particles: int = 30,
        w: float = 0.729,
        c1: float = 2.05,
        c2: float = 2.05,
        max_iter: int = 100,
        adaptive_inertia: bool = True,
    ):
        self.n_particles = n_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter
        self.adaptive_inertia = adaptive_inertia

    @property
    def name(self) -> str:
        return "PSO"

    def optimize(
        self,
        fitness_fn,
        bounds: list[tuple[float, float]],
    ) -> OptResult:
        """Run PSO optimization.

        Args:
            fitness_fn: Function mapping np.ndarray params -> float fitness.
            bounds: List of (min, max) tuples per dimension.

        Returns:
            OptResult with best solution and convergence history.
        """
        dim = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])

        particles = np.random.uniform(
            lower, upper, size=(self.n_particles, dim)
        )
        velocities = np.zeros((self.n_particles, dim))

        pbest = particles.copy()
        pbest_fitness = np.empty(self.n_particles)
        for i in range(self.n_particles):
            pbest_fitness[i] = fitness_fn(particles[i])

        gbest_idx = int(np.argmax(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]

        history = [float(gbest_fitness)]

        for iteration in range(self.max_iter):
            w = self.w
            if self.adaptive_inertia:
                w = 0.9 - (0.5 * iteration / self.max_iter)

            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    w * velocities[i]
                    + self.c1 * r1 * (pbest[i] - particles[i])
                    + self.c2 * r2 * (gbest - particles[i])
                )
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lower, upper)

                fitness = fitness_fn(particles[i])

                if fitness > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = fitness
                    if fitness > gbest_fitness:
                        gbest = particles[i].copy()
                        gbest_fitness = fitness

            history.append(float(gbest_fitness))

        return OptResult(
            best=gbest,
            best_fitness=float(gbest_fitness),
            history=history,
            metadata={
                "algorithm_name": self.name,
                "n_particles": self.n_particles,
                "iterations": self.max_iter,
            },
        )
