"""Algorithm interfaces and base classes.

Defines the contract that all optimization algorithms must follow.
Core algorithm implementations (PSO, ABC, HS, GP) are provided in
separate modules and must implement these interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class OptResult:
    """Standard optimization result container."""

    best: object
    best_fitness: float
    history: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def name(self) -> Optional[str]:
        return self.metadata.get("algorithm_name")


class ContinuousOptimizer(ABC):
    """Base class for continuous parameter optimizers (PSO, ABC, HS).

    Optimizes a parameter vector within specified bounds to maximize
    a fitness function.
    """

    @abstractmethod
    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
    ) -> OptResult:
        """Run optimization.

        Args:
            fitness_fn: Function mapping parameter vector to fitness score.
                        Should accept np.ndarray and return float.
            bounds: List of (min, max) tuples for each dimension.

        Returns:
            OptResult with best solution found and convergence history.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        ...


class StructuralOptimizer(ABC):
    """Base class for structural optimizers (GP).

    Searches tree-structured solution spaces rather than continuous
    parameter vectors.
    """

    @abstractmethod
    def optimize(
        self,
        fitness_fn: Callable[[object], float],
        n_generations: int,
    ) -> OptResult:
        """Run structural optimization.

        Args:
            fitness_fn: Function mapping tree structure to fitness score.
            n_generations: Number of generations/evolutionary steps.

        Returns:
            OptResult with best tree found and convergence history.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        ...


class StubPSO(ContinuousOptimizer):
    """Placeholder PSO implementation — returns random parameters.

    Used for framework testing before real algorithm is implemented.
    """

    def __init__(self, n_particles: int = 30, max_iter: int = 10):
        self.n_particles = n_particles
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "StubPSO"

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
    ) -> OptResult:
        best = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], size=len(bounds)
        )
        best_fitness = fitness_fn(best)
        return OptResult(
            best=best,
            best_fitness=best_fitness,
            history=[best_fitness],
            metadata={"algorithm_name": self.name, "iterations": self.max_iter},
        )


class StubABC(ContinuousOptimizer):
    """Placeholder ABC implementation — returns random parameters."""

    def __init__(self, n_bees: int = 50, max_cycles: int = 10):
        self.n_bees = n_bees
        self.max_cycles = max_cycles

    @property
    def name(self) -> str:
        return "StubABC"

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
    ) -> OptResult:
        best = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], size=len(bounds)
        )
        best_fitness = fitness_fn(best)
        return OptResult(
            best=best,
            best_fitness=best_fitness,
            history=[best_fitness],
            metadata={"algorithm_name": self.name, "cycles": self.max_cycles},
        )


class StubHarmonySearch(ContinuousOptimizer):
    """Placeholder Harmony Search implementation — returns random parameters."""

    def __init__(self, hms: int = 30, max_iter: int = 10):
        self.hms = hms
        self.max_iter = max_iter

    @property
    def name(self) -> str:
        return "StubHarmonySearch"

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
    ) -> OptResult:
        best = np.random.uniform(
            [b[0] for b in bounds], [b[1] for b in bounds], size=len(bounds)
        )
        best_fitness = fitness_fn(best)
        return OptResult(
            best=best,
            best_fitness=best_fitness,
            history=[best_fitness],
            metadata={"algorithm_name": self.name, "iterations": self.max_iter},
        )


class StubGP(StructuralOptimizer):
    """Placeholder GP implementation — returns a simple fixed tree.

    Used for framework testing before real GP is implemented.
    """

    def __init__(self, population_size: int = 100, generations: int = 10):
        self.pop_size = population_size
        self.generations = generations

    @property
    def name(self) -> str:
        return "StubGP"

    def optimize(
        self,
        fitness_fn: Callable[[object], float],
        n_generations: int = 10,
    ) -> OptResult:
        tree = {"type": "sma_crossover", "fast": 10, "slow": 30}
        best_fitness = fitness_fn(tree)
        return OptResult(
            best=tree,
            best_fitness=best_fitness,
            history=[best_fitness],
            metadata={"algorithm_name": self.name, "generations": n_generations},
        )
