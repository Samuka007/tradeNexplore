"""Genetic Programming (GP) for trading rule structure discovery.

Evolves tree-structured trading rules where:
- Terminals: price, sma(N), lma(N), ema(N, alpha), const
- Functions: +, -, *, >, <, AND, IF

The tree evaluates to a scalar at each time step; the sign determines
buy (+1), sell (-1), or hold (0) signals.
"""

from __future__ import annotations

import numpy as np

from trading_bot.algorithms.base import StructuralOptimizer, OptResult
from trading_bot.filters import wma, sma_filter, lma_filter, ema_filter

__all__ = ["GeneticProgramming", "GPNode"]


class GPNode:
    """Node in a GP program tree."""

    def __init__(self, value: str, is_terminal: bool = False):
        self.value = value
        self.is_terminal = is_terminal
        self.children: list[GPNode] = []
        self.params: dict = {}

    def copy(self) -> "GPNode":
        node = GPNode(self.value, self.is_terminal)
        node.params = self.params.copy()
        node.children = [c.copy() for c in self.children]
        return node

    def __repr__(self) -> str:
        if self.is_terminal:
            if self.value in ("sma", "lma"):
                return f"{self.value}({self.params.get('N', '?')})"
            elif self.value == "ema":
                return f"ema({self.params.get('N', '?')},{self.params.get('alpha', '?'):.2f})"
            elif self.value == "const":
                return f"{self.params.get('value', 0):.3f}"
            return self.value
        if self.children:
            return f"({self.value} {' '.join(repr(c) for c in self.children)})"
        return self.value


class GeneticProgramming(StructuralOptimizer):
    """Genetic Programming for discovering tree-structured trading rules.

    Args:
        population_size: Number of individuals per generation.
        generations: Number of evolutionary generations.
        crossover_rate: Probability of crossover (else reproduction).
        mutation_rate: Probability of mutation (else reproduction).
        max_depth: Maximum tree depth.
        tournament_size: Tournament selection pool size.
    """

    FUNCTIONS = ["+", "-", "*", ">", "<", "AND", "IF"]
    TERMINALS = ["price", "sma", "lma", "ema", "const"]

    ARITY = {
        "+": 2, "-": 2, "*": 2, ">": 2, "<": 2, "AND": 2, "IF": 3,
    }

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        crossover_rate: float = 0.9,
        mutation_rate: float = 0.1,
        max_depth: int = 5,
        tournament_size: int = 3,
    ):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size

    @property
    def name(self) -> str:
        return "GeneticProgramming"

    def _create_random_tree(self, depth: int = 0, max_depth: int | None = None) -> GPNode:
        """Grow a random program tree."""
        if max_depth is None:
            max_depth = self.max_depth
        if depth >= max_depth or (depth > 0 and np.random.rand() < 0.3):
            terminal = np.random.choice(self.TERMINALS)
            node = GPNode(terminal, is_terminal=True)
            if terminal in ("sma", "lma"):
                node.params["N"] = np.random.randint(5, 200)
            elif terminal == "ema":
                node.params["N"] = np.random.randint(5, 200)
                node.params["alpha"] = np.random.uniform(0.01, 0.99)
            elif terminal == "const":
                node.params["value"] = np.random.uniform(-1, 1)
            return node

        func = np.random.choice(self.FUNCTIONS)
        node = GPNode(func, is_terminal=False)
        for _ in range(self.ARITY[func]):
            node.children.append(self._create_random_tree(depth + 1, max_depth))
        return node

    def _collect_terminals(self, node: GPNode) -> set[tuple]:
        """Collect all unique terminal references in the tree."""
        if node.is_terminal:
            if node.value == "sma":
                return {("sma", node.params["N"])}
            elif node.value == "lma":
                return {("lma", node.params["N"])}
            elif node.value == "ema":
                return {("ema", node.params["N"], node.params["alpha"])}
            return set()
        refs = set()
        for c in node.children:
            refs |= self._collect_terminals(c)
        return refs

    def _build_indicator_cache(self, tree: GPNode, prices: np.ndarray) -> dict:
        """Pre-compute all indicator values the tree will need."""
        cache: dict = {"price": prices}
        refs = self._collect_terminals(tree)
        n = len(prices)
        for ref in refs:
            if ref[0] == "sma":
                N = min(ref[1], n)
                cache[ref] = wma(prices, N, sma_filter(N))
            elif ref[0] == "lma":
                N = min(ref[1], n)
                cache[ref] = wma(prices, N, lma_filter(N))
            elif ref[0] == "ema":
                _, N_raw, alpha = ref
                N = min(N_raw, n)
                cache[ref] = wma(prices, N, ema_filter(N, alpha))
        return cache

    def _evaluate_tree_fast(self, node: GPNode, cache: dict, idx: int) -> float:
        """Evaluate tree at a single time index using cached indicators."""
        if node.is_terminal:
            if node.value == "price":
                return float(cache["price"][idx])
            elif node.value in ("sma", "lma"):
                arr = cache[(node.value, node.params["N"])]
                return float(arr[idx]) if idx < len(arr) else float(cache["price"][idx])
            elif node.value == "ema":
                arr = cache[("ema", node.params["N"], node.params["alpha"])]
                return float(arr[idx]) if idx < len(arr) else float(cache["price"][idx])
            elif node.value == "const":
                return float(node.params["value"])
            return 0.0

        vals = [self._evaluate_tree_fast(c, cache, idx) for c in node.children]

        if node.value == "+":
            return vals[0] + vals[1]
        elif node.value == "-":
            return vals[0] - vals[1]
        elif node.value == "*":
            return vals[0] * vals[1]
        elif node.value == ">":
            return 1.0 if vals[0] > vals[1] else 0.0
        elif node.value == "<":
            return 1.0 if vals[0] < vals[1] else 0.0
        elif node.value == "AND":
            return 1.0 if (vals[0] and vals[1]) else 0.0
        elif node.value == "IF":
            return vals[1] if vals[0] else vals[2]
        return 0.0

    def evaluate(self, tree: GPNode, prices: np.ndarray) -> np.ndarray:
        """Evaluate tree across all prices to generate signals.

        Returns array of +1 (buy), -1 (sell), 0 (hold).
        """
        n = len(prices)
        cache = self._build_indicator_cache(tree, prices)
        raw = np.zeros(n)
        for i in range(n):
            raw[i] = self._evaluate_tree_fast(tree, cache, i)

        signals = np.zeros(n, dtype=int)
        prev_sign = 0
        for i in range(n):
            curr_sign = 1 if raw[i] > 0 else (-1 if raw[i] < 0 else 0)
            if curr_sign != 0 and curr_sign != prev_sign:
                signals[i] = curr_sign
                prev_sign = curr_sign
        return signals

    def _get_all_nodes(self, node: GPNode) -> list[GPNode]:
        """Collect all nodes in the tree."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes

    def _crossover(self, parent1: GPNode, parent2: GPNode) -> GPNode:
        """Subtree crossover."""
        child = parent1.copy()
        nodes1 = self._get_all_nodes(child)
        nodes2 = self._get_all_nodes(parent2)
        if not nodes1 or not nodes2:
            return child
        node1 = np.random.choice(nodes1)
        node2 = np.random.choice(nodes2)
        node1.value = node2.value
        node1.is_terminal = node2.is_terminal
        node1.params = node2.params.copy()
        node1.children = [c.copy() for c in node2.children]
        return child

    def _mutate(self, individual: GPNode) -> GPNode:
        """Subtree mutation."""
        mutant = individual.copy()
        nodes = self._get_all_nodes(mutant)
        if not nodes:
            return mutant
        target = np.random.choice(nodes)
        new_subtree = self._create_random_tree(max_depth=self.max_depth // 2)
        target.value = new_subtree.value
        target.is_terminal = new_subtree.is_terminal
        target.params = new_subtree.params.copy()
        target.children = [c.copy() for c in new_subtree.children]
        return mutant

    def _tournament_select(
        self, population: list[GPNode], fitness: list[float]
    ) -> GPNode:
        """Tournament selection."""
        idxs = np.random.choice(len(population), size=self.tournament_size, replace=False)
        best_idx = idxs[np.argmax([fitness[i] for i in idxs])]
        return population[best_idx]

    def optimize(self, fitness_fn, n_generations: int = 50) -> OptResult:
        """Run GP optimization loop.

        Args:
            fitness_fn: Function mapping GPNode -> float fitness.
            n_generations: Number of generations to evolve.

        Returns:
            OptResult with best tree and convergence history.
        """
        population = [self._create_random_tree() for _ in range(self.pop_size)]
        fitness = [fitness_fn(ind) for ind in population]

        history = []

        for _ in range(n_generations):
            new_population = []
            best_idx = int(np.argmax(fitness))
            new_population.append(population[best_idx].copy())

            while len(new_population) < self.pop_size:
                r = np.random.rand()
                if r < self.cx_rate:
                    p1 = self._tournament_select(population, fitness)
                    p2 = self._tournament_select(population, fitness)
                    child = self._crossover(p1, p2)
                elif r < self.cx_rate + self.mut_rate:
                    p = self._tournament_select(population, fitness)
                    child = self._mutate(p)
                else:
                    p = self._tournament_select(population, fitness)
                    child = p.copy()
                new_population.append(child)

            population = new_population
            fitness = [fitness_fn(ind) for ind in population]
            history.append(float(np.max(fitness)))

        best_idx = int(np.argmax(fitness))
        return OptResult(
            best=population[best_idx],
            best_fitness=float(fitness[best_idx]),
            history=history,
            metadata={
                "algorithm_name": self.name,
                "generations": n_generations,
                "population_size": self.pop_size,
            },
        )
