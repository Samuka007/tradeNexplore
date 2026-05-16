"""gp_early_stop.py — Instrumented GP with structural early stopping.

Self-contained copy of tiny_bot/gp.py with:
  - Structural metric computation for GP trees.
  - Early stopping in optimize() based on structural thresholds.
  - Control mode (early_stop=False) for fair comparison.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tiny_bot.filters import wma, sma_filter, lma_filter, ema_filter


def _rsi(prices: np.ndarray, N: int) -> np.ndarray:
    """Causal RSI: first N-1 values are NaN."""
    diff = np.diff(prices)
    gains = np.where(diff > 0, diff, 0)
    losses = np.where(diff < 0, -diff, 0)
    avg_gain = np.convolve(gains, np.ones(N) / N, mode='valid')
    avg_loss = np.convolve(losses, np.ones(N) / N, mode='valid')
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    full = np.full(len(prices), np.nan)
    full[N:] = rsi
    return full


def _momentum(prices: np.ndarray, N: int) -> np.ndarray:
    """Causal momentum: P[t] - P[t-N].  First N values are NaN."""
    full = np.full(len(prices), np.nan)
    full[N:] = prices[N:] - prices[:-N]
    return full


def _volatility(prices: np.ndarray, N: int) -> np.ndarray:
    """Causal rolling standard deviation.  First N-1 values are NaN."""
    ret = np.full(len(prices), np.nan)
    for i in range(N - 1, len(prices)):
        ret[i] = np.std(prices[i - N + 1:i + 1])
    return ret


class GPNode:
    """Node in a GP program tree."""

    def __init__(self, value: str, terminal: bool = False):
        self.value = value
        self.terminal = terminal
        self.children: list[GPNode] = []
        self.params: dict = {}

    def copy(self) -> "GPNode":
        n = GPNode(self.value, self.terminal)
        n.params = self.params.copy()
        n.children = [c.copy() for c in self.children]
        return n

    def __repr__(self) -> str:
        if self.terminal:
            if self.value in ("sma", "lma"):
                return f"{self.value}({self.params.get('N', '?')})"
            if self.value == "ema":
                return f"ema({self.params.get('N', '?')},{self.params.get('alpha', '?'):.2f})"
            if self.value in ("rsi", "momentum", "volatility"):
                return f"{self.value}({self.params.get('N', '?')})"
            if self.value == "const":
                return f"{self.params.get('value', 0):.3f}"
            return self.value
        return f"({self.value} {' '.join(repr(c) for c in self.children)})"


class GP:
    """Tree-based Genetic Programming with optional structural early stopping."""

    def __init__(
        self,
        pop_size: int = 100,
        generations: int = 50,
        seed: int | None = None,
        parsimony_penalty: float = 0.0,
    ):
        if seed is not None:
            np.random.seed(seed)
        self.pop_size = pop_size
        self.generations = generations
        self.parsimony_penalty = parsimony_penalty
        self.funs = ["+", "-", "*", "/", "ABS", "MAX", "MIN", ">", "<", "AND", "IF"]
        self.terms = ["price", "sma", "lma", "ema", "rsi", "momentum", "volatility", "const"]
        self.arity = {"+": 2, "-": 2, "*": 2, "/": 2, "ABS": 1, "MAX": 2,
                      "MIN": 2, ">": 2, "<": 2, "AND": 2, "IF": 3}
        self.max_depth = 5

    def _random_tree(self, depth: int = 0, max_depth: int | None = None) -> GPNode:
        if max_depth is None:
            max_depth = self.max_depth
        if depth >= max_depth or (depth > 0 and np.random.rand() < 0.3):
            t = np.random.choice(self.terms)
            n = GPNode(t, True)
            if t in ("sma", "lma"):
                n.params["N"] = np.random.randint(5, 200)
            elif t == "ema":
                n.params["N"] = np.random.randint(5, 200)
                n.params["alpha"] = np.random.uniform(0.01, 0.99)
            elif t in ("rsi", "momentum", "volatility"):
                n.params["N"] = np.random.randint(5, 200)
            elif t == "const":
                n.params["value"] = np.random.uniform(-1, 1)
            return n
        f = np.random.choice(self.funs)
        n = GPNode(f, False)
        for _ in range(self.arity[f]):
            n.children.append(self._random_tree(depth + 1, max_depth))
        return n

    def _collect(self, node: GPNode):
        if node.terminal:
            if node.value in ("sma", "lma"):
                return {(node.value, node.params["N"])}
            if node.value == "ema":
                return {("ema", node.params["N"], node.params["alpha"])}
            if node.value in ("rsi", "momentum", "volatility"):
                return {(node.value, node.params["N"])}
            return set()
        r = set()
        for c in node.children:
            r |= self._collect(c)
        return r

    def _cache(self, tree: GPNode, prices: np.ndarray) -> dict:
        cache = {"price": prices}
        refs = self._collect(tree)
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
            elif ref[0] == "rsi":
                N = min(ref[1], n)
                cache[ref] = _rsi(prices, N)
            elif ref[0] == "momentum":
                N = min(ref[1], n)
                cache[ref] = _momentum(prices, N)
            elif ref[0] == "volatility":
                N = min(ref[1], n)
                cache[ref] = _volatility(prices, N)
        return cache

    def _eval(self, node: GPNode, cache: dict, idx: int) -> float:
        if node.terminal:
            if node.value == "price":
                return float(cache["price"][idx])
            if node.value in ("sma", "lma"):
                arr = cache[(node.value, node.params["N"])]
                return float(arr[idx]) if idx < len(arr) else float(cache["price"][idx])
            if node.value == "ema":
                arr = cache[("ema", node.params["N"], node.params["alpha"])]
                return float(arr[idx]) if idx < len(arr) else float(cache["price"][idx])
            if node.value == "const":
                return float(node.params["value"])
            if node.value in ("rsi", "momentum", "volatility"):
                arr = cache[(node.value, node.params["N"])]
                return float(arr[idx]) if idx < len(arr) and not np.isnan(arr[idx]) else float(cache["price"][idx])
            return 0.0
        v = [self._eval(c, cache, idx) for c in node.children]
        op = node.value
        if op == "+":
            return v[0] + v[1]
        if op == "-":
            return v[0] - v[1]
        if op == "*":
            return v[0] * v[1]
        if op == "/":
            return v[0] / (v[1] + 1e-10)
        if op == "ABS":
            return abs(v[0])
        if op == "MAX":
            return max(v[0], v[1])
        if op == "MIN":
            return min(v[0], v[1])
        if op == ">":
            return 1.0 if v[0] > v[1] else 0.0
        if op == "<":
            return 1.0 if v[0] < v[1] else 0.0
        if op == "AND":
            return 1.0 if (v[0] and v[1]) else 0.0
        if op == "IF":
            return v[1] if v[0] else v[2]
        return 0.0

    def evaluate(self, tree: GPNode, prices: np.ndarray, continuous: bool = False) -> np.ndarray:
        """Evaluate tree across all prices to generate signals.

        continuous=False: discrete {-1, 0, +1} (legacy)
        continuous=True:  continuous [0, 1] via sigmoid (position sizing)
        """
        n = len(prices)
        cache = self._cache(tree, prices)
        raw = np.zeros(n)
        for i in range(n):
            raw[i] = self._eval(tree, cache, i)
        if continuous:
            return 1.0 / (1.0 + np.exp(-np.clip(raw, -500, 500)))
        sig = np.zeros(n, dtype=int)
        prev = 0
        for i in range(n):
            curr = 1 if raw[i] > 0 else (-1 if raw[i] < 0 else 0)
            if curr != 0 and curr != prev:
                sig[i] = curr
                prev = curr
        return sig

    def _nodes(self, node: GPNode) -> list[GPNode]:
        ns = [node]
        for c in node.children:
            ns.extend(self._nodes(c))
        return ns

    def _tree_size(self, node: GPNode) -> int:
        """Number of nodes in the tree."""
        return 1 + sum(self._tree_size(c) for c in node.children)

    def structural_metrics(self, tree: GPNode) -> dict:
        """Compute structural overfitting-risk metrics for a tree.

        Returns dict with keys matching those in analyze_existing.py.
        """
        nodes = self._nodes(tree)
        total = len(nodes)
        internal = sum(1 for n in nodes if not n.terminal)
        terminals = total - internal

        def _depth(node):
            if not node.children:
                return 0
            return 1 + max(_depth(c) for c in node.children)

        if_count = sum(1 for n in nodes if n.value == 'IF')

        const_count = sum(1 for n in nodes if n.terminal and n.value == 'const')

        return {
            'tree_size': total,
            'depth': _depth(tree),
            'nesting_ratio': internal / total if total > 0 else 0,
            'if_count': if_count,
            'constant_ratio': const_count / terminals if terminals > 0 else 0,
        }

    def _crossover(self, p1: GPNode, p2: GPNode) -> GPNode:
        c = p1.copy()
        ns1 = self._nodes(c)
        ns2 = self._nodes(p2)
        if not ns1 or not ns2:
            return c
        n1 = np.random.choice(ns1)
        n2 = np.random.choice(ns2)
        n1.value = n2.value
        n1.terminal = n2.terminal
        n1.params = n2.params.copy()
        n1.children = [c.copy() for c in n2.children]
        return c

    def _mutate(self, ind: GPNode) -> GPNode:
        m = ind.copy()
        ns = self._nodes(m)
        if not ns:
            return m
        t = np.random.choice(ns)
        st = self._random_tree(max_depth=self.max_depth // 2)
        t.value = st.value
        t.terminal = st.terminal
        t.params = st.params.copy()
        t.children = [c.copy() for c in st.children]
        return m

    def _select(self, pop: list[GPNode], fit: list[float]) -> GPNode:
        idxs = np.random.choice(len(pop), size=3, replace=False)
        return pop[idxs[np.argmax([fit[i] for i in idxs])]]

    def optimize(self, fitness_fn, n_workers: int = 1,
                 early_stop: bool = False) -> dict:
        """Return dict with keys: best, fitness, history, stop_gen, stop_reason.

        Args:
            n_workers: number of threads for parallel fitness evaluation.
            early_stop: if True, stop when structural risk thresholds are exceeded.
        """
        pop = [self._random_tree() for _ in range(self.pop_size)]

        def _eval_all(population):
            if n_workers > 1:
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    return list(ex.map(fitness_fn, population))
            return [fitness_fn(ind) for ind in population]

        raw_fit = _eval_all(pop)
        fit = [f - self.parsimony_penalty * self._tree_size(ind) for f, ind in zip(raw_fit, pop)]
        history = []
        stop_gen = self.generations
        stop_reason = "completed"

        for gen in range(self.generations):
            # --- Early stopping check ---
            if early_stop:
                best_idx = int(np.argmax(fit))
                best_tree = pop[best_idx]
                metrics = self.structural_metrics(best_tree)

                if metrics['nesting_ratio'] > 0.7:
                    stop_gen = gen + 1
                    stop_reason = f"nesting_ratio={metrics['nesting_ratio']:.3f} > 0.7"
                    break
                if metrics['if_count'] > 2:
                    stop_gen = gen + 1
                    stop_reason = f"if_count={metrics['if_count']} > 2"
                    break
                if metrics['depth'] > 6:
                    stop_gen = gen + 1
                    stop_reason = f"depth={metrics['depth']} > 6"
                    break

            # --- One generation ---
            new = []
            best_idx = int(np.argmax(fit))
            new.append(pop[best_idx].copy())
            while len(new) < self.pop_size:
                r = np.random.rand()
                if r < 0.9:
                    p1 = self._select(pop, fit)
                    p2 = self._select(pop, fit)
                    new.append(self._crossover(p1, p2))
                elif r < 0.9 + 0.1:
                    p = self._select(pop, fit)
                    new.append(self._mutate(p))
                else:
                    new.append(self._select(pop, fit).copy())
            pop = new
            raw_fit = _eval_all(pop)
            fit = [f - self.parsimony_penalty * self._tree_size(ind) for f, ind in zip(raw_fit, pop)]
            history.append(float(max(fit)))

        best_idx = int(np.argmax(fit))
        return {
            "best": pop[best_idx],
            "fitness": fit[best_idx],
            "history": history,
            "stop_gen": stop_gen,
            "stop_reason": stop_reason,
        }
