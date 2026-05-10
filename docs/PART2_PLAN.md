# Part 2 Trading Bot Implementation Plan

## Based on: AI Trading Bots Project Survey (PSO, ABC, GP, HS)

---

## 1. Executive Summary

This plan implements a comparative trading bot framework using the four nature-inspired algorithms surveyed in Part 1: **Particle Swarm Optimization (PSO)**, **Artificial Bee Colony (ABC)**, **Genetic Programming (GP)**, and **Harmony Search (HS)**. Following the survey's conclusion that these algorithms represent fundamentally different optimization paradigms, we adopt a **two-stage experimental design**:

- **Stage 1**: Compare structure discovery (GP) vs. parameter optimization (PSO, ABC, HS) within fixed structures
- **Stage 2**: Hierarchical combination — parameter refinement on GP-discovered structures

---

## 2. Bot Design and Parameterization

### 2.1 Building Blocks

Based on the specification (Section 2), our bots use **Weighted Moving Average (WMA) filters** as fundamental components:

| Component | Parameters | Description |
|-----------|------------|-------------|
| **SMA** | Window size `N` | Simple Moving Average — equal weights |
| **LMA** | Window size `N` | Linear-Weighted MA — triangular filter |
| **EMA** | Window size `N`, decay `α` | Exponential MA — exponential decay |
| **Sign Change Detector** | — | Convolution filter [1, -1]/2 for crossover detection |

All filters implemented via **convolution** for efficiency and generality:
```python
def wma(P, N, kernel):
    return np.convolve(pad(P, N), kernel, 'valid')
```

### 2.2 Bot Architecture

#### **Bot Variant A: Dual-Crossover Bot (Parametric Optimization)**
This is our primary parametric optimization target for PSO, ABC, and HS.

```
Price Signal → [HIGH Component] → Difference → Sign → Crossover Detection → Buy/Sell
            → [LOW Component]  →
```

Where each component is a **weighted combination** of SMA, LMA, EMA:

```
HIGH = (w1·SMA(d1) + w2·LMA(d2) + w3·EMA(d3, α3)) / Σwi
LOW  = (w4·SMA(d4) + w5·LMA(d5) + w6·EMA(d6, α6)) / Σwi
```

**Parameter vector (14D):**
```
[w1, w2, w3, d1, d2, d3, α3, w4, w5, w6, d4, d5, d6, α6]
```

**Constraints:**
- `wi ≥ 0` (weights non-negative)
- `di ∈ [5, 200]` (window sizes, integer)
- `α ∈ [0.01, 0.99]` (EMA decay factor)

#### **Bot Variant B: MACD Bot (Parametric Optimization)**
Standard MACD(12,26,9) generalization:

```
MACD = EMA(d1, α1) - EMA(d2, α2)
Signal = EMA(MACD, d3, α3)
Trigger = Sign(MACD - Signal)
```

**Parameter vector (7D):** `[d1, α1, d2, α2, d3, α3, threshold]`

#### **Bot Variant C: GP-Discovered Structure Bot (Structural Optimization)**
GP evolves **tree-structured trading rules** where:
- **Terminals:** Price, SMA(N), LMA(N), EMA(N,α), Volume, constants
- **Functions:** +, -, ×, ÷, >, <, AND, OR, IF-THEN

Example evolved rule:
```
IF (SMA(20) > EMA(50, 0.3)) AND (Volume > SMA_Volume(10))
THEN BUY
ELSE IF (LMA(5) < SMA(30))
THEN SELL
```

---

## 3. Algorithm Mapping

### 3.1 Algorithm Roles

| Algorithm | Team Member | Bot Variant | Role | Optimization Type |
|-----------|-------------|-------------|------|-------------------|
| **PSO** | Simona Han | A, B | Parameter optimizer for fixed-structure bots | Continuous 14D/7D |
| **ABC** | Xinqi Lin | A, B | Parameter optimizer (alternative to PSO) | Continuous 14D/7D |
| **GP** | Lyuchen Dai | C | Structure discovery + rule evolution | Tree-based discrete |
| **HS** | Mika Li | A, B | Baseline parameter optimizer | Continuous 14D/7D |

### 3.2 Why This Mapping?

Following the survey's taxonomy (Section "Conclusion and Comparison"):

- **PSO & ABC** are *parametric optimizers*: they improve what is already designed. They compete on finding optimal weights and durations within fixed bot structures.
- **GP** is a *structural optimizer*: it discovers what to design. It searches the space of indicator combinations and logical structures that parametric methods cannot reach.
- **HS** is a *general-purpose heuristic baseline*: it searches without strong structural assumptions. If PSO/ABC cannot consistently outperform HS, their added complexity is unjustified.

---

## 4. Implementation Details

### 4.1 Core Framework (`bot_framework.py`)

```python
class WMAFilter:
    """Base class for weighted moving average filters."""
    def __init__(self, window_size, filter_type='sma', alpha=None):
        self.N = window_size
        self.type = filter_type
        self.alpha = alpha
        self.kernel = self._build_kernel()
    
    def _build_kernel(self):
        if self.type == 'sma':
            return np.ones(self.N) / self.N
        elif self.type == 'lma':
            k = np.arange(self.N)
            return (2 / (self.N + 1)) * (1 - k / self.N)
        elif self.type == 'ema':
            k = np.arange(self.N)
            return self.alpha * (1 - self.alpha) ** k
    
    def apply(self, prices):
        padded = self._pad(prices)
        return np.convolve(padded, self.kernel, 'valid')
    
    def _pad(self, P):
        padding = -np.flip(P[1:self.N])
        return np.append(padding, P)


class TradingBot:
    """Parameterized trading bot using WMA crossover strategy."""
    def __init__(self, params, bot_type='dual_crossover'):
        self.params = params
        self.type = bot_type
        self.cash = 1000.0
        self.btc = 0.0
        self.holding_cash = True
    
    def generate_signals(self, prices):
        """Returns array of signals: 1=buy, -1=sell, 0=hold"""
        if self.type == 'dual_crossover':
            return self._dual_crossover_signals(prices)
        elif self.type == 'macd':
            return self._macd_signals(prices)
    
    def evaluate(self, prices, signals):
        """Back-test bot on price sequence. Returns final cash."""
        fee_rate = 0.03
        for i, signal in enumerate(signals):
            if signal == 1 and self.holding_cash:
                self.btc = self.cash * (1 - fee_rate) / prices[i]
                self.cash = 0.0
                self.holding_cash = False
            elif signal == -1 and not self.holding_cash:
                self.cash = self.btc * prices[i] * (1 - fee_rate)
                self.btc = 0.0
                self.holding_cash = True
        
        # Final liquidation
        if not self.holding_cash:
            self.cash = self.btc * prices[-1] * (1 - fee_rate)
        
        return self.cash
```

### 4.2 Algorithm Implementations

#### PSO (`algorithms/pso.py`)
```python
class PSO:
    """Particle Swarm Optimization for continuous parameter tuning."""
    def __init__(self, n_particles=30, dimensions=14, 
                 w=0.729, c1=2.05, c2=2.05, max_iter=100):
        self.n_particles = n_particles
        self.dim = dimensions
        self.w = w          # Inertia weight
        self.c1 = c1        # Cognitive coefficient
        self.c2 = c2        # Social coefficient
        self.max_iter = max_iter
    
    def optimize(self, fitness_fn, bounds):
        """
        Optimize parameters within given bounds.
        
        Args:
            fitness_fn: Function that takes parameter vector and returns fitness
            bounds: List of (min, max) tuples for each dimension
        
        Returns:
            best_params, best_fitness, convergence_history
        """
        # Initialize swarm
        particles = np.random.uniform(
            [b[0] for b in bounds], 
            [b[1] for b in bounds], 
            (self.n_particles, self.dim)
        )
        velocities = np.zeros((self.n_particles, self.dim))
        
        pbest = particles.copy()
        pbest_fitness = np.array([fitness_fn(p) for p in particles])
        
        gbest_idx = np.argmax(pbest_fitness)
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = pbest_fitness[gbest_idx]
        
        history = [gbest_fitness]
        
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(2)
                
                # Velocity update
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (pbest[i] - particles[i]) +
                    self.c2 * r2 * (gbest - particles[i])
                )
                
                # Position update with boundary handling
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 
                                      [b[0] for b in bounds], 
                                      [b[1] for b in bounds])
                
                # Evaluate
                fitness = fitness_fn(particles[i])
                
                if fitness > pbest_fitness[i]:
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = fitness
                    
                    if fitness > gbest_fitness:
                        gbest = particles[i].copy()
                        gbest_fitness = fitness
            
            history.append(gbest_fitness)
            
            # Adaptive inertia (optional enhancement)
            self.w = 0.9 - (0.5 * iteration / self.max_iter)
        
        return gbest, gbest_fitness, history
```

#### ABC (`algorithms/abc.py`)
```python
class ABC:
    """Artificial Bee Colony for continuous parameter tuning."""
    def __init__(self, n_bees=50, dimensions=14, limit=20, max_cycles=100):
        self.n_bees = n_bees
        self.dim = dimensions
        self.limit = limit      # Scout limit
        self.max_cycles = max_cycles
        self.employed = n_bees // 2
        self.onlookers = n_bees // 2
    
    def optimize(self, fitness_fn, bounds):
        """
        ABC optimization with decoupled exploration/exploitation.
        Uses dimensionally decoupled perturbation: v_ij = x_ij + φ(x_ij - x_kj)
        """
        # Initialize food sources
        foods = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (self.employed, self.dim)
        )
        fitness = np.array([fitness_fn(f) for f in foods])
        trial = np.zeros(self.employed)
        
        history = []
        
        for cycle in range(self.max_cycles):
            # Employed bees phase (local exploitation)
            for i in range(self.employed):
                j = np.random.randint(self.dim)
                k = np.random.choice([x for x in range(self.employed) if x != i])
                phi = np.random.uniform(-1, 1)
                
                new_food = foods[i].copy()
                new_food[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                new_food[j] = np.clip(new_food[j], bounds[j][0], bounds[j][1])
                
                new_fitness = fitness_fn(new_food)
                
                # Greedy selection
                if new_fitness > fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # Onlooker bees phase (roulette wheel selection)
            probs = fitness / fitness.sum()
            for _ in range(self.onlookers):
                i = np.random.choice(self.employed, p=probs)
                j = np.random.randint(self.dim)
                k = np.random.choice([x for x in range(self.employed) if x != i])
                phi = np.random.uniform(-1, 1)
                
                new_food = foods[i].copy()
                new_food[j] = foods[i][j] + phi * (foods[i][j] - foods[k][j])
                new_food[j] = np.clip(new_food[j], bounds[j][0], bounds[j][1])
                
                new_fitness = fitness_fn(new_food)
                
                if new_fitness > fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trial[i] = 0
                else:
                    trial[i] += 1
            
            # Scout bees phase (global exploration)
            for i in range(self.employed):
                if trial[i] > self.limit:
                    foods[i] = np.random.uniform(
                        [b[0] for b in bounds],
                        [b[1] for b in bounds],
                        self.dim
                    )
                    fitness[i] = fitness_fn(foods[i])
                    trial[i] = 0
            
            best_idx = np.argmax(fitness)
            history.append(fitness[best_idx])
        
        best_idx = np.argmax(fitness)
        return foods[best_idx], fitness[best_idx], history
```

#### HS (`algorithms/harmony_search.py`)
```python
class HarmonySearch:
    """Harmony Search for continuous parameter tuning."""
    def __init__(self, hms=30, hmcr=0.9, par=0.3, 
                 bw=0.01, max_iter=100):
        self.hms = hms          # Harmony Memory Size
        self.hmcr = hmcr        # Harmony Memory Considering Rate
        self.par = par          # Pitch Adjusting Rate
        self.bw = bw            # Bandwidth
        self.max_iter = max_iter
    
    def optimize(self, fitness_fn, bounds):
        """
        HS optimization using musical improvisation analogy.
        """
        # Initialize Harmony Memory
        hm = np.random.uniform(
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            (self.hms, len(bounds))
        )
        hm_fitness = np.array([fitness_fn(h) for h in hm])
        
        history = []
        
        for iteration in range(self.max_iter):
            new_harmony = np.zeros(len(bounds))
            
            for j in range(len(bounds)):
                if np.random.rand() < self.hmcr:
                    # Select from harmony memory
                    new_harmony[j] = hm[np.random.randint(self.hms), j]
                    
                    # Pitch adjustment
                    if np.random.rand() < self.par:
                        new_harmony[j] += np.random.uniform(-1, 1) * self.bw
                else:
                    # Random selection
                    new_harmony[j] = np.random.uniform(bounds[j][0], bounds[j][1])
                
                new_harmony[j] = np.clip(new_harmony[j], bounds[j][0], bounds[j][1])
            
            new_fitness = fitness_fn(new_harmony)
            
            # Replace worst harmony if better
            worst_idx = np.argmin(hm_fitness)
            if new_fitness > hm_fitness[worst_idx]:
                hm[worst_idx] = new_harmony
                hm_fitness[worst_idx] = new_fitness
            
            history.append(np.max(hm_fitness))
        
        best_idx = np.argmax(hm_fitness)
        return hm[best_idx], hm_fitness[best_idx], history
```

#### GP (`algorithms/genetic_programming.py`)
```python
class GeneticProgramming:
    """Genetic Programming for structure discovery."""
    def __init__(self, population_size=100, generations=50,
                 crossover_rate=0.9, mutation_rate=0.1,
                 max_depth=5, tournament_size=3):
        self.pop_size = population_size
        self.generations = generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        
        # Function set
        self.functions = ['+', '-', '*', '>', '<', 'AND', 'IF']
        # Terminal set
        self.terminals = ['price', 'sma', 'lma', 'ema', 'volume', 'const']
    
    class Node:
        def __init__(self, value, is_terminal=False):
            self.value = value
            self.is_terminal = is_terminal
            self.children = []
            self.params = {}  # Parameters for terminals (e.g., N for SMA)
    
    def create_random_tree(self, depth=0, max_depth=5):
        """Grow random program tree with ramped half-and-half."""
        if depth >= max_depth or (depth > 0 and np.random.rand() < 0.3):
            # Terminal
            terminal = np.random.choice(self.terminals)
            node = self.Node(terminal, is_terminal=True)
            if terminal in ['sma', 'lma']:
                node.params['N'] = np.random.randint(5, 200)
            elif terminal == 'ema':
                node.params['N'] = np.random.randint(5, 200)
                node.params['alpha'] = np.random.uniform(0.01, 0.99)
            elif terminal == 'const':
                node.params['value'] = np.random.uniform(-1, 1)
            return node
        else:
            # Function
            func = np.random.choice(self.functions)
            node = self.Node(func, is_terminal=False)
            
            if func in ['+', '-', '*', '>', '<']:
                arity = 2
            elif func == 'AND':
                arity = 2
            elif func == 'IF':
                arity = 3
            
            for _ in range(arity):
                node.children.append(self.create_random_tree(depth + 1, max_depth))
            
            return node
    
    def evaluate_tree(self, node, context):
        """Evaluate tree on given price/volume context."""
        if node.is_terminal:
            if node.value == 'price':
                return context['price']
            elif node.value == 'volume':
                return context['volume']
            elif node.value == 'sma':
                return self._sma(context['prices'], node.params['N'])
            elif node.value == 'lma':
                return self._lma(context['prices'], node.params['N'])
            elif node.value == 'ema':
                return self._ema(context['prices'], node.params['N'], node.params['alpha'])
            elif node.value == 'const':
                return node.params['value']
        else:
            if node.value == '+':
                return self.evaluate_tree(node.children[0], context) + \
                       self.evaluate_tree(node.children[1], context)
            elif node.value == '-':
                return self.evaluate_tree(node.children[0], context) - \
                       self.evaluate_tree(node.children[1], context)
            elif node.value == '*':
                return self.evaluate_tree(node.children[0], context) * \
                       self.evaluate_tree(node.children[1], context)
            elif node.value == '>':
                return 1 if self.evaluate_tree(node.children[0], context) > \
                           self.evaluate_tree(node.children[1], context) else 0
            elif node.value == '<':
                return 1 if self.evaluate_tree(node.children[0], context) < \
                           self.evaluate_tree(node.children[1], context) else 0
            elif node.value == 'AND':
                return 1 if (self.evaluate_tree(node.children[0], context) and 
                            self.evaluate_tree(node.children[1], context)) else 0
            elif node.value == 'IF':
                return self.evaluate_tree(node.children[1], context) if \
                       self.evaluate_tree(node.children[0], context) else \
                       self.evaluate_tree(node.children[2], context)
    
    def crossover(self, parent1, parent2):
        """Subtree crossover."""
        child1 = self._copy_tree(parent1)
        child2 = self._copy_tree(parent2)
        
        # Select random nodes
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)
        
        node1 = np.random.choice(nodes1)
        node2 = np.random.choice(nodes2)
        
        # Swap subtrees
        temp = self._copy_tree(node1)
        node1.value = node2.value
        node1.is_terminal = node2.is_terminal
        node1.params = node2.params.copy()
        node1.children = [self._copy_tree(c) for c in node2.children]
        
        return child1
    
    def mutate(self, individual):
        """Subtree mutation."""
        mutant = self._copy_tree(individual)
        nodes = self._get_all_nodes(mutant)
        
        # Replace random subtree with new random tree
        target = np.random.choice(nodes)
        new_subtree = self.create_random_tree(max_depth=self.max_depth // 2)
        
        target.value = new_subtree.value
        target.is_terminal = new_subtree.is_terminal
        target.params = new_subtree.params.copy()
        target.children = [self._copy_tree(c) for c in new_subtree.children]
        
        return mutant
    
    def optimize(self, fitness_fn):
        """Main GP optimization loop."""
        population = [self.create_random_tree() for _ in range(self.pop_size)]
        fitness = [fitness_fn(ind) for ind in population]
        
        history = []
        
        for gen in range(self.generations):
            new_population = []
            
            # Elitism: keep best individual
            best_idx = np.argmax(fitness)
            new_population.append(self._copy_tree(population[best_idx]))
            
            while len(new_population) < self.pop_size:
                if np.random.rand() < self.cx_rate:
                    # Crossover
                    parent1 = self._tournament_select(population, fitness)
                    parent2 = self._tournament_select(population, fitness)
                    child = self.crossover(parent1, parent2)
                elif np.random.rand() < self.mut_rate:
                    # Mutation
                    parent = self._tournament_select(population, fitness)
                    child = self.mutate(parent)
                else:
                    # Reproduction
                    child = self._copy_tree(
                        self._tournament_select(population, fitness)
                    )
                
                new_population.append(child)
            
            population = new_population
            fitness = [fitness_fn(ind) for ind in population]
            
            history.append(np.max(fitness))
        
        best_idx = np.argmax(fitness)
        return population[best_idx], fitness[best_idx], history
```

---

## 5. Experimental Design

### 5.1 Data

| Dataset | Period | Purpose |
|---------|--------|---------|
| **Training** | 2014–2019 | Algorithm optimization |
| **Validation** | 2018–2019 (subset) | Hyperparameter tuning, early stopping |
| **Test** | 2020–2022 | Final evaluation (unseen data) |

**Source:** Kaggle Bitcoin Historical Dataset (daily, hourly, or minute data)

### 5.2 Fitness Function

```python
def fitness(params, bot_type, prices, use_baseline=False):
    """
    Evaluate bot on historical price sequence.
    
    Args:
        params: Bot parameters (algorithm-specific)
        bot_type: 'dual_crossover', 'macd', or 'gp_tree'
        prices: OHLCV price sequence
        use_baseline: If True, compare against buy-and-hold
    
    Returns:
        Final cash value (fitness score)
    """
    bot = TradingBot(params, bot_type)
    signals = bot.generate_signals(prices)
    final_cash = bot.evaluate(prices, signals)
    
    if use_baseline:
        # Normalize against buy-and-hold strategy
        buy_hold = 1000 * (prices[-1] / prices[0])
        return final_cash / buy_hold  # >1 means beating market
    
    return final_cash
```

**Transaction costs:** 3% per trade

### 5.3 Experiment 1: Parametric Optimization Comparison

**Objective:** Compare PSO, ABC, and HS on fixed-structure bots

| Configuration | Description |
|---------------|-------------|
| **Bot Structure** | Dual-Crossover (14D) |
| **Algorithms** | PSO, ABC, HS |
| **Runs per algorithm** | 30 independent runs |
| **Evaluation** | Final cash on training set, tested on hold-out |
| **Metrics** | Mean, std, best, worst, convergence curves |

**Hypothesis:** ABC's decoupled exploration/exploitation will outperform PSO's velocity-based approach on this multimodal landscape; HS will serve as competitive baseline.

### 5.4 Experiment 2: Structure Discovery vs. Parameter Optimization

**Objective:** Test whether GP's structural search offers advantages over parametric methods

| Configuration | Description |
|---------------|-------------|
| **GP Bot** | Tree-discovered rules (variable structure) |
| **Baseline** | Best parametric bot from Experiment 1 |
| **GP Settings** | Population=100, Generations=50, Max Depth=5 |
| **Controls** | Same training/test split, same fitness function |

**Hypothesis:** GP will discover novel indicator combinations in-sample but may overfit; strict depth limits and train-test separation are essential.

### 5.5 Experiment 3: Hierarchical Combination (Two-Stage)

**Objective:** Test if parameter refinement on GP-discovered structures yields improvement

**Stage 1:** GP discovers promising bot structures on training data
**Stage 2:** Extract structure template from best GP tree, fix topology
**Stage 3:** PSO/ABC optimizes numeric parameters of GP-discovered template

| Configuration | Description |
|---------------|-------------|
| **Template** | GP-discovered rule structure (fixed topology) |
| **Optimizers** | PSO, ABC (optimize template parameters) |
| **Baseline** | Original GP tree, Best parametric bot |

**Hypothesis:** Hierarchical combination will outperform both pure GP and pure parametric methods.

### 5.6 Statistical Validation

- **Multiple runs:** 30 independent runs per configuration
- **Reporting:** Mean ± standard deviation
- **Comparison:** Wilcoxon rank-sum test or paired t-test where appropriate
- **Visualization:** Convergence curves, box plots of final fitness distributions

---

## 6. Evaluation and Comparison Framework

### 6.1 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Final Cash** | USD after back-test | Maximize |
| **Sharpe Ratio** | Risk-adjusted return | Maximize |
| **Max Drawdown** | Largest peak-to-trough decline | Minimize |
| **Win Rate** | % of profitable trades | Maximize |
| **Trades** | Number of transactions | Moderate |
| **vs Buy-Hold** | Outperformance ratio | > 1.0 |

### 6.2 Visualization

```python
def visualize_results(results, prices):
    """Generate comprehensive result visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Convergence curves
    for algo, history in results['convergence'].items():
        axes[0,0].plot(history, label=algo)
    axes[0,0].set_title('Convergence Comparison')
    axes[0,0].legend()
    
    # 2. Box plot of final fitness
    axes[0,1].boxplot([results['fitness'][a] for a in results['algorithms']],
                      labels=results['algorithms'])
    axes[0,1].set_title('Fitness Distribution (30 runs)')
    
    # 3. Price + trades overlay
    axes[0,2].plot(prices, label='Price', alpha=0.7)
    for algo in results['algorithms']:
        trades = results['trades'][algo]
        buy_points = [t for t in trades if t['type'] == 'buy']
        sell_points = [t for t in trades if t['type'] == 'sell']
        axes[0,2].scatter(buy_points, [prices[t['idx']] for t in buy_points], 
                         marker='^', color='green', label=f'{algo} buys')
        axes[0,2].scatter(sell_points, [prices[t['idx']] for t in sell_points], 
                         marker='v', color='red', label=f'{algo} sells')
    axes[0,2].set_title('Trade Points')
    
    # 4. Equity curves
    for algo, equity in results['equity_curves'].items():
        axes[1,0].plot(equity, label=algo)
    axes[1,0].set_title('Equity Curves')
    axes[1,0].legend()
    
    # 5. Drawdown analysis
    for algo, equity in results['equity_curves'].items():
        drawdown = compute_drawdown(equity)
        axes[1,1].fill_between(range(len(drawdown)), drawdown, alpha=0.3, label=algo)
    axes[1,1].set_title('Drawdown Analysis')
    axes[1,1].legend()
    
    # 6. Algorithm comparison radar
    categories = ['Return', 'Sharpe', 'Win Rate', 'Low Drawdown']
    # ... radar chart code
    
    plt.tight_layout()
    return fig
```

---

## 7. Deliverables

### 7.1 Code Structure

```
trading-bot-project/
├── README.md
├── requirements.txt
├── data/
│   ├── btc_daily_2014_2022.csv
│   └── data_loader.py
├── core/
│   ├── __init__.py
│   ├── filters.py           # WMA, SMA, LMA, EMA implementations
│   ├── bot.py               # TradingBot class
│   ├── evaluator.py         # Back-testing engine
│   └── visualization.py     # Plotting utilities
├── algorithms/
│   ├── __init__.py
│   ├── pso.py               # Particle Swarm Optimization
│   ├── abc.py               # Artificial Bee Colony
│   ├── harmony_search.py    # Harmony Search
│   └── genetic_programming.py  # Genetic Programming
├── experiments/
│   ├── experiment1_parametric_comparison.py
│   ├── experiment2_structure_vs_parametric.py
│   ├── experiment3_hierarchical.py
│   └── run_all.py
├── results/
│   ├── convergence_plots/
│   ├── trade_visualizations/
│   └── statistics/
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_single_algorithm_demo.ipynb
    └── 03_full_comparison.ipynb
```

### 7.2 Report Outline (3000 words max)

1. **Introduction** (200 words)
   - Brief context from Part 1 survey
   - Experimental objectives

2. **Bot Design** (400 words)
   - WMA filter building blocks
   - Parameterization choices
   - Hypothesis space dimensionality discussion

3. **Algorithm Implementation** (600 words)
   - PSO, ABC, HS, GP implementations
   - Key hyperparameters and their tuning
   - Differences from surveyed algorithms

4. **Experimental Design** (500 words)
   - Data split and preprocessing
   - Fitness function
   - Three experiments and their rationale

5. **Results** (800 words)
   - Convergence comparisons
   - Final performance on test set
   - Statistical significance

6. **Discussion** (400 words)
   - Algorithm behavior observations
   - Overfitting analysis
   - Two-stage design effectiveness

7. **Conclusion** (100 words)
   - Key findings
   - Future directions

### 7.3 Presentation Outline (25 minutes)

| Section | Duration | Content |
|---------|----------|---------|
| Algorithm Overview | 3 min | Brief recap of 4 algorithms from Part 1 |
| Bot Design | 4 min | Building blocks, parameterization, hypothesis space |
| Algorithm Selection | 4 min | Why each algorithm was chosen, implementation highlights |
| Experiments | 5 min | Design, data, evaluation methodology |
| Results | 5 min | Key visualizations, performance comparisons |
| Conclusions | 2 min | What we learned, limitations, future work |
| Q&A buffer | 2 min | — |

---

## 8. Timeline and Task Allocation

### Phase 1: Infrastructure (Week 1)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Data loading and preprocessing | All | `data_loader.py` |
| WMA filter implementations | All | `filters.py` |
| Basic bot framework | All | `bot.py`, `evaluator.py` |

### Phase 2: Algorithm Implementation (Week 1-2)

| Algorithm | Owner | File |
|-----------|-------|------|
| PSO | Simona Han | `algorithms/pso.py` |
| ABC | Xinqi Lin | `algorithms/abc.py` |
| HS | Mika Li | `algorithms/harmony_search.py` |
| GP | Lyuchen Dai | `algorithms/genetic_programming.py` |

### Phase 3: Experimentation (Week 2-3)

| Experiment | Owner | Deliverable |
|------------|-------|-------------|
| Experiment 1 (Parametric comparison) | Simona + Xinqi + Mika | `experiment1_parametric_comparison.py` |
| Experiment 2 (GP structure discovery) | Lyuchen | `experiment2_structure_vs_parametric.py` |
| Experiment 3 (Hierarchical) | All | `experiment3_hierarchical.py` |

### Phase 4: Analysis and Reporting (Week 3-4)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Result visualization | All | Plots, tables |
| Report writing | All | `report.pdf` |
| Presentation | All | Video recording |
| Code finalization | All | Clean `.ipynb` |

---

## 9. Risk Assessment and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Overfitting** | High | Strict train/test split; depth limits for GP; parsimony pressure |
| **Computation time** | Medium | Start with daily data; reduce population/generations for testing |
| **Premature convergence** | Medium | Adaptive inertia (PSO); scout bees (ABC); multiple runs |
| **GP bloat** | High | Depth limits; parsimony penalty in fitness; subtree pruning |
| **Algorithm bugs** | Medium | Unit tests for each algorithm; visual convergence checks |

---

## 10. Key Decisions Log

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| **Dual-crossover as primary bot** | Generalizes SMA crossover; natural 14D parameter space | MACD only; custom WMA weights |
| **Daily data first** | Computationally tractable; sufficient for proof of concept | Hourly/minute data (more noise, slower) |
| **Fixed 3% fee** | Specification requirement; realistic for crypto | Variable fee models |
| **GP max_depth=5** | Balances expressiveness and overfitting | Deeper trees (risk bloat); shallower trees (limited expressiveness) |
| **30 independent runs** | Statistical robustness; standard in optimization literature | 10 runs (insufficient); 100 runs (excessive) |
| **Two-stage design** | Directly tests survey hypothesis about representation vs. parameter optimization | Single-stage only; ensemble methods |

---

## 11. References from Survey

The implementation directly leverages insights from the surveyed literature:

1. **PSO**: Kennedy & Eberhart (1995) — velocity/position updates, inertia weight adaptation (Shi & Eberhart, 1998)
2. **ABC**: Karaboga (2007) — decoupled perturbation equation, employed/onlooker/scout division
3. **GP**: Koza (1992), Allen & Karjalainen (1999) — tree representation, closure property, overfitting warnings
4. **HS**: Geem et al. (2001) — HMCR/PAR parameters, memory-based search, comparison with GA

---

## Appendix: Quick Start Commands

```bash
# Setup
pip install numpy pandas matplotlib ccxt

# Download data
python data/download_btc_data.py

# Run single algorithm test
python experiments/experiment1_parametric_comparison.py --algorithm pso --runs 5

# Run all experiments
python experiments/run_all.py

# Generate visualizations
python core/visualization.py --results-dir results/

# Jupyter notebook demo
jupyter notebook notebooks/02_single_algorithm_demo.ipynb
```
