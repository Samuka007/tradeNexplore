# AI Trading Bot Framework

A Python framework for comparing nature-inspired optimization algorithms (PSO, ABC, HS, GP) on trading strategy optimization.

## Architecture

```
trading_bot/
├── filters.py           # WMA filters (SMA, LMA, EMA) via convolution
├── strategy.py          # Trading strategies (dual crossover, MACD, GP tree)
├── backtester.py        # Backtest engine with fitness evaluation
├── data_loader.py       # Kaggle BTC data loading with train/val/test split
├── algorithms/          # Optimization algorithm interfaces
│   ├── base.py          # ContinuousOptimizer, StructuralOptimizer ABCs
│   ├── pso.py           # PSO implementation (placeholder)
│   ├── abc.py           # ABC implementation (placeholder)
│   ├── harmony_search.py # HS implementation (placeholder)
│   └── genetic_programming.py # GP implementation (placeholder)
├── experiments/         # Experiment orchestration
│   └── runner.py        # Run continuous/structural/hierarchical experiments
├── visualization.py     # Plotting utilities
└── check_fitness.py     # CLI for checking fixed policy fitness
```

## Quick Start

### Check Fitness of a Fixed Policy

```bash
# Using synthetic data (no Kaggle download needed)
python -m trading_bot.check_fitness --synthetic

# With custom parameters
python -m trading_bot.check_fitness --params "1,0,0,10,20,30,0.3,1,0,0,15,25,35,0.3" --synthetic

# MACD strategy with baseline comparison
python -m trading_bot.check_fitness --strategy macd --params "12,0.2,26,0.1,9,0.2,0.0" --synthetic --baseline
```

### Load Data

```python
from trading_bot.data_loader import load_or_generate_data

# Auto-fallback to synthetic if Kaggle data not present
dataset = load_or_generate_data("data/btc_daily_2014_2022.csv")
print(f"Train: {len(dataset.train_prices)}, Val: {len(dataset.val_prices)}, Test: {len(dataset.test_prices)}")
```

### Run an Experiment

```python
from trading_bot.experiments import ExperimentRunner
from trading_bot.algorithms import StubPSO
from trading_bot.data_loader import load_or_generate_data
import numpy as np

dataset = load_or_generate_data(fallback_to_synthetic=True)
runner = ExperimentRunner(dataset)

# 14D bounds for dual_crossover: [w1,w2,w3,d1,d2,d3,a3,w4,w5,w6,d4,d5,d6,a6]
bounds = [(0.0, 1.0)]*3 + [(2.0, 200.0)]*3 + [(0.01, 0.99)] + [(0.0, 1.0)]*3 + [(2.0, 200.0)]*3 + [(0.01, 0.99)]

result = runner.run_continuous(StubPSO(), bounds, strategy_type="dual_crossover")
print(f"Train fitness: {result.train_fitness:.2f}")
print(f"Test fitness: {result.test_fitness:.2f}")
```

## Data

The framework expects the **Kaggle Bitcoin Historical Dataset**:
- Download from: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
- Place at: `data/btc_daily_2014_2022.csv`
- Split per specification:
  - **Train**: 2014-2017 (algorithm optimization)
  - **Validation**: 2018-2019 (hyperparameter tuning, early stopping)
  - **Test**: 2020-2022 (final evaluation, unseen data)

If real data is unavailable, synthetic data is auto-generated.

## Testing

```bash
pytest tests/ -v
```

## Project Structure

- **Framework components** (implemented): data loader, backtester, strategy interfaces, algorithm ABCs, experiment runner, visualization
- **Core algorithms** (stubs provided): PSO, ABC, Harmony Search, Genetic Programming — implement `optimize()` in respective modules

## Specification Compliance

- ✅ Convolution-based WMA filters (np.convolve)
- ✅ $1000 initial cash, 3% fee per trade
- ✅ Full position buy/sell with forced liquidation
- ✅ Fitness = final cash
- ✅ Train/Validation/Test split (2014-2017 / 2018-2019 / 2020-2022)
- ✅ Dual-crossover and MACD strategies
- ✅ GP TreeStrategy for structural optimization
- ✅ Buy-and-hold baseline comparison
- ✅ Penalized fitness function (trade count penalty)
- ✅ Modular architecture with clean interfaces
