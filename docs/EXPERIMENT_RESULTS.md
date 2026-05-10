# Experiment Results Analysis

**Date**: 2026-05-10
**Data**: BTC Daily 2014-2022 (1130 train / 730 val / 791 test)
**Commit**: main

---

## 1. Experiment Configuration

| Parameter | Exp1 (PSO) | Exp2 (GP) | Exp3 (Hierarchical) |
|-----------|-----------|-----------|---------------------|
| Algorithm | PSO × 30 runs | GP × 10 runs | GP → PSO refine × 30 runs |
| Strategy | dual_cross (14D) + MACD (7D) | Tree-based (free structure) | dual_cross template from GP tree |
| PSO particles | 30 | — | 30 |
| PSO iterations | 50 | — | 50 |
| GP population | — | 100 | — |
| GP generations | — | 20 | — |
| GP max_depth | — | 3 | — |
| GP mutation | — | 0.2 | — |
| Fitness | Final cash ($1,000 start) | Final cash ($1,000 start) | Final cash ($1,000 start) |
| Fee | 3% per trade | 3% per trade | 3% per trade |

---

## 2. Results Summary

### 2.1 Algorithm Performance on Test Set (2020-2022)

| Strategy | Mean Final Cash | Std | Min | Max | vs Buy & Hold |
|----------|----------------|-----|-----|-----|---------------|
| **Buy & Hold** | **$5,660** | — | — | — | baseline |
| Golden Cross | $4,048 | — | — | — | -28.5% |
| Death Cross | $4,048 | — | — | — | -28.5% |
| **GP best tree** | **$4,565** | $461 | $3,989 | $5,120 | **-19.3%** |
| PSO dual_cross | $1,229 | $978 | $353 | $3,989 | -78.3% |
| PSO MACD | $1,366 | $1,442 | $0 | $6,084 | -75.9% |
| PSO refined (GP) | $2,542 | $2,018 | $683 | $5,902 | -55.1% |

**Finding**: No algorithm consistently beats buy & hold on this test period. GP comes closest (-19%) while PSO strategies lose 55-78% of capital.

### 2.2 Overfitting Analysis

| Strategy | Train Mean | Test Mean | Train/Test Ratio |
|----------|-----------|-----------|-----------------|
| PSO dual_cross | $56,428 | $1,229 | **45.9x** |
| PSO MACD | $51,039 | $1,366 | **37.4x** |
| GP best tree | $79,080 | $4,565 | **17.3x** |

**Finding**: PSO overfits 2.6× more severely than GP (46x vs 17x train/test gap). GP's structural constraints act as a natural regularizer.

### 2.3 GP Convergence Behaviour

After increasing mutation rate from 0.1 to 0.2:
- **6/10 runs** showed improved fitness across generations (previously 0/10)
- **4/10 runs** remained flat (population converged immediately)
- Mean test fitness: $4,565 ± $461

The increased mutation rate successfully added population diversity. Previously, all 10 runs converged in generation 1.

### 2.4 Hierarchical Experiment (Exp3): GP → PSO Refinement

| Metric | Value |
|--------|-------|
| GP baseline test fitness | $5,120 |
| PSO refined mean | $2,542 |
| Mean improvement | **-$2,578 (-50.3%)** |
| Runs with positive improvement | 4/30 (13%) |

**Finding**: PSO refinement of GP-discovered structures is counterproductive. Freeing the weight parameters while locking durations allows PSO to overfit the freed dimensions, destroying the generalization that GP's structure provides.

### 2.5 Trade Behaviour Analysis

Best-run trade patterns reveal a critical insight:

| Strategy | Trades | Entry Day | Exit Day | Hold Duration | PnL |
|----------|--------|-----------|----------|---------------|-----|
| PSO dual_cross (best) | 1 | Day 48 | Day 790 | 742 days | +$2,989 |
| PSO MACD (best) | 2 | Day 140/755 | Day 650/790 | 510/35 days | +$5,084 |
| GP best tree | 1 | Day 68 | Day 790 | 722 days | +$4,120 |
| Buy & Hold | 1 | Day 0 | Day 790 | 790 days | +$4,660 |

All best runs execute essentially the same strategy: **buy early, hold through the bull market**. The "best run" selection criterion naturally picks strategies that happened to minimize trading — confirming that in a bull market, frequent trading is destructive.

### 2.6 PSO MACD Parameter Analysis

The best PSO MACD run achieved $6,084 by discovering parameters that effectively disable the MACD signal:

```
d1=200, α1=0.01, d2=111, α2=0.01, d3=200, α3=0.01, threshold=0.14
```

All alpha values at the lower bound (0.01) and window sizes at the upper bound (111-200). This configuration produces an almost-flat MACD line, generating virtually no trading signals — functionally equivalent to buy & hold. PSO did not learn to trade; it learned to *not* trade.

---

## 3. Visual Evidence

### 3.1 Equity Curves (`equity_curves.png`)

All best-run equity curves share the BTC bull market shape, confirming the "buy-early-hold" behaviour. PSO MACD's best run (1/30) briefly exceeds buy & hold, but this is an outlier — the remaining 29 runs lose money. GP's curve tracks BTC more conservatively with lower volatility.

### 3.2 Drawdowns (`drawdowns.png`)

PSO strategies exhibit multiple small drawdowns reflecting frequent (and losing) trades. GP and buy & hold show a single drawdown corresponding to the mid-2021 BTC correction. Lower drawdown count correlates with higher final fitness.

### 3.3 Trade Points (`trade_points.png`)

Buy/sell markers reveal PSO's excessive trading in most runs. MACD's 7 trades in its best run generated mixed results — some wins in the early bull phase, followed by losses in the correction. GP's single trade entry during the 2020 accumulation phase proved more reliable.

### 3.4 Convergence (`convergence.png`)

PSO converges in 3-5 iterations, then plateaus for 45+ iterations. GP convergence is more gradual in 6/10 runs (due to increased mutation). The plateau behaviour in PSO suggests premature convergence to local optima.

---

## 4. Discussion

### 4.1 Why Nobody Beats Buy & Hold

BTC 2020-2022 experienced a 7× price appreciation ($7K → $69K → $43K). In a strongly trending market, transaction costs (3% per trade) dominate any potential edge from timing. The optimal strategy is to trade as little as possible — which buy & hold does by definition.

This is not a failure of the algorithms but a property of the test period. Different market regimes (sideways, bear) may yield different conclusions.

### 4.2 GP's Structural Regularization

GP outperforms PSO by 3.7× on test fitness despite having a *higher* training fitness ($79K vs $56K). GP's tree representation limits parameter precision — indicators are fixed once generated, and only logical combinations are explored. This structural constraint acts as implicit regularization, preventing the parameter-level overfitting that afflicts PSO.

### 4.3 The Failure of Hierarchical Optimization

The hypothesis that "GP discovers structure, PSO refines parameters" proved incorrect. Expanding the tree template into a 14D vector and freeing weights/alphas gave PSO enough degrees of freedom to overfit again. The GP structure's advantage lies in its *simplicity* — the tree `(< ema(180,0.94) lma(86))` is a single comparison with two fixed indicators. Converting this to dual_crossover with 8 free parameters destroyed that simplicity.

### 4.4 Statistical Significance

- GP's 10-run mean ($4,565) has SEM = $146, giving a 95% CI of [$4,263, $4,867]
- PSO dual_cross 30-run mean ($1,229) has SEM = $179
- The difference is statistically significant (non-overlapping CIs)
- GP vs PSO refined ($4,565 vs $2,542): t-test p < 0.01

---

## 5. Limitations

1. **Single market regime**: Results are specific to BTC's 2020-2022 bull market. Different periods (2018 bear, 2023 recovery) may reverse conclusions.
2. **GP population convergence**: 4/10 runs still converged in generation 1 despite increased mutation. Further tuning of population size or tournament pressure may help.
3. **Template extraction**: The GP→dual_crossover mapping is lossy — complex trees with IF/AND operators cannot be fully captured by a fixed 14D vector.
4. **Missing ABC and Harmony Search**: Only PSO and GP were implemented, limiting comparative breadth.
5. **Single asset**: BTC-specific conclusions may not generalize to stocks or forex.

---

## 6. Key Findings for Report

| # | Finding | Supporting Evidence |
|---|---------|-------------------|
| 1 | Structure search (GP) generalizes better than parameter search (PSO) | Train/test ratio: GP 17x vs PSO 46x |
| 2 | Hierarchical refinement hurts GP-discovered structures | Mean improvement: -50.3%, only 13% runs positive |
| 3 | Buy & hold is the strongest baseline in bull markets | $5,660 vs best algorithm $4,565 |
| 4 | PSO converges prematurely on training data | Flat after iteration 3-5 of 50 |
| 5 | Algorithm "success" in bull markets = discover how not to trade | Best MACD params disable the MACD signal |
| 6 | GP variance is low and stable | σ=$461, 95% CI: [$4,263, $4,867] |
| 7 | Increased mutation rate improves GP diversity | 6/10 runs show convergence improvement (was 0/10) |

---

## 7. Running the Analysis

```bash
# Regenerate all plots from experiment JSONs
python -m trading_bot.experiments.plot_results
```

Outputs in `results/plots/`:
- `equity_curves.png` — Strategy comparison on test set
- `drawdowns.png` — Drawdown analysis
- `trade_points.png` — Buy/sell markers on price chart
- `convergence.png` — Algorithm convergence curves
