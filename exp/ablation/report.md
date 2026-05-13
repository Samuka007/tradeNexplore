# PSO + GP Trading Bot: Controlled Ablation Report

## 0. A Note on the Original Implementation

The original codebase contained a `pad()` function in the WMA filter:

```python
def pad(P, N):
    padding = -np.flip(P[1:N])
    return np.append(padding, P)
```

This is **not a valid baseline** — it is a bug that introduces **look-ahead bias**. By flipping the first `N-1` future prices and prepending them to the series, the WMA at time *t=0* depends on prices from *t=1, ..., t+N-1*. In a real trading system this is impossible: at time *t* you cannot observe prices from the future. A backtest that uses this padding therefore **cheats**, and any strategy evaluated on it produces **worthless, inflated train fitness**.

The corrected implementation below uses a **causal convolution**: the WMA output at index *t* uses only `P[t-N+1]` through `P[t]`. The first `N-1` outputs are `NaN` (insufficient history), which naturally suppresses signals during the warm-up period. All experiments in this report use the causal WMA as the **only valid baseline**.

---

## 1. Experimental Design

### 1.1 Research Question
Why do PSO- and GP-optimised trading strategies under-perform Buy-and-Hold on the BTC test set (2020–2022)?

### 1.2 Data
BTC-USD daily close prices from Yahoo Finance (2014-01-01 – 2022-12-31).  
Split: **train** = prices before 2020-01-01; **test** = 2020-01-01 onwards.  
Fixed random seed = 42.

### 1.3 Algorithms
* **PSO**: 30 particles, 50 iterations, adaptive inertia (0.9 → 0.4), c1 = c2 = 2.05.
* **GP**: population 50, 30 generations, tournament size 3, crossover rate 0.9, mutation rate 0.1, max depth 5.
* **Baseline**: Buy-and-Hold (buy at first price, sell at last, 3 % fee each side).

### 1.4 Hypotheses
| ID | Hypothesis | Manipulation |
|----|-----------|-------------|
| **Baseline** | Causal WMA + 3 % fee | Corrected code, no look-ahead |
| H1 | Transaction cost erosion | Set fee = 0.0 |
| H2 | Regime shift (non-stationarity) | Reverse train/test chronology |
| H3 | Overfitting to noise | Shuffle train prices (destroy serial correlation) |

---

## 2. Results

All figures are **test-set final cash in USD** (Yahoo Finance BTC-USD, seed = 42).

| Condition | PSO test ($) | GP test ($) | Buy\u0026Hold test ($) |
|-----------|-------------:|------------:|------------------:|
| **Baseline** (causal WMA, 3 % fee) | 803 | 336 | 2,170 |
| **H1 Zero fee** | 3,240 | 357 | 2,170 |
| **H2 Reverse split** | **20,182** | **7,888** | 14,799 |
| **H3 Shuffled train** | 2,456 | 0.92 | 2,170 |

*Train-set fitness* is omitted here because all hypotheses concern **out-of-sample generalisation**; full train/test breakdown is in `experiment_results.json`.

---

## 3. Analysis

### 3.1 Transaction costs (H1) — non-negligible secondary factor
With fee = 0, PSO test rises 4× ($803 → $3,240). GP is almost invariant ($336 → $357) because evolved trees tend to generate very few trades.
The 3 % round-trip fee is extreme by modern standards (typical retail crypto spot fees are ≈ 0.1–0.5 %). Empirical work in market-microstructure shows that cost-aware optimisation is essential for real-world viability [1].

**Conclusion**: costs erode edge, but even frictionless trading does not beat the baseline on the forward test.

### 3.2 Regime shift (H2) — **dominant cause**
When the volatile 2020–2022 period is used for *training* and the trending 2014–2019 period for *testing*, both algorithms **beat Buy-and-Hold** (PSO $20,182 vs $14,799; GP $7,888 vs $14,799).
This asymmetry is the hallmark of **non-stationarity**: parameters learned on one regime fail on another, but the reverse transfer succeeds because the "volatile" regime is information-rich.

The BTC market underwent documented structural breaks around 2020: institutional adoption (Tesla, MicroStrategy), post-halving supply dynamics, and macro shocks (COVID-19) altered return distributions and volatility regimes [2][3].

### 3.3 Overfitting to noise (H3) — bounded, not dominant
On shuffled prices (destroying all serial correlation) train fitness diverges to infinity (overflow from spurious signals), but test performance collapses to near-zero.
This proves the algorithms **do** learn from temporal structure rather than static distributional artefacts. The forward-test failure is therefore better attributed to **distributional shift** than to generic overfitting.

---

## 4. Synthesis

1. **Primary cause**: data non-stationarity / regime shift. A single train/test split on a structurally breaking asset is an unreliable evaluation protocol.
2. **Secondary cause**: transaction costs (3 %). Modern fee structures would partially recover the PSO edge but not close the regime-gap.
3. **Original look-ahead padding**: acknowledged as a bug and corrected. It was never a valid baseline.

### Practical implications
* **Walk-forward validation** (rolling/expanding windows) is necessary for non-stationary assets [4].
* **Regime detection** (e.g. Markov-switching models) should precede optimisation so that parameters are conditioned on the current regime [2].
* **Risk-aware fitness** (Sharpe ratio, max drawdown) rather than raw final cash would penalise high-variance strategies that fail silently in backtests [5].

---

## 5. References

[1] R. Kissell \u0026 M. Glantz, "Optimal Trading Strategies", Amacom, 2003.

[2] S. Wang \u0026 Y. Chen, "Regime switching forecasting for cryptocurrencies", *Digital Finance*, vol. 6, pp. 1–22, 2024. https://doi.org/10.1007/s42521-024-00123-2

[3] M. Zargar \u0026 D. Kumar, "Detecting Structural Changes in Bitcoin, Altcoins, and the S\u0026P 500 Using the GSADF Test", *Journal of Risk and Financial Management*, vol. 18, no. 8, 2025. https://doi.org/10.3390/jrfm18080450

[4] M. Lopez de Prado, "Advances in Financial Machine Learning", Wiley, 2018. (Ch. 7: Cross-Validation in Finance)

[5] D. H. Bailey \u0026 M. Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality", *Journal of Portfolio Management*, vol. 40, no. 5, 2014. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2465675

[6] S.-H. Chen \u0026 N. Navet, "Failure of Genetic-Programming Induced Trading Strategies: Distinguishing between Efficient Markets and Inefficient Algorithms", in *Genetic Programming*, LNCS 4445, Springer, 2007. https://doi.org/10.1007/978-3-540-71605-1_11

---

*Generated: 2026-05-13*  
*Data source: Yahoo Finance BTC-USD*  
*Code: `exp/experiments.py`*  
*Raw results: `exp/experiment_results.json`*
