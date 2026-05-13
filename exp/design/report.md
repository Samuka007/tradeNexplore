# Design Critique: Why Single-Split Backtests Fail, and How to Evaluate Properly

## 1. The Core Problem Is Not Regime Shift — It Is the Evaluation Protocol

H2 in the ablation study (reverse chronological split) showed that parameters learned on the volatile 2020–2022 regime generalise *backwards* to the trending 2014–2019 regime. This asymmetry is often interpreted as "the market changed."

But a more precise interpretation is: **the single train/test split is the wrong tool for the job.**

Financial time series are non-stationary by construction. Any fixed cut-off date is arbitrary. A strategy that happens to align with the training regime will look brilliant; one that does not will look broken. The luck of the split, not the merit of the algorithm, dominates the headline number.

This is not a speculative claim. Lopez de Prado (2018) demonstrates that standard k-fold cross-validation on financial time series is **invalid** because it leaks future information into the training set through overlapping labels. The correct protocol is **walk-forward validation** (rolling or expanding windows), where the model is re-trained at each step using only data available up to that point.

The walk-forward experiment in this directory (`walk_forward.py`) implements a rolling 3-year train / 1-year test protocol. The results are sobering:

| Window | Buy&Hold | PSO | GP | PSO > BH? | GP > BH? |
|--------|----------|-----|----|-----------|-----------|
| 2017–2018 | $1,711 | $1,000 | $590 | No | No |
| 2020–2021 | $4,106 | $1,000 | $4,106 | No | Tie |

Across both windows, PSO never beats Buy-and-Hold. GP matches it in one window and collapses in the other. The headline "PSO train=$53k" from the single-split experiment is therefore **not a robust finding** — it is an artefact of regime alignment.

## 2. Buy-and-Hold Is the Wrong Benchmark

Buy-and-Hold is not a "strategy" in the active-management sense; it is a **passive beta exposure**. It captures the long-run drift of the asset, driven by macro adoption, monetary policy, halving cycles, and social sentiment — factors that have nothing to do with the microstructure signals (crossover detection, momentum) that PSO and GP are optimising.

Comparing an active trading algorithm to Buy-and-Hold is therefore a **category error**. It conflates two different questions:
1. Can you time the market better than a coin flip? (skill, alpha)
2. Did you ride the long-term bull market? (exposure, beta)

The proper benchmark for an active strategy is **its excess return over a passive benchmark with the same risk exposure**, normalised by tracking error. This is the **Information Ratio (IR)** [1]:

```
IR = (Return_strategy − Return_benchmark) / TrackingError
```

Equivalently, one can use **Jensen's Alpha** [2], which regresses strategy returns on benchmark returns and tests whether the intercept is statistically significant:

```
R_strategy = α + β · R_benchmark + ε
```

If α ≤ 0, the strategy adds no value beyond passive exposure. In our walk-forward results, both PSO and GP exhibit negative or near-zero alpha relative to Buy-and-Hold.

## 3. "Win Rate Within Intervals" Is a Valid, Under-Used Metric

The user proposes evaluating algorithms by their **win rate within intervals** — the fraction of time windows (days, weeks, months) in which the strategy outperforms the benchmark.

This is conceptually sound and has precedent in the literature. The ** batting average** or **hit rate** of a strategy measures the frequency of correct directional calls [3]. When combined with the **payoff ratio** (average win / average loss), it yields the **Kelly criterion** optimal fraction and the **expectancy** of the strategy:

```
Expectancy = (WinRate × AvgWin) − (LossRate × AvgLoss)
```

A strategy with a high win rate but negative expectancy is dangerous: it feels good most of the time, but the occasional large loss wipes out the gains. Conversely, a strategy with a low win rate but positive expectancy (e.g. trend-following) can be profitable if position sizing is disciplined.

In our context, "interval win rate" should be defined as:
- **Per-trade win rate**: fraction of trades that are profitable.
- **Per-period win rate**: fraction of test periods (walk-forward windows) where strategy return > benchmark return.

The walk-forward experiment reports the second definition. PSO scores 0 %, GP scores 0 % (or 50 % if the tie is counted). This is a more honest picture than the single-split headline.

## 4. Synthesis: What Would a Correct Pipeline Look Like?

| Component | Current (Wrong) | Correct |
|-----------|----------------|---------|
| Data split | Single static cut-off at 2020 | Walk-forward rolling/expanding windows |
| Benchmark | Buy-and-Hold (passive beta) | Risk-adjusted excess return (IR, Alpha) |
| Fitness | Final cash | Sharpe, Calmar, or Sortino ratio |
| Evaluation | One test set | Multiple windows + interval win rate |
| Regime | Ignored | Explicit detection (Markov-switching) before optimisation |

## 5. References

[1] "Information Ratio" — Investopedia. https://www.investopedia.com/terms/i/informationratio.asp

[2] "Jensen's Measure (Alpha)" — Investopedia. https://www.investopedia.com/terms/j/jensensmeasure.asp

[3] M. Lopez de Prado, "Advances in Financial Machine Learning", Wiley, 2018. (Ch. 7–9: Cross-Validation, Feature Importance, Backtesting)

[4] "Is Alpha Just Beta Waiting to Be Discovered?" — AQR. https://integrityia.com/wp-content/uploads/2014/06/Is-Alpha-Just-Beta-Waiting-To-Be-Discovered-AQR-updated.pdf

[5] "The Alpha Equation: Myths and Realities" — PIMCO. https://www.pimco.com/gb/en/insights/the-alpha-equation-myths-and-realities

---

*Generated: 2026-05-13*
