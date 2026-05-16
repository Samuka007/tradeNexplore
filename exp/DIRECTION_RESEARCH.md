# Direction Research — Eight Candidate Narratives for CITS4404 Part 2

Each direction was investigated via web search for theoretical grounding
and evaluated against the BTC trading problem, existing data, and
feasibility constraints.

Ratings: Narrative Strength (★1-5) | Feasibility (◆1-5)

---

## A. The Neutrality Hypothesis

**Core claim**: GP's syntactic space is vastly larger than its semantic
space. Many different trees produce identical trading signals. This
"neutrality" means GP is searching a much smaller _effective_ space than
it appears to be.

**Theoretical grounding**:
- Langdon & Poli, _Foundations of Genetic Programming_ (2002): GP
  landscapes are highly neutral — many genotypes map to the same
  phenotype.
- Vanneschi et al. (2011): "A study of the neutrality of Boolean
  function landscapes in GP" — neutrality can help drift to good
  regions but also wastes evaluations.
- Hu (2016): "Neutrality, Robustness, and Evolvability in GP" —
  neutrality is a double-edged sword.

**Application to BTC trading**:
Two trees like `(> sma(20) sma(50))` and `(< sma(50) sma(20))` are
syntactically different but semantically equivalent (just negated). The
GP population may contain dozens of such neutral variants, all consuming
evaluations but contributing no new information.

**Proposed experiments**:
1. **E-A1**: During a GP run (λ=500, 10 seeds), hash the signal vector
   of every individual. Compute the "unique-signal ratio" (unique
   signals / total evaluations) across generations.
2. **E-A2**: Compare unique-signal ratio between λ=0 (unrestricted) and
   λ=500 (regularized). Hypothesis: λ=500 prunes neutral variants.
3. **E-A3**: Measure whether neutral drift correlates with
   out-of-sample performance. Does exploring many neutral variants
   help escape local optima, or is it pure waste?

**Resource estimate**: E-A1 requires modifying GP.optimize() to log
signals. ~10 seeds × 1,500 evals = 15K evals. Analysis is O(n²) signal
comparison but signals are short vectors (~2000 points).

**Narrative strength**: ★★★☆☆
Neutrality is a real phenomenon but it's somewhat technical for a course
report. The insight — "GP wastes evaluations on semantically identical
individuals" — is valuable but may not resonate as a "big story."

**Feasibility**: ◆◆◆◆◆
Requires only signal hashing and counting. Existing data (Exp 17) could
be re-analyzed if signals were saved.

---

## B. The Epistasis Hypothesis

**Core claim**: In GP, crossover is destructive because subexpression
semantics depend on context (epistasis). A subtree that works well in
one parent may work poorly in another. PSO avoids this because
parameters are largely independent. This structural difference explains
PSO's stability.

**Theoretical grounding**:
- Krawiec (2013): "Locally geometric semantic crossover: a study on the
  roles of semantics and homology in recombination operators" — context
  determines subtree value.
- Spector & Klein (2006): "Semantic Building Blocks in Genetic
  Programming" — semantics are context-dependent, not compositional.
- O'Reilly & Oppacher (1995): "The Troubling Aspects of a Building
  Block Hypothesis for GP" — GP crossover does NOT preserve building
  blocks.

**Application to BTC trading**:
Consider a subtree `(> sma(20) price)`. In a parent where this evaluates
to a buy signal early in the series, it may be profitable. But in
another parent with different surrounding structure, the same subtree
may generate noise trades. Crossover extracts the subtree without its
original context, destroying its value.

**Proposed experiments**:
1. **E-B1**: Measure parent-offspring fitness correlation after GP
   crossover. Sample 100 crossover events per generation. Compute
   Pearson r between parent mean fitness and offspring fitness.
   Hypothesis: r < 0.3 (weak correlation = high epistasis).
2. **E-B2**: Compare with PSO: offspring fitness is highly correlated
   with parent fitness (velocity update is a smooth perturbation).
   Compute parent-offspring correlation for PSO particles.
3. **E-B3**: Test "context-aware crossover" (Krawiec 2006): only
   swap subtrees with similar semantic context. Does this reduce
   epistasis and improve GP stability?

**Resource estimate**: E-B1 requires instrumenting GP._crossover().
~10 seeds × 1,500 evals = 15K evals. Analysis is post-hoc.

**Narrative strength**: ★★★★☆
This directly explains WHY GP is unstable and PSO is stable — a
mechanistic explanation, not just an observation. It's a strong
algorithmic insight.

**Feasibility**: ◆◆◆◆◆
Instrumentation is simple. Existing data can be re-analyzed if parent
and offspring were logged.

---

## C. The Regime-Contingent Performance Hypothesis

**Core claim**: Different market regimes (bull/bear/sideways) favor
different strategy structures. GP can adapt structure to regime because
it searches over structures; PSO is locked into a fixed template
(position_sma). Walk-forward validation fails because the optimal
structure changes across regimes, but PSO can't change structure — only
parameters.

**Theoretical grounding**:
- Standard finance: momentum strategies work in trending markets,
  mean-reversion in ranging markets.
- Hu et al. (2025): "Intelligent trading strategy based on improved
  directional change and regime change detection" — regime-aware
  strategies outperform static ones.
- The walk-forward collapse of both PSO (29% drop) and GP (20% win
  rate) suggests neither algorithm handles regime shifts well, but for
  different reasons.

**Application to BTC trading**:
BTC 2014-2022 spans: bull (2015-2017), bear (2018-2022), and sideways
(2019). The PSO position_sma template (sigmoid of SMA difference) is
fundamentally a trend-following strategy. It works in bull markets but
fails in bear/sideways. GP can discover non-trend strategies (e.g.,
mean-reversion with `IF` conditions) but doesn't because selection
favors the highest train fitness, which is dominated by the bull period.

**Proposed experiments**:
1. **E-C1**: Classify BTC into 3 regimes using simple thresholds
   (drawdown > 30% = bear, gain > 50% = bull, else sideways).
2. **E-C2**: Run GP separately on each regime's training data. Do the
   winning trees have different structures? (Bull → simple SMA cross;
   Bear → conditional logic; Sideways → volatility-based)
3. **E-C3**: Compare PSO walk-forward (fixed template, varying params)
   vs GP walk-forward (varying structure AND params). Does GP's
   structural flexibility provide any advantage across regime shifts?

**Resource estimate**: E-C2 requires 3 × 10 seeds = 30 GP runs.
~45K evals. E-C3 requires re-running walk-forward with GP (Exp 19
exists but is single-seed).

**Narrative strength**: ★★★★★
This directly answers "why does walk-forward fail?" with a structural
explanation. It connects algorithm choice to market microstructure — a
sophisticated insight for a course report.

**Feasibility**: ◆◆◆◆◆
Regime classification is trivial. GP per-regime is straightforward.
Walk-forward GP needs multi-seed replication.

---

## D. The Structural Bias Hypothesis

**Core claim**: Every representation imposes structural bias. PSO's
bias is explicit: fixed template. GP's bias is implicit: the function
set. Neither is "unbiased." The question is not which has less bias but
which bias aligns with the true data-generating process.

**Theoretical grounding**:
- Krawiec & Swan (2013): "Guiding Evolutionary Learning by Searching
  for Regularities in Behavioral Trajectories: A Case for
  Representation Agnosticism" — all representations impose bias.
- Morgan & Gallagher (2023): "Uncovering structural bias in
  population-based optimization algorithms" — the Generalized Signature
  Test measures algorithmic bias independently of the problem.
- NFL theorem (Wolpert & Macready 1997): bias is not a bug but a
  necessity for performance.

**Application to BTC trading**:
PSO's position_sma template is biased toward smooth, continuous,
trend-following strategies. GP's extended function set (with `IF`,
`AND`, `MAX`, `MIN`, `ABS`, `RSI`, `volatility`, `momentum`) is biased
toward discontinuous, conditional, multi-indicator strategies. If the
"true" BTC strategy is trend-following, PSO's bias is advantageous. If
the true strategy is regime-dependent, GP's bias is advantageous. But
we don't know the true strategy — and the signal is weak, so neither
bias strongly aligns.

**Proposed experiments**:
1. **E-D1**: Restrict GP to only functions that PSO can express
   (`>`, `<`, `sma`, `lma`, `ema`). Run 10 seeds. Does GP match PSO's
   performance? If yes, GP's function-set richness is NOT the source
   of its (few) wins.
2. **E-D2**: Run the Generalized Signature Test (Gallagher 2023) on
   both PSO and GP to characterize their inherent biases independently
   of the trading problem.
3. **E-D3**: Compare GP with minimal vs standard vs extended function
   sets, but with MULTI-SEED (10 seeds each). Is there a systematic
   performance difference, or is the single-seed result (Exp 12)
   noise?

**Resource estimate**: E-D1: 10 seeds × 1,500 = 15K evals.
E-D3: 3 function sets × 10 seeds × 1,500 = 45K evals.

**Narrative strength**: ★★★★★
This reframes the entire comparison from "which algorithm is better" to
"which bias matches the problem." It directly invokes NFL and
representation theory — exactly what the course spec asks for.

**Feasibility**: ◆◆◆◆◆
E-D1 and E-D3 are straightforward code modifications. The Generalized
Signature Test (E-D2) may be too complex for a course project.

---

## E. The Regularization Landscape Hypothesis

**Core claim**: Different regularization dimensions (parsimony
pressure, function set restriction, depth limits) control different
aspects of the hypothesis space. Parsimony controls tree size; function
set controls expressiveness; depth controls nesting complexity. These
are NOT interchangeable — each prunes a different subspace.

**Theoretical grounding**:
- Poli (2013): "Parsimony Pressure Made Easy: Solving the Problem of
  Bloat in GP" — parsimony pressure is the most studied form of
  regularization.
- Vanneschi et al. (2010): "Measuring bloat, overfitting and
  functional complexity in genetic programming" — structural metrics
  correlate with generalization.
- Luke (2002): "Code Growth in Genetic Programming" — multiple
  mechanisms (depth limits, size limits, parsimony) have different
  effects on search dynamics.

**Application to BTC trading**:
Exp 17 shows λ=500 is the sweet spot. But λ=500 doesn't tell the whole
story: it penalizes tree size, not depth or function complexity. A
deeply nested tree with few nodes might still overfit. Conversely, a
shallow tree with many nodes might generalize. The "regularization
landscape" is multi-dimensional.

**Proposed experiments**:
1. **E-E1**: 3×3 grid: λ ∈ {0, 500, 1000} × depth limit ∈ {3, 5, 7},
  10 seeds each = 90 runs. Measure test performance AND structural
  metrics (depth, nesting ratio, constant ratio).
2. **E-E2**: Correlation analysis: which structural metric best
  predicts train-test gap? (tree size? depth? IF frequency? constant
  ratio?)
3. **E-E3**: Test whether depth limit alone (no λ) can prevent
  overfitting. Run depth=3, 5, 7 with λ=0. Does depth limit substitute
  for parsimony pressure?

**Resource estimate**: E-E1: 90 runs × 1,500 = 135K evals.
This is substantial but manageable.

**Narrative strength**: ★★★☆☆
Interesting but somewhat technical. The insight — "regularization is
multi-dimensional" — is correct but may not be the central story of the
report.

**Feasibility**: ◆◆◆◆◆
Straightforward modifications to existing code.

---

## F. The Overfitting Geometry Hypothesis

**Core claim**: Train-test divergence in GP follows specific structural
patterns. Overfitting trees tend to have: deep nesting, many constants,
specific function combinations (`IF`/`AND`), or specific terminal types.
We can predict overfitting from tree structure WITHOUT testing.

**Theoretical grounding**:
- Vanneschi et al. (2010): structural complexity metrics correlate
  with overfitting.
- Castelli et al. (2015): "Model Selection and Overfitting in Genetic
  Programming: Empirical Study" — tree size alone is insufficient;
  functional complexity matters.
- Burlacu et al. (2019): multi-objective GP with structural complexity
  measures.

**Application to BTC trading**:
Exp 20 shows λ=0 trees grow from 6 to 30 nodes while test collapses to
$1,000. But size isn't the only factor. The 96-node tree in Exp 09
(`AND price (AND (> ...) (/ ...))`) has deep nesting and many
constants. The 3-node tree `(> volatility(20) rsi(49))` generalizes
well despite being tiny. Can we predict which trees will overfit from
structure alone?

**Proposed experiments**:
1. **E-F1**: Define structural metrics: depth, nesting ratio,
   constant ratio, IF frequency, unique terminal count. Compute for
   all trees in Exp 17 (42 trees with known train/test values).
2. **E-F2**: Train a simple regression (or just correlation) from
   structural metrics to train-test gap. Which metrics are predictive?
3. **E-F3**: Test "structural early stopping": stop GP when tree
   structure crosses an overfitting threshold (e.g., nesting ratio >
   0.7). Does this improve test performance vs generation-based
   stopping?

**Resource estimate**: E-F1 is pure analysis of existing data (0 runs).
E-F3: ~10 seeds × 1,500 = 15K evals.

**Narrative strength**: ★★★★☆
"Predicting overfitting from tree shape" is a concrete, testable claim
with practical implications. It's a strong secondary finding.

**Feasibility**: ◆◆◆◆◆
E-F1 is free (existing data). E-F3 is straightforward.

---

## G. The Algorithm-Problem Alignment Hypothesis (NFL)

**Core claim**: No Free Lunch says no algorithm is universally better.
But we can ask: what problem characteristics make GP align well vs
PSO? On the BTC trading task, the "true" strategy may be simple (2-3
parameters) OR complex (conditional logic). If simple, PSO wins; if
complex, GP wins. The signal is weak, so neither wins consistently.
This leads to a meta-question: can we diagnose FROM THE DATA whether
the problem is "parametric" or "structural"?

**Theoretical grounding**:
- Wolpert & Macready (1997): "No Free Lunch Theorems for
  Optimization" — all algorithms are equivalent averaged over all
  problems; advantage comes from problem-specific alignment.
- Morgan & Gallagher (2023): "Uncovering structural bias" — measure
  algorithm bias independently.
- The "alignment" concept: an algorithm's performance advantage is
  proportional to how well its bias matches the problem's structure.

**Application to BTC trading**:
If we could measure "problem compressibility" (can a simple linear
model explain most variance?), we could predict whether PSO or GP
should win. If BTC returns are mostly noise (low compressibility),
neither algorithm should win consistently — which matches the data
(GP mean < BH, PSO mean > BH but walk-forward collapses).

**Proposed experiments**:
1. **E-G1**: Measure problem compressibility: fit a linear model to
   BTC returns using lagged prices. How much variance is explained?
   (If R² < 0.05, the signal is genuinely weak.)
2. **E-G2**: Measure whether the "optimal" strategy is likely
   parametric or structural by testing if a constrained GP
   (SMA-only, depth=3) matches or exceeds unrestricted GP.
3. **E-G3**: Run a "diagnostic suite": test multiple simple models
   (linear, SMA cross, EMA cross) on BTC. If simple models perform
   near BH, the problem is not structurally complex. If they perform
   far below BH, the problem requires structural search.

**Resource estimate**: E-G1 is a simple linear regression (instant).
E-G2: 10 seeds × 1,500 = 15K evals. E-G3: multiple simple models
(instant).

**Narrative strength**: ★★★★★
This is the most philosophically sophisticated direction. It elevates
the paper from "we compared two algorithms" to "we diagnosed whether
the problem rewards parametric or structural search." Directly invokes
NFL — a famous theorem the course would recognize.

**Feasibility**: ◆◆◆◆◆
E-G1 and E-G3 are instantaneous. E-G2 is a standard GP run with
modified function set.

---

## H. The Semantic Diversity Hypothesis

**Core claim**: GP's high variance isn't purely a bug — it's a feature
of semantic diversity. PSO converges to the same basin because the
parametric landscape has few semantic attractors. GP explores many
semantic regions because the tree space has many more. The question is
whether semantic diversity correlates with out-of-sample performance.

**Theoretical grounding**:
- Krawiec (2013): "Semantic variation operators for multidimensional
  genetic programming" — semantic diversity is a search asset.
- Szubert et al. (2016): "Reducing Antagonism between Behavioral
  Diversity and Fitness in Semantic GP" — diversity and fitness are
  often in tension.
- Forstenlechner et al. (2015): "Introducing Semantic-Clustering
  Selection in Grammatical Evolution" — clustering by behavior
  improves search.

**Application to BTC trading**:
In a GP run with λ=500, the population might contain: (a) SMA
strategies, (b) RSI strategies, (c) momentum strategies, (d)
volatility strategies. This diversity means some individuals might
perform well in bear markets while others perform in bull markets. But
if selection favors the highest single-period fitness, diversity
collapses. PSO has no such diversity: all particles are in the same
parametric basin.

**Proposed experiments**:
1. **E-H1**: Cluster GP population by signal correlation at each
   generation. Measure number of clusters (semantic diversity) over
time. Does diversity collapse as fitness converges?
2. **E-H2**: Compare λ=0 vs λ=500: does λ=500 maintain higher
diversity because smaller trees have fewer semantic possibilities?
   (Counterintuitive but testable.)
3. **E-H3**: Test whether higher-diversity populations generalize
   better: run GP with a diversity-promoting selection (e.g.,
   semantic-clustering selection) vs standard tournament selection.

**Resource estimate**: E-H1 requires logging all population signals.
~10 seeds × 1,500 evals = 15K evals. Analysis: clustering on signal
vectors.

**Narrative strength**: ★★★☆☆
Semantic diversity is a rich concept but somewhat abstract for a course
report. The tension between "diversity helps exploration" and
"diversity means less exploitation" is interesting but may confuse
readers.

**Feasibility**: ◆◆◆◆◆
Instrumentation is moderate (signal logging). Clustering is standard.

---

# Comparative Summary

| Dir | Narrative | Theoretical Depth | Novelty | Exp Cost | Overall |
|-----|-----------|-------------------|---------|----------|---------|
| A   | Neutrality         | ★★★★☆ | ★★★☆☆ | Low  | Solid but narrow |
| B   | Epistasis          | ★★★★★ | ★★★★☆ | Low  | **Strong** |
| C   | Regime-Contingent  | ★★★★☆ | ★★★★★ | Med  | **Very Strong** |
| D   | Structural Bias    | ★★★★★ | ★★★★★ | Low  | **Very Strong** |
| E   | Regularization     | ★★★☆☆ | ★★★★☆ | High | Good but heavy |
| F   | Overfitting Geom   | ★★★★☆ | ★★★★☆ | Low  | **Strong** |
| G   | NFL Alignment      | ★★★★★ | ★★★★★ | Low  | **Very Strong** |
| H   | Semantic Diversity | ★★★★☆ | ★★★☆☆ | Med  | Solid but abstract |

---

# Recommended Combinations

## Option 1: The Deep Insight (G + D + B)
**Narrative**: "NFL says no algorithm is universally better. We diagnose
whether the BTC problem is parametric or structural (G), test whether
each algorithm's bias matches (D), and explain the stability difference
via epistasis (B)."

- **G** provides the philosophical framing.
- **D** provides the empirical test (restrict GP to PSO's function
  family).
- **B** provides the mechanistic explanation.

**New experiments needed**: E-G1 (instant), E-G2 (15K evals), E-D1
(15K evals), E-B1 (15K evals). Total: ~45K evals.

## Option 2: The Market Story (C + F + G)
**Narrative**: "Walk-forward fails because market regimes change, and
PSO's fixed template can't adapt. We diagnose regime-specific strategy
structures (C), show that tree shape predicts overfitting (F), and use
NFL to explain why no single algorithm works across regimes (G)."

- **C** provides the market-structure hook.
- **F** provides the practical insight.
- **G** provides the theoretical framing.

**New experiments needed**: E-C2 (45K evals), E-F1 (free), E-F3 (15K
evals), E-G1 (instant). Total: ~60K evals.

## Option 3: The Algorithmic Mechanism (B + D + F)
**Narrative**: "GP is unstable not because trees are bad but because
crossover is destructive (epistasis, B). PSO is stable because it
avoids structural bias — but that same stability is a cage (D). We can
predict overfitting from tree geometry (F), making GP controllable."

- **B** explains the instability.
- **D** explains the trade-off.
- **F** provides the control mechanism.

**New experiments needed**: E-B1 (15K), E-D1 (15K), E-D3 (45K), E-F1
(free), E-F3 (15K). Total: ~90K evals.

---

**All three options avoid the pitfalls identified in PITFALLS.md:**
- No single-seed claims
- No apples-to-oranges comparisons
- No "algorithm X wins" framing
- Honest about limitations
