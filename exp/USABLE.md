# USABLE — Boundary Conditions for Existing Experiments

Every experiment listed here has been audited. If an experiment is not
listed, it is either archived (exp/archive/) or carries a CAUTION.md
marker. No experiment may be cited in the paper unless it appears in
this file with verified boundary conditions.

---

## A-Grade: Core Evidence

### Exp 17 — Systematic Hyperparameter Grid
- **Runs**: 42 (3 seeds × 7 λ × 2 depth)
- **Boundary**: λ ∈ [100, 5000], depth ∈ {5, 7}, extended function set,
  single split (train < 2020, test ≥ 2020), seed ∈ {0, 42, 88}
- **buy_and_hold**: $2,169.58
- **Usable for**: λ sweet-spot identification; depth comparison at fixed
  budget (1,500 evals)
- **Verified facts**: λ = 500 is the sweet spot (mean test highest across
  seeds); depth 5 slightly outperforms depth 7 at λ = 500
- **Limitations**: Only 3 seeds; seed 88 is an outlier at λ = 500,
  depth 7 ($3,142). Do not generalize variance estimates.
- **Not usable for**: Cross-seed population claims (insufficient N);
  claims about other function sets or depths.

### Exp 15 — Warm-Start vs Random Initialization
- **Runs**: 20 (10 seeds × 2 conditions)
- **Boundary**: 10 seeds, λ = 0, depth = 5, extended function set,
  single split
- **buy_and_hold**: $2,169.58
- **Usable for**: Initialization effect on GP variance; demonstration of
  common-mode failure
- **Verified facts**: Random: μ = $1,689, σ = $677, 2/10 beat BH.
  Warm-start: μ = $2,362, σ = $894, 6/10 beat BH.
- **Critical caveat**: Warm-start uses 20 % human rules (10 / 50
  individuals), NOT 50 % as previously claimed. The prior claim of
  "50 % human rules" was wrong.
- **Limitations**: λ = 0 means no regularization; results reflect
  unregularized GP behavior only.
- **Not usable for**: Absolute performance under regularized conditions;
  claims about warm-start diversity (must compute tree overlap
  separately).

### Exp 18 — GA on 3D Parametric Space
- **Runs**: 10 seeds
- **Boundary**: position_sma representation [fast, slow, scale],
  single split, GA (tournament selection + uniform crossover + Gaussian
  mutation), NOT GP
- **buy_and_hold**: $2,169.58
- **Usable for**: Same-representation algorithm comparison; basin
  convergence reliability
- **Verified facts**: μ = $2,220, σ = $230, 5/10 beat BH. Converges to
  same two basins as PSO.
- **Critical caveat**: Must be labeled as GA, not "restricted GP" or
  "GP restricted". Individuals are single-node parametric vectors.
- **Limitations**: scale is NOT fixed at 0.1; GA finds scale values
  between 0.1 and 16.2. The parameter space is the same as PSO's, not
  the specific optimum.
- **Not usable for**: GP-vs-PSO claims (algorithm changed to GA).

### Exp 19 — Landscape Grid
- **Points**: 171 (fast × slow with scale = 0.1)
- **Boundary**: 2D grid over (fast, slow), scale fixed at 0.1,
  single split
- **buy_and_hold**: $2,169.58
- **Usable for**: 2D basin visualization; geometry of parametric
  landscape
- **Verified facts**: Basin A ≈ (120, 180), Basin B ≈ (40, 100).
  Landscape is smooth with low ridge between basins.
- **Critical caveat**: Point (190, 200) has test = $4,038 with
  train = $7,487. This is likely a market-event artifact, not a genuine
  basin. Do not cite it as a discovered optimum.
- **Not usable for**: Higher-dimensional landscape claims; claims about
  scale sensitivity (scale is fixed).

### PSO Convergence (analysis/pso_convergence/)
- **Runs**: 5 seeds
- **Boundary**: position_sma, 30 particles × 50 iterations, single split
- **buy_and_hold**: $2,169.58
- **Usable for**: Basin convergence pattern; PSO stability demonstration
- **Verified facts**: 3 / 5 seeds converge to Basin A ($2,366 test);
  1 / 5 to Basin B ($2,264); seed 99 stuck at suboptimal basin
  ($2,265). Convergence typically within 15 iterations.
- **Limitations**: 5 seeds only.
- **Not usable for**: Population-level statistical claims; claims about
  other PSO variants.

### Seed Robustness (analysis/seed_robustness/)
- **Runs**: 10 seeds each
- **Boundary**: GP with λ = 500, depth = 5; PSO position_sma;
  single split
- **buy_and_hold**: $2,169.58
- **Usable for**: Variance comparison under controlled conditions
- **Verified facts**: PSO: μ = $2,297, σ = $77, 10 / 10 beat BH.
  GP (λ = 500): μ = $1,594, σ = $432, 0 / 10 beat BH.
- **Limitations**: GP condition (λ = 500) differs from Exp 15
  (λ = 0). Direct mean comparison between GP and PSO is still
  apples-to-oranges (different representations).
- **Not usable for**: Cross-algorithm mean comparison without noting
  representation difference.

---

## B-Grade: Auxiliary Evidence (Use with Caveats)

### Exp 09 — Parsimony Pressure
- **Runs**: 15 (single seed 42, 5 λ × 3 depth)
- **Boundary**: seed = 42, λ ∈ {0, 100, 500, 1000, 5000}, depth ∈ {3, 5, 7}
- **Usable for**: Qualitative bloat demonstration (λ = 0 → 96 nodes)
- **Limitations**: Superseded by Exp 17 for quantitative claims.
  Single-seed λ sweet spot (λ = 1000) was wrong; Exp 17 with 3 seeds
  shows λ = 500 is correct.
- **Not usable for**: λ sweet spot claims; quantitative performance
  claims.

### Exp 20 — Trajectory
- **Runs**: 2 (seed = 42, λ = 0 vs λ = 500)
- **Boundary**: seed = 42, 20 generations
- **Usable for**: Qualitative train-test divergence visualization
- **Verified observation**: λ = 0: train rises $20,674 → $38,003 while
  test collapses to $1,000 at gen 11, tree grows 6 → 30 nodes.
  λ = 500: stabilizes at gen 3, train $25,019, test $2,245, tree 3 nodes.
- **Limitations**: Single seed; not reproducible.
- **Not usable for**: Quantitative claims about typical GP behavior.

### Hybrid Supplementary (exp/hybrid-supplementary/)
- **Runs**: 7 trees (sizes 1–10)
- **Boundary**: Fixed trees from prior GP runs, PSO refinement on leaf
  parameters, single seed
- **Usable for**: Hypothesis generation about non-monotonic tree-size
  interaction
- **Verified observation**: Overfit trees (size 5, 9) collapse −85 %
  under PSO refinement; larger signal tree (size 10) gains +10 %.
- **Limitations**: 7 samples, no structural controls, single seed.
- **Not usable for**: Conclusive claims about hybridization.

### Exp 13 — PSO Walk-Forward
- **Runs**: 5 windows
- **Boundary**: position_sma, 5 windows (3-year train / 1-year test)
- **Usable for**: Walk-forward protocol effect on PSO
- **Verified facts**: 2 / 5 windows beat BH; avg test = $1,636 (29 %
  drop from single-split $2,297)
- **Limitations**: 5 windows not statistically independent; low power.
- **Not usable for**: Statistical significance claims.

### Exp 03 — Robust Optimization
- **Runs**: 1 best-of run over 52 windows
- **Boundary**: 52 windows, 14D dual_crossover (NOT standard 3D)
- **Usable for**: Robust optimization protocol demonstration
- **Verified facts**: PSO wins 38.5 % of windows; avg excess return
  = −0.33; wins concentrated in bull markets (+1.99×), losses in bear
  markets (−4.73×)
- **Limitations**: 14D representation differs from all main experiments.
- **Not usable for**: Direct numerical comparison with 3D results.

---

## C-Grade: Do Not Use (CAUTION.md present)

The following experiments each contain a CAUTION.md file explaining
why they are unreliable. Do not cite them in the paper.

- **exp/02-walk_forward/** — GP walk-forward, single seed, data missing
- **exp/04-comparison/** — Single seed; PSO+MACD $948 is noise
- **exp/10-position-scale/** — Single seed; suspicious boundary behavior
- **exp/11-gp-tradeoff/** — Single-seed outlier ($3,142); superseded
- **exp/12-gp-functionset/** — Single seed; original set result likely
  outlier
- **exp/14-gp-pso-hybrid/** — Single seed
- **exp/16-gp-pso-lambda-sweep/** — Single seed, 4 lambdas
- **exp/20-budget-multiseed/** — BH = $5,660 ≠ standard $2,169; data
  range inconsistent

---

## D-Grade: Archived (exp/archive/)

The following experiments have been moved to exp/archive/:

- **01-ablation** — Infinity values, poor design
- **05-moe** — Strategy abandoned
- **06-position** — No information
- **07-pso-tradeoff** — Trivial conclusion, single seed
- **08-pso-inertia** — Near-identical results, single seed
- **design** — Not an experiment

---

## Registration Log

| Date | Experiment | Boundary | Registered by |
|------|-----------|----------|---------------|
| 2025-05-16 | Audit of all existing | — | system audit |
