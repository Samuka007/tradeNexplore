#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/wordometer:0.1.5": total-words, word-count

#let abstract = [
  PSO and GP behave characteristically differently on the same Bitcoin trading task. PSO's velocity-averaging converges to two stable basins regardless of initialisation, revealing a smooth fitness landscape. GP's discrete tournament selection produces high variance ($sigma$ = \$677) on unrestricted trees, revealing a rugged structural space. A control experiment---restricting GP to PSO's exact 3-parameter representation---shows both algorithms discover the same basins, but PSO's convergence reliability is higher (10/10 vs 5/10 seeds beating buy-and-hold). A 171-point grid search visualises the two-basin geometry, and generation-level trajectories show GP's train--test divergence without parsimony pressure. A 42-run grid search identifies $lambda = 500$ as GP's sweet spot, correcting our own earlier single-seed result (Exp.~09) that favoured $lambda = 1,000$. Walk-forward validation cuts returns by 29\% and causes GP to collapse to a no-trade strategy while PSO remains robust, revealing that evaluation protocol choice interacts with search mechanism. For this problem, our results suggest that representation shapes attractor geometry while the algorithm shapes exploration reliability.
]

#show: word-count

#show: ieee.with(
  title: [
    Comparing Parametric and Structural Optimisation for BTC Trading Bots:
    A Systematic Study of PSO and GP
    #v(-10pt)
    #text(12pt)[_Total Words: #total-words _]
    #v(-10pt)
  ],
  abstract: abstract,
  authors: (
    (name: "Lyuchen Dai", department: [School of Physics, Mathematics and Computing], organization: [The University of Western Australia], location: [Perth, Australia], email: "24754678@student.uwa.edu.au"),
    (name: "Mika Li", department: [School of Physics, Mathematics and Computing], organization: [The University of Western Australia], location: [Perth, Australia], email: "24386354@student.uwa.edu.au"),
    (name: "Simona Han", department: [School of Physics, Mathematics and Computing], organization: [The University of Western Australia], location: [Perth, Australia], email: "25152074@student.uwa.edu.au"),
    (name: "Xinqi Lin", department: [School of Physics, Mathematics and Computing], organization: [The University of Western Australia], location: [Perth, Australia], email: "24745401@student.uwa.edu.au"),
  ),
  index-terms: ("particle swarm optimization", "genetic programming", "technical analysis", "algorithmic trading", "overfitting"),
  bibliography: bibliography("refs.bib"),
)


= Introduction

Nature-inspired algorithms differ fundamentally in _what_ they optimise and _how_ they represent solutions. PSO @kennedy1995particle tunes continuous parameters within a fixed template; GP @koza1992genetic discovers the template itself by evolving program trees. Which is better---and does the question make sense without specifying the template?

We choose PSO and GP as two ends of a spectrum---parametric vs.\ structural optimisation---and apply them to Bitcoin trading with a 3\% transaction fee. Our question is not "which algorithm wins?" but _what does each algorithm's behaviour reveal about the search space it inhabits?_

This reframing matters because comparisons are typically apples-to-oranges: PSO searches 3 parameters while GP searches an infinite-dimensional tree space. We run a control experiment---restricting GP to PSO's 3-parameter representation---to disentangle representation from algorithm. Across 20 experiments and ~120 runs, the answer is nuanced. The algorithms' _behavioural signatures_ are diagnostic of different search-space geometries. When representation is held constant, both discover the same basins, but PSO's continuous velocity-averaging provides convergence stability that GP's discrete tournament selection lacks.

= Bot Design and Parameterisation

We implemented three strategy families: discrete crossover (`dual_crossover`), trivial SMA (`trivial_sma`), and position SMA (`position_sma`), the latter using a sigmoid to produce continuous exposure in $[0, 1]$:
$ p_t = "sigmoid"(s("fast") - s("slow")) $
We abandoned MACD and Mixture-of-Experts after pilot ablations (test \$949 and \$478 respectively). Evaluation uses \$1,000 initial cash, 3\% fee per round-trip, and final cash as fitness. Training: pre-2020 BTC-USD daily; testing: 2020--2022 @bitcoin_kaggle.

= Algorithm Selection

From Part 1's four algorithms, PSO and GP were selected; ABC and HS were dropped after pilots showed no advantage.

*PSO.* 30 particles, 50 iterations, inertia decay $w: 0.9 arrow.r 0.4$, $c_1 = c_2 = 2.05$. The velocity-update formula---weighted sum of inertia, cognitive, and social terms---provides implicit smoothing that damps local noise.

*GP.* Tournament selection (size 3), subtree crossover 0.9, mutation 0.1, grow initialization (depth 5/7). Function set: arithmetic, comparison (LT, GT), logical (IF, AND), and technical indicators (SMA, EMA, LMA, RSI, momentum, volatility). Parsimony pressure subtracts $lambda dot "tree size"$ from raw fitness. Without it, trees bloat to 96 nodes.

= Experimental Design

Experiments follow controlled ablation: change one decision at a time. The suite comprises 20 numbered experiments plus 10 supplementary analyses (~120 runs). Every experiment uses a fixed seed (42 default; 10-seed sweeps for robustness). The `tiny_bot` package implements all code from scratch without external optimisation libraries.

Three evaluation protocols are compared: single split (train pre-2020, test 2020--2022); walk-forward (5 windows, 3-year train / 1-year test); and robust optimisation (worst-case across 52 sliding windows). Controlled ablations (Exp.~01) confirm regime shift and transaction costs are dominant constraints.

= Results

== Search-Space Structure: What PSO and GP Reveal

=== PSO Reveals a Smooth Parametric Landscape

Across 10 seeds, PSO (`position_sma`) achieves mean test \$2,297 ($sigma$ = \$77), with every seed beating BH (\$2,170). Two dominant basins emerge: Basin A $(119, 179, 0.1)$ at \$2,366 (6 trades) and Basin B $(37, 99, 0.1)$ at \$2,264 (12 trades). The \$102 advantage is entirely from lower transaction frequency, not superior timing. Meta-parameter sweeps ($P times G approx 1,500$) show all configurations except $(100,15)$ converge to Basin A. Inertia strategy makes no meaningful difference.

=== GP Reveals a Rugged Structural Landscape

Raw GP (75-pop/20-gen) achieves mean \$1,689 ($sigma$ = \$677), with only 2/10 seeds beating BH. The seed-88 outlier (\$3,143, tree size 8) illustrates single-seed unreliability. This volatility is diagnostic: unrestricted tree search on weak signal produces a hypothesis space so large that selection is effectively random without regularisation, consistent with Allen and Karjalainen @allen1999using.

#figure(image("assets/seed_robustness.pdf", width: 100%), caption: [GP seed robustness: random vs.\ warm-start (Exp.~15). Dashed = buy-and-hold.]) <fig-seed>

*Generation-level trajectory.* Under $lambda = 0$, train fitness rises from \$20,674 to \$38,003 while test stagnates at ~\$1,600 and collapses to \$1,000 at generation 11; tree size grows from 6 to 30 nodes. Under $lambda = 500$, train stabilises at \$25,019 from generation 3, test holds at \$2,245, and tree size stays at 3 nodes (@fig-trajectory). Parsimony pressure defines the effective search space by pruning overfit subspaces.

#figure(image("assets/gp_trajectory.pdf", width: 100%), caption: [GP trajectory: $lambda = 0$ vs.\ $lambda = 500$ (Exp.~20).]) <fig-trajectory>

Warm-start (50\% human rules) raises mean to \$2,362 (6/10 beat BH), though 5/10 seeds converge to the same \$3,143 tree, inflating the mean; adjusted mean is \$1,678.

== The Control Experiment: Same Representation, Different Algorithm

To isolate algorithm from representation, Experiment 18 restricts GP to PSO's exact 3-parameter space (`position_sma`). The "tree" is a single node; crossover swaps parameters uniformly; mutation adds Gaussian noise.

#figure(image("assets/landscape_grid.pdf", width: 100%), caption: [Parametric landscape (Exp.~19, 171 points). Left: train; Right: test. Basin A $(120, 180)$ and Basin B $(40, 100)$ visible.]) <fig-landscape>

GP restricted converges to the same basins as PSO. A 171-point grid search confirms the two-basin geometry: Basin A mean train \$23,861, Basin B mean train \$20,642. The landscape is smooth---no sharp local optima, only a low ridge.

On this landscape, our results suggest representation shapes attractor geometry while the algorithm shapes exploration reliability.

== Regularisation Defines the Effective Hypothesis Space

=== GP: Parsimony Pressure

A single-seed sweep suggested $lambda = 1,000$ was optimal. A 42-run grid search (3 seeds $times$ 7 $lambda times$ 2 depths) corrects this: $lambda = 500$ is the sweet spot (@tbl-lambda). $lambda < 500$ permits bloat; $lambda > 500$ shrinks trees to 1--3 nodes and underfits. At $lambda = 500$, mean tree size is 5.7 nodes. Depth 5 slightly outperforms depth 7 (\$1,656 vs.\ \$1,408). Extended function set dominates (\$1,622 vs.\ \$359 original, \$1,000 minimal). Larger budgets paradoxically hurt: $50 times 30$ achieves \$2,275; $150 times 75$ collapses to \$359.

#figure(table(columns: (auto, auto, auto, auto, auto), stroke: none, inset: (x: 6pt, y: 3pt), table.hline(stroke: 1pt), table.header([*$lambda$*], [*Mean test (\$)*], [*Beat BH*], [*Mean tree size*], [*Std (\$)*]), table.hline(stroke: 0.5pt), [100], [988], [0/6], [7.0], [526], [250], [1,025], [0/6], [3.8], [675], [500], [2,366], [3/6], [5.7], [593], [750], [1,553], [0/6], [4.8], [402], [1,000], [1,478], [1/6], [3.3], [934], [2,000], [1,434], [0/6], [3.0], [482], [5,000], [1,879], [1/6], [1.3], [280]), caption: [Systematic hyperparameter grid (Exp.~17, 42 runs).]) <tbl-lambda>

#figure(image("assets/lambda_sweep.pdf", width: 100%), caption: [GP parsimony pressure vs.\ test return (Exp.~17).]) <fig-lambda>

== PSO: Implicit Regularisation

PSO needs no explicit parsimony because velocity-update averages noisy gradients across the swarm. The control experiment confirms this: GP restricted discovers the same basins with 3$times$ higher variance. The difference is mechanism, not landscape.

== Evaluation Protocol Is Not Neutral

Walk-forward validation (5 windows) reduces PSO mean from \$2,296 to \$1,636 (29\% drop); win rate falls from 10/10 to 2/5. GP degenerates to a no-trade strategy (0\% win rate). Walk-forward averaging destroys the sharp gradients GP's tournament selection relies on; with fitness compressed to noise, selection collapses. PSO's swarm averaging is naturally robust to window-wise noise. This reveals a mechanistic difference: GP's selection is discrete (keep/kill), PSO's is continuous (weighted velocity update). Evaluation protocol choice interacts with search mechanism.

Robust optimisation (52 windows) yields the same pattern: PSO wins 38.5\%; GP degenerates to no-trade.

#figure(image("assets/walk_forward.pdf", width: 100%), caption: [Walk-forward results (Exp.~13).]) <fig-wf>

== Joint Optimisation Cannot Be Decoupled

GP-then-PSO hybridisation: 3-node trees improve +7\%; 5-node trees destroy performance (-51\%); 37-node trees show no effect. GP's fitness jointly optimises structure and parameters; post-hoc PSO overfits in ways that break GP's implicit regularisation. Structure and parameters are entangled and must be optimised jointly.

== Transaction Costs Shape the Effective Landscape

At 0\% fees PSO returns \$2,841; at 3\%, \$2,366. Break-even: ~4.4\%. Classic 50/200 SMA (1 trade) achieves \$2,236, confirming trading frequency dominates net profit. Both algorithms converge to low-frequency solutions, suggesting the optimum lies near the no-trade boundary. Transaction costs are a structural feature of the landscape.

== Statistical Significance

Wilcoxon signed-rank (one-sided, $alpha = 0.05$) and Mann-Whitney U tests. We note the data-snooping problem in technical trading rules highlighted by White @white2000reality and Sullivan et al.\ @sullivan1999data; our 7-value $lambda$ grid warrants caution with multiple comparisons. We report raw p-values and apply Bonferroni correction as a simple conservative heuristic, acknowledging that more sophisticated methods such as White's reality check are beyond this course's scope.

+ *PSO vs.\ BH*: $W = 55$, $p = 0.001$. Significant.
+ *GP random vs.\ BH*: $W = 9$, $p = 0.976$. Not significant.
+ *GP warm-start vs.\ BH*: $W = 36$, $p = 0.211$. Not significant despite 6/10 win rate.
+ *PSO vs.\ GP random*: $U = 85$, $p = 0.004$, $r = -0.70$. Large effect.
+ *Warm-start vs.\ random GP*: $U = 79$, $p = 0.014$. Significant.
+ *Across $lambda$*: Kruskal-Wallis $H = 16.2$, $p = 0.013$. $lambda = 500$ vs.\ $lambda = 1000$: $U = 30$, $p = 0.032$.
+ *Walk-forward*: $W = 3$, $p = 0.625$. Not significant (5 windows only).

== Risk-Adjusted Returns

PSO CV = 3.3\%; GP random = 40\%; warm-start = 38\%; GP restricted = 10.4\%. Variance is driven by both representation and algorithm mechanism. Estimated annualised Sharpe: ~0.75--0.88 for PSO. Precise Sharpe and max drawdown require daily equity curves not exported---a limitation.

= Discussion and Limitations

PSO's stability reveals a benign parametric landscape; GP's instability reveals a deceptive structural space. The control experiment resolves the confound that plagued all previous comparisons: when GP is restricted to PSO's representation, it finds the same basins with lower reliability. Our initial working hypothesis---that representation dominates algorithm selection---proved too simplistic. For this problem, the evidence suggests representation shapes attractor geometry while the algorithm shapes exploration reliability.

López de Prado

*Limitations.* (1) Single asset (BTC-USD). (2) Moving-average variants only. (3) Three market regimes only. (4) Incomplete $lambda times$ depth validation. (5) No daily equity curves for Sharpe/drawdown (Sullivan et al.\ @sullivan1999data). (6) Walk-forward limited to 5 windows.

= Conclusion

Three patterns recur across experiments. First, PSO's swarm averaging makes it robust on smooth parametric landscapes; its 3.3\% CV reveals broad attractors and low ridges. Second, GP's discrete selection makes it powerful but fragile on rugged landscapes; $lambda = 500$ is an empirically tuned complexity control identified by grid search. Third, evaluation protocol is an algorithmic variable---walk-forward destroys GP's selection signal while leaving PSO intact.

The practical implication: match the algorithm to what you know about the problem. Known structure $arrow.r$ parameterise + implicit regularisation. Unknown structure $arrow.r$ structural search with explicit regularisation, multi-seed validation, and honest protocols that preserve the selection signal.
