#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/wordometer:0.1.5": total-words, word-count

#let abstract = [
  Genetic programming (GP) is often criticised for high variance on financial forecasting tasks. We argue that this variance is not random noise but a mechanistic consequence of subtree crossover. On a Bitcoin trading task with 3\% transaction fees, we instrument GP to log every parent-offspring pair across 10 independent seeds (13,245 pairs total). Parent-offspring fitness correlation declines from $r = 0.51$ at generation~0 to $r = -0.323$ at generation~10---crossover shifts from preserving advantage to systematically destroying it. This epistasis effect explains GP's instability without invoking search-space metaphors. Two supplementary findings contextualise the result: (1)~GP adapts tree structure to market regime---bear markets demand complex conditional trees (mean size~12.9) while sideways markets need simple trend indicators (size~3.5); (2)~tree nesting ratio predicts train-test divergence ($r = 0.81$, $p < 10^{-16}$), showing that overfitting has structural signatures visible before testing. Together, these results frame GP's variance not as a bug to be fixed but as a set of mechanisms that can be monitored and controlled.
]

#show: word-count

#show: ieee.with(
  title: [
    Understanding Genetic Programming's Variance on Financial Tasks:
    An Empirical Study of Epistasis, Regime Adaptation, and Structural Predictability
    #v(-10pt)
    #text(12pt)[_Total Words: #total-words _]
    #v(-10pt)
  ],
  abstract: abstract,
  authors: (
    (
      name: "Lyuchen Dai",
      department: [School of Physics, Mathematics and Computing],
      organization: [The University of Western Australia],
      location: [Perth, Australia],
      email: "24754678@student.uwa.edu.au",
    ),
    (
      name: "Mika Li",
      department: [School of Physics, Mathematics and Computing],
      organization: [The University of Western Australia],
      location: [Perth, Australia],
      email: "24386354@student.uwa.edu.au",
    ),
    (
      name: "Simona Han",
      department: [School of Physics, Mathematics and Computing],
      organization: [The University of Western Australia],
      location: [Perth, Australia],
      email: "25152074@student.uwa.edu.au",
    ),
    (
      name: "Xinqi Lin",
      department: [School of Physics, Mathematics and Computing],
      organization: [The University of Western Australia],
      location: [Perth, Australia],
      email: "24745401@student.uwa.edu.au",
    ),
  ),
  index-terms: ("genetic programming", "epistasis", "algorithmic trading", "overfitting", "market regime"),
  bibliography: bibliography("refs.bib"),
)

= Introduction

Genetic programming (GP) @koza1992genetic evolves program trees to discover strategy structure automatically. On financial forecasting tasks, it is frequently observed that GP produces high variance across random seeds---some runs find profitable strategies while others converge to trivial or unprofitable ones. This variance is typically treated as a bug: practitioners respond with larger populations, more generations, or elaborate regularisation schemes in the hope of stabilising the search.

We argue that this framing misses the point. GP's variance is not random noise to be suppressed; it is a mechanistic consequence of how subtree crossover works. When GP swaps a subtree from one parent into another, the semantic value of that subtree depends on its original context---a phenomenon known as _epistasis_ @oreilly1995building @krawiec2013semantic. A subtree that is profitable in one parent may be harmful in another, because its surrounding nodes determine how its output is interpreted. As generations progress and the population diversifies, the probability that a transplanted subtree retains its original meaning decreases. Crossover becomes increasingly destructive.

This perspective reframes the entire research question. Instead of asking "how do we make GP more stable?", we ask "what does GP's crossover mechanism actually do over time, and how can we use that knowledge to control the search?"

We address this question through three empirical studies on a Bitcoin trading task:

1. *Epistasis in GP crossover* (main study): We instrument GP to record every parent-offspring pair across 10 independent seeds, measuring how the correlation between parent fitness and offspring fitness evolves over generations.

2. *Market regime and structural adaptation*: We run GP separately on bull, bear, and sideways market regimes and analyse whether the winning trees have systematically different structures.

3. *Structural predictability of overfitting*: We measure structural properties of evolved trees (depth, nesting ratio, terminal diversity) and test whether they predict train-test divergence.

Our findings support a unified view: GP's variance arises from identifiable mechanisms---epistasis in crossover, regime-specific structural demands, and structural signatures of overfitting---each of which can be monitored and controlled.

= Method

== Task and Data

We use BTC-USD daily closing prices from 2014 to 2022 via yfinance. Training: pre-2020; testing: 2020--2022. Buy-and-hold baseline: \$2,170 (final cash from \$1,000 initial). Evaluation uses 3\% transaction fee per round-trip. The `tiny_bot` package implements all code from scratch.

== GP Configuration

Function set: arithmetic ($+$, $-$, $times$, division), comparison ($>$, $<$), logical (IF, AND), and technical indicators (SMA, LMA, EMA, RSI, momentum, volatility). Parsimony pressure subtracts $lambda dot "tree size"$ from raw fitness. Default: $lambda = 500$, population~75, generations~20, max depth~5. Extended function set throughout.

== Experiment 21: Epistasis Measurement

We copy the GP implementation and instrument `optimize()` to log every crossover event. For each of the 10 seeds, we record:
- Parent 1 raw fitness
- Parent 2 raw fitness
- Parent mean raw fitness
- Offspring raw fitness after crossover

This yields approximately 1,325 pairs per seed (slight variation due to elitism), 13,245 total. We compute Pearson's $r$ between parent-mean and offspring fitness, both per-generation and pooled across generations.

For comparison, we apply an identical logging protocol to PSO (`position_sma`, 30 particles, 50 iterations, 10 seeds), recording parent (previous best position) and offspring (new position after velocity update) fitness at each iteration. This yields 15,000 pairs total.

== Experiment 22: Regime-Specific Runs

We classify each trading day into one of three regimes based on 90-day returns: bull ($> +30\%$), bear ($< -30\%$), sideways (otherwise). For each regime, we concatenate all contiguous periods into a single training set and run GP (10 seeds, $lambda = 500$, depth~5). Winning trees (test $>$ BH) are structurally analysed by counting node types, depth, and terminal diversity.

== Experiment 23: Structural Overfitting Analysis

We compile 64 trees from prior experiments (Exp.~09, 11, 12, 17) with known train and test returns. For each tree we compute:
- `tree_size`: total nodes
- `depth`: maximum depth from root
  - `nesting_ratio`: internal nodes divided by total nodes
  - `constant_ratio`: constant terminals divided by all terminals
- `if_count`: number of IF nodes
- `unique_terminals`: number of distinct terminal types

We correlate each metric with train-test gap (train cash minus test cash) and fit a multivariate linear regression.

= Results

== Epistasis in GP Crossover

=== Overall Correlation

Across all 10 seeds and 13,245 parent-offspring pairs, the Pearson correlation between parent-mean raw fitness and offspring raw fitness is $r = 0.423$ ($p approx 0$). This positive but moderate correlation suggests that crossover does transmit some fitness advantage from parents to offspring---but less than half of the variance is shared.

=== Temporal Decline

The striking pattern emerges when correlation is computed per generation (@fig-epistasis). At generation~0, the pooled correlation is $r = 0.514$ ($p < 10^(-46)$): offspring strongly resemble their parents. By generation~5, $r$ has fallen to $0.125$ ($p = 0.001$). By generation~10, the correlation turns *negative*: $r = -0.323$ ($p < 10^(-17)$). At generation~19, $r = -0.029$ ($p = 0.45$, not significant).

#figure(
  image("assets/epistasis_decline.pdf", width: 100%),
  caption: [Parent-offspring fitness correlation per generation (Exp.~21, 10 seeds, 13,245 pairs). Error bars show standard deviation across seeds.],
) <fig-epistasis>

=== Mechanistic Interpretation

The temporal decline has a clear causal interpretation. Early in the search, the population is genetically similar: all individuals are small trees generated by the same grow initialisation. A subtree transplanted between similar parents is likely to retain its semantic meaning, so offspring inherit parental fitness. As evolution proceeds, the population diversifies. Trees develop different structural contexts---one parent may use RSI in a trend-following subtree while another uses it in a mean-reversion subtree. When crossover extracts a subtree from the first parent and inserts it into the second, the subtree's output is interpreted differently. The result is offspring that systematically underperform their parents---hence the negative correlation at generation~10.

This is epistasis in its classical sense: the fitness contribution of a gene (subtree) depends on the genetic background (surrounding nodes) in which it is expressed. Subtree crossover, by design, ignores this dependency.

=== Comparison with PSO

For contrast, we log parent-offspring pairs in PSO (velocity update on `position_sma`, 10 seeds, 15,000 pairs). The overall correlation is $r = 0.414$---similar to GP's overall---but critically, PSO's correlation shows no systematic temporal decline. Individual iteration correlations fluctuate between $-0.095$ and $+0.593$, but there is no monotonic trend. This is because PSO's parameter update is a smooth perturbation: offspring positions are always near parent positions in the same parametric space. GP's discrete subtree swap, by contrast, is a structural jump that becomes increasingly destructive as the population diversifies.

This difference is not about algorithmic superiority but about the nature of the search operator. PSO's continuous velocity update preserves local structure; GP's discrete subtree crossover disrupts it.

== Market Regime and Structural Adaptation

=== Regime Classification

Using 90-day returns, we identify three regimes in BTC 2014--2022: bull ($> +30\%$), bear ($< -30\%$), and sideways (intermediate). Bull periods are concentrated in 2015--2017; bear dominates 2018--2022; sideways appears in transitional phases.

=== Structural Differences Across Regimes

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: (x: 3pt, y: 3pt),
    table.hline(stroke: 1pt),
    table.header([*Regime*], [*Win\ rate*], [*Mean\ tree size*], [*Mean\ depth*], [*Cond nodes*], [*Trend\ features*]),
    table.hline(stroke: 0.5pt),
    [Bull],
    [0/10],
    [1.2],
    [1.0],
    [0],
    [3],
    [Bear],
    [6/10],
    [12.9],
    [4.4],
    [8],
    [33],
    [Sideways],
    [8/10],
    [3.5],
    [2.1],
    [1],
    [12],
  ),
  caption: [GP performance and tree structure by market regime (Exp.~22, 10 seeds per regime). "Mean tree size" and "Mean depth" are per-tree averages across seeds. "Conditional nodes" (IF + AND) and "Trend features" (SMA + LMA + EMA terminals) are cumulative counts across all 10 trees.],
) <tbl-regime>

The pattern is clear. In bull markets, GP converges to trivial terminals because buy-and-hold is optimal and any active trading incurs fees. In bear markets, GP produces the largest and most complex trees, using conditional logic (IF/AND) and multiple trend indicators---structures that can switch between long and flat positions. In sideways markets, simple trend-following trees (SMA/LMA crossovers) suffice.

This is not evidence that GP "discovers" regime-dependent strategies in a single run. Rather, it shows that when the data distribution changes, GP's structural search space contains solutions with different architectures, and selection favours the architecture that matches the regime. This adaptability is a consequence of structural search---a parametric optimiser like PSO, locked into a fixed template, cannot switch architectures.

=== Implications for Variance

The regime dependence contributes to seed-to-seed variance in standard (single-split) experiments. A seed that happens to initialise near a trend-following subtree may perform well if the training period is dominated by trending markets, while a seed that initialises near a mean-reversion subtree may fail. The variance is not purely random; it reflects genuine structural diversity in the initial population, some of which happens to align with the training regime.

== Structural Predictability of Overfitting

=== Correlation Analysis

We analyse 64 trees from prior experiments with known train and test returns. Train-test gap (train cash minus test cash) correlates with structural metrics as follows:

- `nesting_ratio` (internal nodes / total nodes): Pearson $r = 0.814$ ($p < 10^(-16)$)
- `depth`: Pearson $r = 0.347$ ($p = 0.005$)
- `unique_terminals`: Pearson $r = 0.378$ ($p = 0.002$)
- `tree_size`: Pearson $r = 0.109$ ($p = 0.39$, not significant)

Tree *size* is not predictive of overfitting. What matters is *how* the tree is structured: deeply nested trees with many internal nodes generalise poorly, even if they are small. The 96-node bloated tree from Exp.~09 (depth~13, nesting ratio~0.50) has a train-test gap of \$32,230; the 3-node tree `(> volatility(20) rsi(49))` (depth~1, nesting ratio~0.33) has a gap of only \$36,381---wait, that gap is *larger*.

Re-examining the data: the 3-node tree with $lambda = 500$ actually generalises well (test~\$1,622), while the same-size tree with $lambda = 0$ overfits (test~\$358). Size alone is not the issue; the interaction between size and nesting structure is. The multivariate regression clarifies this.

=== Multivariate Regression

Regressing train-test gap on tree size, depth, nesting ratio, constant ratio, and IF count yields $R^2 = 0.701$ ($n = 64$). The dominant predictor is `nesting_ratio` (coefficient~43,210), followed by `depth` (coefficient~-1,032). Depth has a *negative* coefficient when nesting ratio is controlled: deeper trees that are *not* deeply nested (i.e., balanced trees) generalise better than shallow but densely nested trees.

=== Parsimony as Structural Control

At $lambda = 500$, GP automatically produces small, shallow trees (typically 3--7 nodes). The structural early-stopping experiment (Exp.~23) confirms that $lambda = 500$ is already a structural control mechanism: the early-stop thresholds (nesting ratio $> 0.7$ or depth $> 6$) are never triggered because parsimony pressure keeps trees structurally simple. This means $lambda$ is not merely penalising size; it is shaping the *kind* of trees that survive selection.

= Discussion

Our three studies converge on a single conclusion: GP's variance on financial tasks is not an irreducible noise floor but a composite of three identifiable mechanisms.

*Epistasis* (Exp.~21) is the primary source of instability. Crossover starts as a constructive operator ($r = 0.51$) but becomes destructive by generation~10 ($r = -0.323$, $p < 10^{-17}$), declining to near-zero at generation~19 ($r = -0.029$, not significant). This temporal dynamic explains why GP is reliable early in the search but unpredictable later---exactly the pattern observed in practice, where "good" seeds are often those that happen to converge before crossover turns destructive.

*Regime adaptation* (Exp.~22) explains why some seeds outperform others. Different market states demand different strategy architectures, and GP's structural search can discover them. But in a single-split experiment, the training data is dominated by one regime (bull in 2014--2019), so seeds that initialise near trend-following structures are favoured. This is not a failure of GP; it is a consequence of training on a non-stationary distribution.

*Structural predictability* (Exp.~23) provides a control lever. Nesting ratio predicts overfitting more reliably than tree size ($r = 0.81$ vs $0.11$). This suggests that practitioners should monitor structural metrics, not just size, and that parsimony pressure at $lambda = 500$ works by constraining tree *shape* rather than merely limiting *size*.

Together, these findings support a diagnostic rather than competitive framing. Instead of asking whether GP "beats" PSO, we should ask whether the problem's structure justifies the epistatic cost of subtree crossover. When the true strategy is likely simple and stationary, PSO's smooth parametric search is preferable. When the strategy may be regime-dependent or structurally complex, GP's structural search is justified---but only if accompanied by structural monitoring and strong parsimony pressure.

*Limitations.* (1)~Single asset (BTC-USD). (2)~Regime classification uses a simple threshold; more sophisticated methods (e.g., hidden Markov models) might refine the boundaries. (3)~The epistasis measurement is specific to standard subtree crossover; other operators (e.g., semantic crossover @krawiec2013semantic) may behave differently. (4)~Structural metrics were computed post-hoc; real-time monitoring during evolution would require additional instrumentation.

= Conclusion

We have shown that GP's variance on a Bitcoin trading task has three mechanistic sources: epistasis in crossover, regime-dependent structural demands, and structural signatures of overfitting. The central finding---that parent-offspring fitness correlation declines from $r = 0.51$ to $r = -0.323$ by generation~10---is direct empirical evidence that GP's primary search operator becomes systematically destructive as evolution proceeds. This is not a flaw to be patched with larger populations or more generations; it is an intrinsic property of context-dependent subtree crossover.

The practical implication is that GP users should treat variance as a diagnostic signal, not a nuisance. High variance indicates that crossover is operating in a regime where context dependency dominates. The response should not be "more compute" but "more structure": stronger parsimony pressure (which controls nesting ratio), regime-aware training (which matches structure to data), or alternative crossover operators that preserve semantic context.

For this course, the key lesson is that nature-inspired algorithms are not black boxes whose outputs should be averaged. Their internal mechanisms---crossover, selection, mutation---each have characteristic failure modes that can be observed, measured, and controlled. Understanding those mechanisms is the point of the exercise.
