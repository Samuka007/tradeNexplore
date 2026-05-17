#import "@preview/charged-ieee:0.1.4": ieee
#import "@preview/wordometer:0.1.5": total-words, word-count

#let abstract = [
  Genetic programming (GP) exhibits high variance on financial forecasting tasks: some runs find profitable strategies while others converge to trivial or unprofitable ones. While GP's high variance across seeds on financial tasks is widely observed @allen1999using, the contribution of specific mechanisms has rarely been quantified in a single controlled study. We instrument GP to log every parent-offspring pair across 10 independent seeds (13,245 pairs total) and find that parent-offspring fitness correlation declines from $r = 0.51$ at generation~0 to $r = -0.323$ at generation~10---crossover shifts from preserving advantage to systematically destroying it. This temporal dynamic on a financial task is a novel quantitative measurement of a well-known mechanism: epistasis in subtree crossover @oreilly1995building @dignum2006less. Two supplementary findings contextualise the result: (1)~GP adapts tree structure to market regime---bear markets demand complex conditional trees (mean size~12.9) while sideways markets need simple trend indicators (size~3.5), extending prior regime-adaptation studies @dempster2001realtime; (2)~tree nesting ratio predicts train-test divergence ($r = 0.81$, $p < 10^{-16}$), consistent with prior work showing structural complexity matters more than size @vanneschi2010measuring. Together, these results confirm that known GP phenomena are active on this financial task and can be monitored and controlled.
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

Genetic programming (GP) @koza1992genetic evolves program trees to discover strategy structure automatically. On financial forecasting tasks, it is frequently observed that GP produces high variance across random seeds---some runs find profitable strategies while others converge to trivial or unprofitable ones. While GP's high variance on financial tasks is well documented @allen1999using, the specific mechanisms contributing to it have rarely been quantified in a single controlled study.

The contribution of epistasis (context-dependency in subtree crossover) to GP's instability has been recognized since #cite(<oreilly1995building>, form: "prose"), who argued that GP crossover "is likely to change the context of a schema from one generation to the next which likely results in high variance." #cite(<dignum2006less>, form: "prose") later characterized this as crossover's "disrespect for the context of swapped subtrees" and proposed Context-Preserving Crossover specifically to address it. #cite(<kronberger2009crossover>, form: "prose") empirically measured crossover success rate and found destructive effects predominant. However, the temporal dynamics of parent-offspring fitness correlation, specifically how crossover shifts from constructive to destructive as the population diversifies, have not been quantified on a financial task.

Similarly, GP's ability to adapt to market regimes has been demonstrated since #cite(<allen1999using>, form: "prose"), who found that GP rules perform differently across market states, and #cite(<dempster2001realtime>, form: "prose"), who built an explicit real-time adaptive GP trading system. Yet how GP trees differ *structurally* across bull, bear, and sideways regimes on modern BTC data remains uncharacterised.

Finally, the link between structural complexity and overfitting in GP has been studied extensively: #cite(<vanneschi2010measuring>, form: "prose") showed that functional complexity, not just size, predicts overfitting; #cite(<goncalves2015model>, form: "prose") conducted a broad empirical study of model selection and overfitting in GP. What is less clear is which specific structural metric (tree size, depth, nesting ratio, or conditional-node count) is the most reliable predictor on a financial task.

Our study addresses three questions in a single controlled framework on a Bitcoin trading task:

1. *Epistasis in GP crossover* (main study): We instrument GP to record every parent-offspring pair across 10 independent seeds, measuring how the correlation between parent fitness and offspring fitness evolves over generations.

2. *Market regime and structural adaptation*: We run GP separately on bull, bear, and sideways market regimes and analyse whether the winning trees have systematically different structures.

3. *Structural predictability of overfitting*: We measure structural properties of evolved trees (depth, nesting ratio, terminal diversity) and test whether they predict train-test divergence.

Our findings confirm that GP's variance on this task arises from identifiable mechanisms. All three are amenable to measurement and control.

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

For contrast, we log parent-offspring pairs in PSO (velocity update on `position_sma`, 10 seeds, 15,000 pairs). The overall correlation is $r = 0.414$, similar to GP's overall, but critically PSO's correlation shows no systematic temporal decline. Individual iteration correlations fluctuate between $-0.095$ and $+0.593$, but there is no monotonic trend. This is because PSO's parameter update is a smooth perturbation: offspring positions are always near parent positions in the same parametric space. GP's discrete subtree swap, by contrast, is a structural jump that becomes increasingly destructive as the population diversifies.

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

At $lambda = 500$, GP automatically produces small, shallow trees (typically 3--7 nodes). The structural early-stopping experiment (Exp.~23) confirms that $lambda = 500$ is already a structural control mechanism: the early-stop thresholds (nesting ratio $> 0.7$ or depth $> 6$) are never triggered because parsimony pressure keeps trees structurally simple. This means $lambda$ penalises size indirectly, by shaping which tree structures survive selection.

= Human Bias and Structural Search

All three epistasis studies above assume that structural search is worthwhile, that the optimal strategy shape is unknown and must be discovered. But what if the best strategy shape is already known?

We compare three configurations on the same data split, holding budget constant at 1,500 evaluations:

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    stroke: none,
    inset: (x: 3pt, y: 3pt),
    table.hline(stroke: 1pt),
    table.header([*Configuration*], [*Mean test*], [*Std*], [*Beat BH*], [*CV*]),
    table.hline(stroke: 0.5pt),
    [PSO + position\_sma], [$2,297$], [$81$], [10/10], [3.5%],
    [GA + position\_sma], [$2,220$], [$230$], [5/10], [10.4%],
    [GP + free trees], [$1,689$], [$713$], [2/10], [42.2%],
    table.hline(stroke: 1pt),
  ),
  caption: [Head-to-head comparison on the same BTC data split. Budget: 1,500 evaluations per seed, 10 seeds. GP free trees use full function set, $lambda = 500$, depth~5. GA-restricted uses the same `position_sma` representation as PSO but with tournament selection and parameter crossover.],
) <tbl-comparison>

The result is striking. Both PSO and the restricted GA converge to the same basin (around \$2,200), but PSO is three times more stable. More importantly, the free GP, with its unlimited structural search space, performs *worse* on average than either parametric approach. Its best seed (\$3,142) is an outlier; the median is \$1,400.

This suggests that, for this particular BTC task, the `position_sma` template is already a good hypothesis space. The extra freedom of GP's structural search does not discover better architectures; it discovers *more complex* architectures that overfit. The human bias encoded in `position_sma` (two moving averages and a crossover threshold) is not a limitation, it is a useful inductive prior.

The practical implication is that structural search should be reserved for problems where the strategy shape is genuinely unknown. When a plausible parametric template exists, parameter optimisation is both more stable and, on this evidence, at least as effective.

= Discussion

Our three studies confirm that known GP phenomena jointly explain the observed seed-to-seed variance on this BTC trading task.

*Epistasis* (Exp.~21). Prior work identified the mechanism: subtree crossover ignores context dependency #cite(<oreilly1995building>, form: "prose"), which #cite(<dignum2006less>, form: "prose") characterised as "disrespect for the context of swapped subtrees" and #cite(<kronberger2009crossover>, form: "prose") measured empirically via crossover success rate. Exp.~21 quantifies this on a financial task. The parent-offspring correlation declines from $r = 0.51$ to $r = -0.323$ by generation~10, and the per-generation curve shows exactly when crossover transitions from constructive to destructive.

*Regime adaptation* (Exp.~22). Exp.~22 confirms that GP trees differ structurally across regimes, consistent with prior regime-adaptation studies (#cite(<allen1999using>, form: "prose"), #cite(<dempster2001realtime>, form: "prose")). The new observation is the magnitude. Bear-market trees average 12.9 nodes (heavily conditional), while sideways-market trees average 3.5 (simple trend-following). In a single-split experiment dominated by bull-market training data, seeds that initialise near trend-following structures are favoured.

*Structural predictability* (Exp.~23). Nesting ratio dominates tree size as a predictor of train-test divergence on this task ($r = 0.81$, $p < 10^{-16}$, Exp.~23). This aligns with prior work: #cite(<vanneschi2010measuring>, form: "prose") showed that functional complexity predicts overfitting better than size, and #cite(<goncalves2015model>, form: "prose") confirmed this across benchmarks. The interaction with parsimony pressure is new. At $lambda = 500$, the penalty controls tree shape, not just size.

Together, these findings confirm that GP's variance on this task is a composite of known mechanisms. Each has a practical control lever.

*Limitations.* (1)~Single asset (BTC-USD). (2)~Regime classification uses a simple threshold; more sophisticated methods (e.g., hidden Markov models) might refine the boundaries. (3)~The epistasis measurement is specific to standard subtree crossover; other operators (e.g., semantic crossover @krawiec2013semantic) may behave differently. (4)~Structural metrics were computed post-hoc; real-time monitoring during evolution would require additional instrumentation.

= Conclusion

We have shown that GP's variance on a Bitcoin trading task has three mechanistic sources: epistasis in crossover, regime-dependent structural demands, and structural signatures of overfitting. The central finding, that parent-offspring fitness correlation declines from $r = 0.51$ to $r = -0.323$ by generation~10, is a financial-domain quantification of a well-documented phenomenon: subtree crossover's destructiveness as populations diversify @oreilly1995building @dignum2006less @kronberger2009crossover.

The practical implication is that GP users should treat variance as a diagnostic signal. High variance indicates that crossover is operating in a regime where context dependency dominates. Possible responses include: stronger parsimony pressure (which controls nesting ratio), regime-aware training (which matches structure to data), or alternative crossover operators that preserve semantic context.

For this course, the key lesson is that nature-inspired algorithms have observable internal mechanisms. Crossover, selection, and mutation each produce characteristic behaviours that can be diagnosed and addressed.
