#import "lib.typ": *
#import "taxonomy.typ"

#let title = "AI Trading Bots Project Survey"
#let unit = "CITS4404 Artificial Intelligence and Adaptive Systems"

#set page(
  numbering: "1",
  header: context {
    let page-num = counter(page).get().first()

    if page-num > 1 [
      #align(right, text(
        size: 10pt,
        style: "italic",
      )[#unit - #title])
      #v(-0.8em)
      #line(length: 100%, stroke: 0.5pt)
      #v(-0.5em)
    ]
  },
  footer: context {
    align(center, counter(page).display())
  },
)

#cover(
  title: title,
  subtitle: unit,
  // students: (
  //   "Lyuchen Dai (24754678)",
  //   "Mika Li (24386354)",
  //   "Simona Han (25152074)",
  //   "Xinqi Lin (24745401)",
  // ),
  date: "24 April 2026",
)

#set heading(numbering: "I.A.a)")
  #show heading: it => {
    // Find out the final number of the heading counter.
    let levels = counter(heading).get()
    let deepest = if levels != () {
      levels.last()
    } else {
      1
    }

    // set text(10pt, weight: 400)
    if it.level == 1 {
      // First-level headings are centered smallcaps.
      // We don't want to number the acknowledgment section.
      let is-ack = it.body in ([Acknowledgment], [Acknowledgement], [Acknowledgments], [Acknowledgements])
      set align(center)
      // set text(if is-ack { 10pt } else { 11pt })
      // show: block.with(above: 15pt, below: 13.75pt, sticky: true)
      // show: smallcaps
      if it.numbering != none and not is-ack {
        // numbering("I.", deepest)
        // h(7pt, weak: true)
      }
      it.body
    } else if it.level == 2 {
      // Second-level headings are run-ins.
      set text(style: "italic")
      // show: block.with(spacing: 10pt, sticky: true)
      if it.numbering != none {
        numbering("A.", deepest)
        h(7pt, weak: true)
      }
      it.body
    } else [
      // Third level headings are run-ins too, but different.
      #if it.level == 3 {
        numbering("a)", deepest)
        [ ]
      }
      _#(it.body):_
    ]
  }
#v(5mm)
#line(length: 100%)
#grid(
  columns: (1fr, 1fr, 1fr),
  // inset: 10pt,
  // column-gutter: 1.2cm,
  row-gutter: 1em,
  align: center + horizon,
  [*Name*], [*Student Number*], [*Algorithm*],
  [Simona Han], [25152074], [Particle Swarm Optimization \ @title:pso],  
  [Xinqi Lin], [24745401], [Artificial Bee Colony \ @title:abc],
  [Lyuchen Dai], [24754678], [Genetic Programming \ @title:gp],
  [Mika Li], [24386354], [Harmony Search \ @title:hs]
)
#line(length: 100%)
// #show: word-count

// Quick start: 
// - 对于需要引用的文献，可以在 Google Scholar 处检索，然后点击 cites， copy bib 样式的 reference 到 ref.bib 文件下，顺序不重要它会自动整理，然后直接 @xxx 引用即可

#let make-abstract(abstract) = {
  set block(spacing: 0pt)
  v(2em, weak: true)
  block(width: 100%, {
    set align(center)
    set text(size: 14pt)
    // smallcaps[Abstract]
    [*Abstract*]
  })
  v(16.4pt)
  pad(left: 0.5in, right: 0.5in, {
    // set text(size: 10pt)
    // set par(leading: 4.35pt)
    abstract
    v(1em)
  })
  v(3.8pt, weak: false)
}

#make-abstract(
  text[
    Trading bot design is a multi-dimensional optimization problem: selecting indicators, combining them into rules, and tuning parameters. Nature-inspired algorithms offer powerful tools for this task, but they differ fundamentally in what they optimize and how they represent solutions. This survey examines four algorithms spanning distinct paradigms: PSO and ABC for continuous parameter optimization, GP for discrete structure discovery, and HS as a single-state baseline. PSO and ABC optimize parameters within fixed structures; GP discovers the structures themselves; HS searches without structural assumptions. This divergence implies that algorithm performance depends critically on whether the problem is framed as parameter tuning or structure discovery. The reviewed literature indicates that parametric methods are robust but ceiling-limited by human design; structural methods discover novel combinations but face severe overfitting; simple heuristics provide a surprisingly competitive baseline. Algorithm selection, therefore, cannot be separated from representation design in trading bot optimization.
  ]
)

= Algorithms

== PSO @kennedy1995particle <title:pso>

=== Background and Motivation
Particle Swarm Optimization (PSO), introduced by Kennedy and Eberhart in 1995 @kennedy1995particle, is a population-based stochastic optimization technique inspired by the social behavior of bird flocking and fish schooling. It was introduced for the optimization of continuous nonlinear functions. The initial intention was to build an algorithm with simplicity and computational efficiency, requiring minimal memory and processing power.

The method is grounded in the idea that social behavior itself can be viewed as a form of optimization. By simulating how individuals share information and adjust their movements accordingly, PSO captures a mechanism that many traditional algorithms overlook. While Genetic Algorithms (GAs) simulate genetic evolution (survival of the fittest across generations), they largely ignore the benefits of social information sharing within a single generation. The authors, drawing from sociobiologist E.O. Wilson@wilson1975sociobiology, argued that the ability of individuals to profit from the discoveries of their peers is a decisive evolutionary advantage when resources (or solutions) are unpredictably distributed.

=== Core Idea and Algorithmic Novelty
The paper presents PSO as a “mid-level” form of artificial life, positioned between the slow timescales of genetic evolution and the millisecond-level processing of neural networks. A key distinction from genetic algorithms (GAs) lies in how candidate solutions are represented: rather than static points, PSO models each agent as a particle characterized by both *position* and *velocity*. The position represents a candidate solution in the search space, while the velocity determines the direction and magnitude of the particle's movement toward its next state, representing how these particles "fly" through a multidimensional hyperspace.

To navigate toward the optimum, each particle's movement is guided by two forms of "memory":
- pbest (Personal Best): A "nostalgic" tendency to return to the best location the particle has personally encountered.
- gbest (Global Best): A "social" tendency to move toward the best location discovered by any member of the entire swarm.

A key innovation is the concept of "controlled overshooting." By maintaining momentum (velocity) rather than simply jumping to new coordinates, particles tend to "hurtle past" their targets. This mechanical inertia allows the swarm to escape the traps of local optima and maintains a delicate balance between exploitation and exploration.

=== Demonstration and Empirical Results
The PSO model started as a simulation of a simplified social milieu - the movement of a flock of birds. Then it evolved through several stages of simplification to identify the essential components for functional optimization:

- Precursor Models: Early versions relied on "nearest-neighbor velocity matching" and a stochastic "craziness" variable to maintain movement synchrony and lifelike unpredictability.
- The Cornfield Vector: This milestone introduced the goal-seeking behavior just like how birds find food. The two critical values "pbest" and "gbest" were introduced at this stage and the "craziness" was eliminated.
- Final Simplified Formula: The model gor further simplified and a few other unnecessary ancillary variables were eliminated, such as p_increment and g_increment. The current version adjusts velocity (v) based on a weighted stochastic acceleration toward both pbest and gbest:
$  v_(t+1) = v_(t) + 2 ⋅ "rand"() ⋅ ("pbest−present")+2⋅"rand"()⋅("gbest−present") $

The algorithm demonstrated good performance on multidimensional experiments as well. It was validated on the Fisher Iris Data Set and the Schaffer f6 function and proved that PSO can train neural network weights as effectively as backpropagation and demonstrated its robustness.

=== Critical Assessment
In general, PSO is very simple, which only requires a few parameters. The implementation requires only a few lines of code and primitive mathematical operators, making it highly accessible. It is especially efficient for tracking dynamic system and high-dimensional, non-linear problems and effective for any network architecture@eberhart2001.

A few updated were done after the initial release of the algorithm. The use of inertia weight were introduced in 1998@shi1998 to optimize exploration and exploitation and constriction factor were introduced by Clerc in 1999@clerc1999 to insure convergence. PSO was shown to be robust in tracking and optimizing dynamic systems.

While PSO is highly efficient and easy to implement, it is not without limitations: the algorithm's inherent simplicity can be a drawback when optimizing complex trading models. To address the challenges of stock price prediction, a performance-based reward strategy (PRS) can be optimized using an improved time variant particle swarm optimization (TVPSO) to manage a large universe of trading rules. However, such complex strategies still face barriers such as computational time and potential signal noise, suggesting that future enhancements should incorporate rule screening or parallel computing to maintain optimization efficiency@fei2014.

// #pagebreak()

== Artificial Bee Colony @karaboga2007powerful <title:abc>

=== Background and Motivation
The paper addresses the optimization of continuous, high-dimensional multivariable functions. The core algorithmic challenge lies in navigating non-linear, strongly multimodal fitness landscapes characterized by severe epistasis and randomly distributed local optima, while avoiding the premature convergence and curse of dimensionality that plague traditional stochastic methods.

The development of the Artificial Bee Colony (ABC) algorithm addresses three structural flaws in previous heuristics:

+ *Space limits of early swarm models:* Prior algorithms (BCO @teodorovic2006bee, BSO @akbari2010novel) only solved discrete problems. The sole numerical attempt (VBA @yang2005engineering) failed against the curse of dimensionality, leaving continuous mapping ($R^D$) unresolved.

+ *Unstable movements in continuous optimizers:* PSO's global velocity vectors frequently cause "faulty updates" and out-of-bound individuals. Researchers built complex hybrid models (PS-EA @srinivasan2003particle) instead of fixing this core velocity instability.

+ *Coupled mechanisms causing premature convergence:* In multimodal landscapes, traditional methods stagnate in local optima because their tightly coupled exploration and exploitation behaviors force a trade-off between diversity and convergence.

*Consequently*, ABC aims to replace weak hybrid patches with a stable stochastic framework that naturally balances exploration and exploitation without complex derivative calculations.

=== Core Idea and Algorithmic Novelty
The ABC algorithm transitions swarm intelligence into continuous domains ($R^D$) via the *structural decoupling* of exploration and exploitation. To resolve PSO's faulty updates, it replaces global momentum with a dimensionally decoupled perturbation equation: 

$ v_(i j) = x_(i j) + phi_(i j) (x_(i j) - x_(k j)) $

Here, $x_(i j)$ is the current coordinate, $x_(k j)$ a random neighbor, and $phi_(i j) in [-1, 1]$ a stochastic scaling factor. Crucially, $v_(i j)$ calculates a *static target coordinate* based on spatial difference, which naturally prevents boundary overshoots by eliminating velocity inertia.

Functionally distinct agents optimize candidate solutions:

- *Employed & Onlooker Bees (Local Exploitation):* Employed bees perform neighborhood searches via the perturbation equation. This is intrinsically adaptive: as populations converge, shrinking spatial differences $(x_(i j) - x_(k j))$ automatically refine step sizes without external decay parameters. Onlookers allocate resources non-uniformly via *Roulette wheel selection* ($p_i = "fit"_i / (sum "fit"_n)$), driving intense selection pressure toward high-fitness regions. Both employ greedy selection.

- *Scout Bees (Global Exploration):* To prevent premature convergence, solutions stagnating beyond a `limit` hyperparameter are deemed local optima. The agent transforms into a scout, forcefully abandoning the trap for a purely random initialization in the global space.

=== Demonstration and Empirical Results
The architecture was empirically validated on five high-dimensional ($D = 10, 20, 30$) benchmarks (Griewank, Rastrigin, Rosenbrock, Ackley, Schwefel) simulating severe topological challenges like narrow valleys and deceptive local optima. ABC was benchmarked against GA, PSO, and a complex hybrid (PS-EA) across 30 independent runs with a population of 125. 

Statistical outcomes (mean and standard deviation) confirm ABC's superiority. While PS-EA deteriorated on highly epistatic landscapes (Griewank, Ackley), ABC consistently avoided premature convergence. Notably, despite GA and PS-EA converging faster initially on the deceptive Schwefel function, extending the maximum cycle number (MCN) allowed ABC to escape local traps and reach the global optimum. This empirically proves that ABC's decoupled structure inherently guarantees global exploration under sufficient computational cycles, rendering complex hybridization redundant.

=== Critical Assessment
The authors claim ABC's decoupled architecture decisively outperforms standard and hybrid baselines in multivariable, multimodal optimization. This is strictly justified by robust statistics (mean and standard deviation) from independent stochastic runs on severe benchmark landscapes. 

Critically, however, its global convergence lacks formal mathematical proof, relying solely on empirical validation. Furthermore, sensitivity analysis for key hyperparameters (e.g., `limit`, population size) is deferred to future studies. Despite these gaps, ABC's gradient-free robustness and natural evasion of local optima make it highly promising for our complex optimization experiments, provided that strict parameter tuning is applied.


== Genetic Programming @koza1992genetic <title:gp>

=== Background and Motivation
Conventional genetic algorithms represent solutions as fixed-length character strings over a finite alphabet. Koza @koza1992genetic argued that this representation "presents difficulties for some problems, particularly problems where the desired solution is hierarchical and where the size and shape of the solution is unknown in advance." Prior attempts to escape this constraint, including Smith's variable-length if-then rule systems (1980), Cramer's specialized languages (1985), and Holland's classifier systems (1986), all retained either fixed-length components or flat string encodings. Even "messy GAs" (Goldberg et al., 1989) patched the string representation rather than abandoning it. The core limitation was that specifying the representation in advance, as Koza put it, "narrows the window by which the system views the world and might well preclude finding the solution to the problem at all."

In financial markets, this representation problem manifests as ad hoc rule specification. Allen and Karjalainen @allen1999using identified this as the central methodological flaw in prior technical analysis research: studies like Brock, Lakonishok, and LeBaron (1992) hand-picked trading rules based on past performance, creating a data-snooping bias that invalidated their reported returns. If the rule structure is chosen by looking at historical data, then the backtest is no longer an independent test. The only principled alternative is to let the algorithm discover the rule structure itself, from data available before the test period.

=== Core Idea and Algorithmic Novelty
GP takes a different approach: instead of optimizing parameters within a fixed representation, it evolves the representation itself. A program is represented as a parse tree, where internal nodes are functions and leaf nodes are terminals (variables and constants). The key structural property is closure: every function in the function set can accept as arguments the output of any other function or any terminal, so that every possible tree is syntactically valid. This means the search can explore an open-ended space of program structures without requiring the designer to specify the size or shape of the solution in advance.

The genetic operators extend naturally to trees. *Selection* picks fitter individuals to reproduce based on a fitness function. *Crossover* swaps subtrees between two parent trees at randomly chosen nodes, producing offspring that combine structural elements from both parents. *Mutation* replaces a randomly chosen subtree with a newly generated one. Over generations, the population converges toward programs that score well on the fitness measure.

=== Demonstration and Empirical Results
Koza @koza1992genetic demonstrated GP on a range of problems, including symbolic regression and Boolean multiplexer synthesis, where the size and shape of the correct solution were not known in advance. These problems were chosen precisely because fixed-length string representations struggled with them: the solution required hierarchical composition of unknown depth, which flat encodings could not express without extensive pre-processing. The results showed that GP could discover correct or near-correct programs in these domains without human specification of the program structure.

Allen and Karjalainen @allen1999using then applied GP to financial markets, evolving trading rules from daily S&P 500 data over 1928--1995. Their terminal set included price, volume, and standard indicators; their function set included logical and relational operators. They deliberately included transaction costs and a minimum holding period to keep the evaluation realistic. The GP-discovered rules did beat buy-and-hold in some subperiods (especially volatile or declining markets), but the edge shrank substantially when they corrected for data-snooping bias.

=== Critical Assessment
The results are mixed in a revealing way. On the training set, GP reliably finds elaborate indicator combinations that look great, sometimes nesting momentum conditions inside channel-breakout logic in ways no human analyst would design. But out-of-sample, these complex rules often fall apart. The best in-sample performers frequently overfit to historical noise, and simple benchmarks like a plain SMA crossover can end up winning on held-out data. Allen and Karjalainen are upfront about this: the space of possible tree structures is enormous compared to the actual signal in financial time series, so spurious correlations are a real danger.

So does GP "work"? It depends on what you mean. If "work" means discovering compositional structures that parametric optimization cannot find, then yes, the experiments show that clearly. But if "work" means reliably producing profitable, generalizable trading strategies without careful regularization, then no. The bloat problem is real: unconstrained trees grow into unwieldy, uninterpretable monsters that encode historical accidents rather than robust patterns. The claim that GP-discovered rules "beat the market" is overstated; a fairer assessment is that GP reveals the *potential* of automated structure discovery, while also showing how easily that potential turns into overfitting without proper safeguards. For our project, this tension is precisely what makes GP worth studying: it searches the structural layer that parametric methods cannot reach, but the experiments make clear that depth limits, parsimony pressure, and strict train-test separation are not afterthoughts but essential constraints.


== Harmony Search @geem2001new <title:hs>

=== Background and Motivation
Harmony Search (HS) was proposed by Geem, Kim, and Loganathan in 2001 as a new heuristic optimisation algorithm inspired by musical improvisation @geem2001new. The paper is motivated by two related problems. First, traditional mathematical optimisation methods such as linear programming, nonlinear programming, and dynamic programming can work well in simple models, but they become less effective in complex real-world problems. Linear programming may lose important nonlinear features, dynamic programming suffers from rapidly increasing computational cost as the number of variables grows, and nonlinear programming may struggle when functions are not differentiable or when suitable initial values are hard to choose.

Second, although earlier heuristic methods such as simulated annealing@kirkpatrick1984optimization, tabu search@glover1977heuristics, and evolutionary algorithms@back1993overview already showed strong search ability, the authors argue that there was still room for a new search mechanism that could find good solutions with fewer iterations. In this sense, HS was not introduced because earlier methods had failed completely. Rather, it was proposed as an alternative framework based on a different analogy. This makes HS relevant to our project, since trading bot design is itself a multi-parameter optimisation problem in which a search method must handle many interacting variables efficiently.

=== Core Idea and Algorithmic Novelty
The main novelty of HS is that it treats optimisation as a process of musical improvisation. In this analogy, each decision variable is treated as a musical note, while one complete candidate solution is treated as a harmony. The algorithm stores several candidate solutions in Harmony Memory (HM), then generates a new harmony by drawing values from the existing memory or from the full allowable range. If the new harmony is better than the current worst harmony in memory, it replaces it.

Two parameters define this search process. Harmony Memory Considering Rate (HMCR) controls the probability of choosing a value from the existing memory rather than generating it randomly. Pitch Adjusting Rate (PAR) controls whether a selected value is slightly adjusted to a neighbouring value, which gives the algorithm a simple local refinement mechanism. Structurally, HS combines three search behaviors in one loop: reuse of past good solutions, random exploration, and small local adjustment. The paper also highlights an important difference between HS and the genetic algorithm. HS constructs a new vector by considering all stored harmonies in memory, whereas GA usually creates offspring from only two parents and must preserve gene structure. This gives HS a distinct search logic rather than making it just a minor variation of GA.

=== Demonstration and Empirical Results
The paper demonstrates HS through algorithm explanation, a brief discussion of convergence tendency, and three application problems. The first is a 20-city travelling salesman problem. In this case, HS reached the global optimum in 7 out of 30 runs within 20,000 iterations. After adding two extra operators, 11 out of 21 runs found the shortest tour within 5,000 iterations. The second is a constrained continuous optimisation problem, where HS was compared with the exact solution, the Generalized Reduced Gradient (GRG) method, a genetic algorithm, and evolutionary programming@fogel1995comparison. The paper reports that one HS solution achieved the best objective value among the heuristic methods, while another HS solution showed the best overall accuracy among the heuristic approaches, especially when some competing methods violated constraints.

The third application is the Hanoi water distribution network design problem. In this benchmark, HS found a least-cost solution of 6.056 million dollars, compared with 6.073 million dollars for GA and 6.320 million dollars for NLPG. The paper also includes an additional simple network example in the conclusion section, where HS reportedly found the optimal solution in only 1,095 iterations, while GA and simulated annealing required far more iterations. Taken together, these experiments suggest that HS is competitive across both combinatorial and continuous optimisation tasks, and that its performance can be better than several established methods in the tested cases.

=== Critical Assessment
Overall, the paper makes a reasonable case that HS is a promising heuristic optimisation method. Its main strengths are its clear search structure, its simple but original analogy, and its ability to work on both combinatorial and continuous problems. The combination of HM, HMCR, and PAR gives the method a natural balance between exploitation and exploration without relying on derivative information. The comparison with GA is also meaningful, because the method is presented as structurally different rather than just rhetorically different.

However, the paper’s conclusions should still be treated with caution. The benchmark set is limited, the number of test problems is small, and the effect of parameter settings is not studied in a systematic way. The paper also gives only a brief argument about convergence tendency rather than a formal mathematical proof. Therefore, the results support the claim that HS is promising, but they do not prove that HS is universally better than other optimisation methods. For our project, HS remains a strong candidate because it is a genuine multi-parameter optimisation algorithm with a relatively simple structure, and this makes it suitable for trading bot optimisation experiments.

// #pagebreak(weak: true)
= Conclusion and Comparison

The algorithms selected in this review are not variants on a theme. They represent fundamentally different approaches to search and optimization, each making distinct assumptions about the problem structure.

PSO and ABC are swarm intelligence methods. They maintain populations of candidate solutions in a continuous parameter space and navigate that space through social interaction (PSO's velocity update) or role-based foraging (ABC's employed/onlooker/scout division). Both assume the solution structure is fixed by a human designer; their contribution is finding good parameter values within that predetermined architecture. This makes them natural choices when the trading strategy template is known, for example, optimizing the window sizes and weights of a MACD or moving-average crossover system.

GP belongs to the evolutionary computation family, but it differs from standard genetic algorithms in a crucial way: it evolves program trees rather than parameter vectors. This shifts the optimization target from "what values" to "what structure." For trading bots, this means GP can discover which indicators to combine, which logical operators to use, and how to nest conditions. By contrast, PSO and ABC do not directly make these structural decisions when they are applied to a fixed representation. The trade-off is severe: GP's expressive power invites overfitting, and the literature shows that GP-discovered rules often underperform simple benchmarks on out-of-sample data.

HS occupies a different position. Inspired by musical improvisation, it stores candidate solutions in a harmony memory and generates new solutions by recombining stored values or introducing random pitches. Unlike PSO, HS does not rely on velocity-based swarm movement. Unlike ABC, it does not use explicit bee roles such as employed, onlooker, and scout bees. Unlike GP, it does not search over program structures. Instead, HS is better understood as a memory-based metaheuristic that balances exploitation through harmony memory and exploration through random selection and pitch adjustment. Its value in this review is as a baseline: if a more complex algorithm cannot consistently outperform HS, then its added complexity is hard to justify.

These differences lead to a clear taxonomy, as illustrated in @fig:taxonomy. PSO and ABC are *parametric optimizers*: they improve what is already designed. GP is a *structural optimizer*: it discovers what to design. HS is a *general-purpose heuristic*: it searches without strong assumptions about problem structure. For the trading bot project, this taxonomy maps directly to design decisions.

#figure(
  taxonomy.diagram,
  caption: [Taxonomy of reviewed algorithms by optimization target and search paradigm.],
) <fig:taxonomy>

If the strategy template is fixed, PSO or ABC is appropriate. If the strategy structure is unknown and the goal is to discover novel indicator combinations, GP is the natural choice, but only if regularization (depth limits, parsimony pressure, strict train-test separation) is enforced. HS provides a sanity check: any algorithm that claims superiority must first beat this simple baseline.

Based on this analysis, this suggests a possible two-stage experimental design. First, a representation comparison tests whether structure discovery (GP) offers end-to-end performance advantages over parameter optimization within fixed structures (PSO, ABC, HS). This stage tests whether GP’s larger search space is worth the extra complexity and overfitting risk in the trading bot task. Second, if GP identifies viable structures, a hierarchical combination examines whether parameter refinement on those structures (PSO or ABC optimizing GP-discovered templates) yields further improvements. Therefore, the comparison should not only ask which algorithm performs best overall. It should also examine which type of algorithm fits each design choice: fixed-structure parameter tuning, structural rule discovery, or simple baseline search.

// #pagebreak(weak: true)

#bibliography("refs.bib", style: "ieee", title: "Reference")
