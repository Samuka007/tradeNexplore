# AGENT.md — Experiment Protocol for This Repository

## Mandatory Checklist Before Running Any Experiment

1. [ ] Read `exp/PITFALLS.md` — Know every pitfall that has already
   been discovered. Do not repeat them.

2. [ ] Read `exp/USABLE.md` — Know what boundary conditions apply to
   existing experiments. Do not cite unregistered experiments.

3. [ ] Verify your experiment's boundary conditions do not conflict
   with existing A-grade experiments (see Data Range below).

4. [ ] After running, register the experiment in `exp/USABLE.md` with
   exact boundary conditions and limitations.

## Data Range Verification (Do Not Skip)

Before running any experiment, verify:

- Train period: prices before 2020-01-01
- Test period: prices from 2020-01-01 onward
- Expected buy_and_hold baseline: **$2,169.58**

If your experiment produces a different BH, STOP. You are using a
different data range and your results are NOT comparable to any
existing A-grade experiment. Either fix your data split or run the
experiment in a completely separate analysis branch with no cross-citation.

## Seed Requirements

| Claim type | Minimum seeds | Notes |
|-----------|--------------|-------|
| Pilot / hypothesis | 1 | Must be labeled as pilot |
| Qualitative pattern | 3 | e.g., "converges to basin A" |
| Mean performance | 10 | Report mean, std, median |
| Variance estimate | 10 | Report sample std (s), not population σ |
| Win rate | 10 | Report exact fraction (e.g., 5/10) |

Single-seed experiments are ONLY for hypothesis generation. They may
NOT support quantitative claims in the paper.

## Algorithm Labeling Requirements

- **GP** = tree-based genetic programming with function nodes, subtree
  crossover, and tree mutation. Individuals have tree_size ≥ 1 and
  may contain non-terminal function nodes.

- **GA** = genetic algorithm with parametric individuals, uniform or
  arithmetic crossover, and Gaussian or bit-flip mutation. Individuals
  are vectors, not trees.

- **PSO** = particle swarm optimization with velocity update.

If your experiment modifies the standard algorithm, describe the
modification explicitly. Do not reuse a standard label for a modified
algorithm.

## Experiment Registration Format

When adding a new experiment to `exp/USABLE.md`, use this template:

```
### Exp NN — Short Name
- **Runs**: N (seeds × conditions)
- **Boundary**: algorithm, parameters, representation, data split,
  seeds, λ, depth, etc.
- **buy_and_hold**: $X
- **Usable for**: what claims this experiment can support
- **Verified facts**: specific numbers that have been cross-checked
- **Limitations**: what this experiment cannot do
- **Not usable for**: explicit exclusions
```

## What To Do If You Discover a Pitfall

1. Add it to `exp/PITFALLS.md` with the "Do not X" format.
2. Audit existing experiments to see if they violate the new pitfall.
3. Update `exp/USABLE.md` boundary conditions for affected experiments.
4. Add CAUTION.md to any newly downgraded experiment.

## What To Do If You Want to Re-run an Existing Experiment

1. Read its entry in `exp/USABLE.md`.
2. Do not change boundary conditions (data range, λ, seeds) unless
   you are explicitly testing the effect of that change.
3. If you change boundary conditions, register it as a NEW experiment
   with a new number; do not overwrite the old one.
4. If your re-run contradicts the original, note the contradiction
   in USABLE.md and investigate why (seed sensitivity? code change?
   data drift?).

## Directory Structure

```
exp/
  AGENT.md          ← you are here
  PITFALLS.md       ← what not to do
  USABLE.md         ← boundary conditions
  archive/          ← D-grade experiments (moved here)
  NN-name/          ← active experiments
    CAUTION.md      ← present if C-grade
    results.json
    run.py
    report_zh.md
  analysis/         ← post-hoc analyses
```

## Final Rule

**If an experiment is not in USABLE.md, it does not exist for the
purpose of the paper.**
