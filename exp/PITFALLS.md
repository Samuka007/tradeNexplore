# PITFALLS — First-Principles List of What Not To Do

This document lists pitfalls discovered through audit of 20+ experiments.
Each item states a negative ("Do not X") derived from observed failures,
without prescribing what should be done instead.

---

## Data Integrity

- Do not mix results from different train/test splits.
  The baseline buy_and_hold = $2,169.58 is only valid for pre-2020 train /
  2020-2022 test. Any experiment with a different date range produces a
  different BH and is not numerically comparable.

- Do not silently use a buy_and_hold value that differs from $2,169.58
  without verifying the data range changed.

- Do not compare fitness values across different evaluation functions
  (e.g., raw return vs parsimony-penalized fitness vs robust-opt
  worst-case fitness).

- Do not compare results from different λ values as if they came from
  the same experimental condition.

## Statistical Validity

- Do not use single-seed results to support claims about an algorithm's
  mean performance, variance, win rate, or typical behavior.

- Do not report a "best found" value as representative of an algorithm's
  performance distribution.

- Do not compute a standard deviation from n < 5 and present it as a
  stable estimate.

- Do not apply Bonferroni correction without also reporting uncorrected
  p-values and the exact number of comparisons.

## Algorithm Labeling

- Do not label an algorithm as "GP" if individuals have tree_size = 1,
  contain no function nodes, and use uniform crossover on a parameter
  vector. That algorithm is GA with tournament selection, not GP.

- Do not claim an experiment "restricts GP to PSO's representation" if
  the algorithmic mechanism was also changed (from tree-based search to
  parametric search).

## Experimental Design

- Do not compare two algorithms on different search spaces and attribute
  performance differences to algorithmic superiority.

- Do not change multiple variables simultaneously (representation +
  algorithm + regularization) and claim to have isolated a single effect.

- Do not assume a grid-search point with anomalously high test return
  but low train return represents a genuine basin. It may be a market
  event artifact.

- Do not claim a warm-start uses X% human rules without reading the
  actual initialization code to verify the fraction.

## Initialization and Bias

- Do not report warm-start mean performance without also reporting
  solution diversity (e.g., tree structure overlap across seeds).

- Do not treat common-mode convergence (multiple seeds finding identical
  or near-identical solutions) as evidence of algorithm reliability.

## Representation and Structure

- Do not compare PSO's 3D parametric results with GP's unrestricted
  tree results as if they address the same optimization problem.

- Do not claim GP "discovers structure" without a control experiment
  that restricts GP to the same structural family PSO uses.

- Do not claim hybridization "improves" or "harms" performance from a
  single tree or single seed.

## Protocol Interactions

- Do not compare walk-forward results with single-split results as if
  the only variable changed was the protocol. The effective fitness
  landscape changes when windows are averaged.

- Do not compare robust-optimization results (min-max over 52 windows)
  with single-split results without noting the objective function
  changed from mean to worst-case.
