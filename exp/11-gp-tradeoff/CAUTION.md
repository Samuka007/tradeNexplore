# CAUTION: Unreliable Experiment

## Status: C-GRADE — Superseded by multiseed validation

## Why
- Single-seed outlier dominates: 75-pop/20-gen gives $3,142 at seed=42
- Multiseed validation (exp/20-budget-multiseed, 10 seeds) shows mean=$1,506, std=$474 for same config
- The $3,142 result is not reproducible

## Boundary
- Seed=42 only
- Result is an outlier, not representative

## What you CAN use this for
- NOTHING. Use exp/20-budget-multiseed instead, but note that exp/20-budget-multiseed has its own data-range issues (BH=$5,660).
