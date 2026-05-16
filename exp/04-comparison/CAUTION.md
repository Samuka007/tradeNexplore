# CAUTION: Unreliable Experiment

## Status: C-GRADE — Do not use for quantitative claims

## Why
- Single-seed only (seed=42)
- PSO+MACD test=$948 is likely noise (single seed, no verification)
- PSO+trivial_sma test=$1,847 vs PSO+position_sma baseline=$2,297: difference not reproducible
- GP_extended_no_penalty tree_size=96, test=$1,738 — single seed

## Boundary
- Seed=42 only
- No cross-seed validation

## What you CAN use this for
- Hypothesis generation about dimensionality effect (7D MACD vs 3D position_sma)
- Must verify with multi-seed before any claim
