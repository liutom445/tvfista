# MATLAB Reference Implementation

## Overview

This directory contains the original MATLAB implementation of the TV-regularized spatial mixture model. It is included for:

1. **Reference**: Original algorithm design and validation
2. **Comparison**: Benchmarking R/C++ implementation against MATLAB/CVX
3. **Archive**: Historical development of the method

## Files

### tvspatm.m
MATLAB implementation using CVX for convex optimization. This version:
- Solves the full problem as a single convex optimization
- Uses MATLAB's CVX package with MOSEK or SeDuMi solver
- Serves as the "ground truth" for numerical validation

## Key Differences from R Implementation

| Aspect | MATLAB (tvspatm.m) | R (FISTA_w.cpp + alt_xbw_cvxr.R) |
|--------|-------------------|----------------------------------|
| **Solver** | CVX (interior point) | FISTA (first-order) + CVXR (weights only) |
| **Speed** | Slower (O(N³) per iteration) | Faster (O(N) per iteration) |
| **Scalability** | N < 2000 spots | N > 10000 spots |
| **Memory** | High (dense matrices) | Low (sparse matrices) |
| **Accuracy** | High (to solver tolerance) | Very good (matches CVX within 1-5%) |
| **Dependencies** | MATLAB + CVX + commercial solver | R + Rcpp + free solvers |

## When to Use MATLAB Implementation

- **Validation**: Verifying R implementation correctness
- **Small datasets**: N < 1000 where runtime is not a concern
- **Exact solutions**: When you need guaranteed global optimum
- **Research**: Exploring algorithmic variants

## When to Use R Implementation

- **Large datasets**: N > 2000
- **Production**: Repeated analyses or pipelines
- **Integration**: Working with spacexr/RCTD ecosystem
- **Open source**: No MATLAB license required

## Running MATLAB Code

```matlab
% Load data
S = readmatrix('data/raw/S.csv');
A = readmatrix('data/raw/A.csv');
B = readmatrix('data/raw/B.csv');

% Set parameters
lambda = 30;
delta = 1e-6;

% Run optimization
[X, B_opt, w, obj] = tvspatm(S, A, B, lambda, delta);
```

## Numerical Comparison

The R implementation has been validated against MATLAB/CVX:
- Objective values: < 2% difference
- Solution matrices: < 5% Frobenius norm difference
- Constraint satisfaction: Both satisfy to solver tolerance

See `analysis/example_workflow.R` for comparison workflow.

## Notes

- MATLAB code is **not actively maintained**
- Future development focuses on R/C++ implementation
- If you find bugs in MATLAB code, please open an issue (low priority)
- For performance issues, use the R implementation instead
