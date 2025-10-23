# C++ Source Code

## Overview

This directory contains the core C++ optimization routines for the TV-regularized spatial mixture model.

## Files

### FISTA_w.cpp
Main C++ implementation with Rcpp bindings. Includes:
- **xb_fista_wfixed()**: Alternating FISTA for X and B with fixed weights w
- **eval_softL1_w_cpp()**: Evaluate soft-L1 TV penalty (for diagnostics)

## Compilation

### Using Rcpp::sourceCpp (Recommended)

```r
library(Rcpp)
sourceCpp("src/FISTA_w.cpp")
```

This automatically handles compilation and exports the functions to R.

### Using R CMD SHLIB (Manual)

```bash
R CMD SHLIB FISTA_w.cpp
```

Then load in R:
```r
dyn.load("FISTA_w.so")  # or .dll on Windows
```

### Using CMake (Advanced)

For building a standalone shared library:

```bash
mkdir build && cd build
cmake ..
make
```

See `CMakeLists.txt` for configuration options.

## Function Reference

### xb_fista_wfixed()

**Signature**:
```cpp
List xb_fista_wfixed(
  NumericMatrix S,         // Score matrix (N x K)
  S4 A,                    // Sparse incidence (m x N)
  NumericMatrix B,         // Initial dictionary (K x D)
  NumericVector w,         // Edge weights (length m)
  double lambda = 1.0,     // TV regularization
  double delta = 1e-6,     // Soft-L1 smoothing
  int it_X = 1500,         // X iterations
  double eps_proj_X = 1e-4,
  double eps_step_X = 1e-2,
  bool normalize_for_bound = true,
  double safety_X = 0.6,
  int log_every_X = 1000,
  int it_B = 500,          // B iterations
  double eps_proj_B = 1e-4,
  double eps_step_B = 1e-2,
  double safety_B = 0.6,
  int log_every_B = 1000,
  bool update_B = true,
  Nullable<NumericMatrix> X0_opt = R_NilValue
)
```

**Returns**: List with
- `X`: Optimized mixture matrix (N x D)
- `B`: Optimized dictionary (K x D)
- `obj`: Final objective value

**Algorithm**: FISTA with momentum (Nesterov acceleration)

### eval_softL1_w_cpp()

**Signature**:
```cpp
double eval_softL1_w_cpp(
  S4 A,                    // Sparse incidence
  NumericMatrix X,         // Mixture matrix
  NumericVector w,         // Edge weights
  double delta = 1e-6
)
```

**Returns**: Weighted soft-L1 TV penalty value

## Implementation Details

### Key Features

1. **Sparse matrix operations**: Uses `Eigen::SparseMatrix` for efficient A^T A operations
2. **In-place updates**: Minimizes memory allocations
3. **Vectorized operations**: Uses Eigen's array operations
4. **Lipschitz bound**: Adaptive step size based on data and penalty bounds

### Simplex Projection

Euclidean projection onto floor-simplex Δ^{D-1}_ε:
```cpp
// Algorithm (O(D log D) via sorting)
1. Shift by epsilon: v = x - ε
2. Sort positive part: u = sort(max(v, 0))
3. Find threshold via cumulative sum
4. Project: x_new = max(v - θ, 0) + ε
```

### FISTA Momentum

Standard Nesterov acceleration:
```cpp
omega = (k - 1.0) / (k + 2.0)
Y = X + omega * (X - X_old)
```

where k is the iteration counter.

### Lipschitz Constant

**Data term**:
```cpp
L_data ≈ ||S||_max · ||B||_F² / ε_min
```

**Penalty term**:
```cpp
L_pen ≤ λ · ||A^T A||_2 · max(w) / √δ
```

**Total**:
```cpp
L = L_data + L_pen
η = safety / L
```

## Performance Considerations

### Compilation Flags

For production use, compile with optimizations:
```bash
PKG_CXXFLAGS="-O3 -march=native -ffast-math" R CMD SHLIB FISTA_w.cpp
```

**Warning**: `-ffast-math` may reduce numerical precision slightly.

### Memory Usage

Per iteration:
- Gradients: O(N·D + K·D)
- Sparse matrix multiply: O(nnz(A)·D) where nnz(A) ≈ 6N for Delaunay
- Total: ~10-20 MB for N=1000, K=10, D=3

### Parallelization

Currently single-threaded. Potential parallelization points:
- Row-wise simplex projection (OpenMP)
- Matrix-matrix products (threaded BLAS)

Future versions may add OpenMP pragmas.

## Debugging

### Enable verbose output

Set `log_every_X = 100` to print progress:
```r
res <- xb_fista_wfixed(S, A, B, w, log_every_X = 100)
```

### Check for NaN/Inf

```r
# After optimization
any(!is.finite(res$X))
any(!is.finite(res$B))
```

If TRUE, likely issues:
- Delta too small (< 1e-8)
- Step size too large (reduce safety)
- Input data has NaN/Inf

### Profile C++ code

Use Rcpp sugar profiling or external tools like Valgrind:
```bash
R -d valgrind --vanilla < test_script.R
```

## Testing

Unit tests for C++ functions:
```r
# Test simplex projection
X <- matrix(runif(15), 5, 3)
# (Projection tested indirectly via optimization)

# Test soft-L1 evaluation
tv <- eval_softL1_w_cpp(A, X, w, delta = 1e-6)
stopifnot(tv > 0)
```

## Dependencies

- **Rcpp**: >= 1.0.0
- **RcppEigen**: >= 0.3.3 (header-only, auto-installed with Rcpp)
- **C++ Standard**: C++11 or later

## Modifications

If you modify the C++ code:

1. **Test compilation**:
   ```r
   Rcpp::sourceCpp("src/FISTA_w.cpp", rebuild = TRUE)
   ```

2. **Check exports**:
   ```r
   exists("xb_fista_wfixed")
   exists("eval_softL1_w_cpp")
   ```

3. **Run validation**:
   ```r
   source("tests/test_dimensions.R")
   ```

4. **Benchmark**:
   ```r
   library(microbenchmark)
   microbenchmark(
     xb_fista_wfixed(S, A, B, w, it_X=100, it_B=50),
     times = 10
   )
   ```

## References

- [Rcpp documentation](https://www.rcpp.org/)
- [Eigen documentation](https://eigen.tuxfamily.org/)
- [FISTA paper](https://doi.org/10.1137/080716542) (Beck & Teboulle, 2009)
