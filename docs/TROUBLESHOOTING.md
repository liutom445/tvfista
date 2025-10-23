# Troubleshooting Guide

## Common Issues and Solutions

### 1. C++ Compilation Errors

#### Error: "R.h not found"

**Cause**: R headers not installed

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install r-base-dev

# macOS (should be automatic, but if needed)
brew reinstall r
```

#### Error: "Rcpp.h not found"

**Cause**: Rcpp not properly installed

**Solution**:
```r
# Reinstall Rcpp
remove.packages("Rcpp")
install.packages("Rcpp")

# Verify
library(Rcpp)
Rcpp::evalCpp("2 + 2")
```

#### Error: "undefined reference to __gxx_personality_v0"

**Cause**: C++ linker issue (Windows/Rtools)

**Solution**:
```r
# Check Rtools installation
Sys.which("make")
Sys.which("g++")

# Should return paths like C:/rtools43/mingw64/bin/make.exe
# If not, reinstall Rtools and check "Add to PATH"
```

### 2. CVXR Solver Failures

#### Error: "CVXR w-step failed; keeping previous w"

**Cause**: LP infeasible or solver issue

**Diagnosis**:
```r
# Check weight bounds
wmin <- 0.01
wmax <- 0.50
stopifnot(wmin > 0 && wmin < wmax)

# Check constraint
cvec <- as.numeric(rowSums(abs(A %*% X)))
sum(cvec)  # Should be > 0
```

**Solution**:
1. **Relax bounds**: Try `wmin = 0.001, wmax = 1.0`
2. **Check X**: Ensure no NaN/Inf values: `any(!is.finite(X))`
3. **Increase tolerance**: Add `prob <- Problem(...); solve(prob, feastol=1e-6)`
4. **Use different solver**:
   ```r
   solve(prob, solver = "SCS")  # Alternative to ECOS
   ```

#### Error: "Solver 'ECOS' not available"

**Cause**: ECOS not installed

**Solution**:
```r
# Reinstall CVXR (should include ECOS)
install.packages("CVXR")

# If still fails, install ECOSolveR separately
install.packages("ECOSolveR")
```

### 3. Memory Issues

#### Error: "Cannot allocate vector of size X GB"

**Cause**: Insufficient memory

**Solution**:

**Short-term fixes**:
```r
# 1. Reduce data size
S_sub <- S[1:1000, ]  # Subsample spots
A_sub <- A[, 1:1000]  # Match incidence matrix

# 2. Use sparse matrices consistently
A <- as(A, "dgCMatrix")

# 3. Reduce iterations
res <- alt_xbw_cvxr(S, A, B, outer_it=5, it_X=500, it_B=200)

# 4. Clear workspace
rm(list=setdiff(ls(), c("S", "A", "B")))
gc()
```

**Long-term fixes**:
- Increase system RAM
- Use compute server
- Implement batch processing for large datasets

#### Memory leak / increasing memory usage

**Diagnosis**:
```r
# Monitor memory
library(pryr)
for (i in 1:10) {
  res <- alt_xbw_cvxr(S, A, B, outer_it=1, log_every=0)
  cat(sprintf("Iter %d: %.1f MB\n", i, mem_used() / 1e6))
}
```

**Solution**:
```r
# Explicitly clean up after runs
res <- alt_xbw_cvxr(...)
X <- res$X; B <- res$B; w <- res$w
rm(res)
gc()
```

### 4. Convergence Problems

#### Issue: Objective increases instead of decreases

**Cause**: Step size too large or Lipschitz bound underestimated

**Solution**:
```r
# Reduce safety factor (more conservative steps)
res <- alt_xbw_cvxr(S, A, B, safety=0.5)

# Increase epsilon (more stable simplex)
res <- alt_xbw_cvxr(S, A, B, eps_proj=1e-3)
```

#### Issue: NaN or Inf in results

**Diagnosis**:
```r
# Check input data
any(is.na(S))
any(!is.finite(S))
any(S < 0)

# Check initialization
any(is.na(B))
any(!is.finite(B))
colSums(B)  # Should all be 1.0
```

**Solution**:
1. **Clean data**:
   ```r
   S[!is.finite(S)] <- 0
   S <- pmax(S, 0)  # Ensure non-negative
   ```

2. **Re-initialize B**:
   ```r
   source("R/init_B.R")
   B <- init_B(S, A, D=3)
   ```

3. **Check delta and epsilon**:
   ```r
   # Avoid delta too small (numerical issues)
   delta <- 1e-6  # Not smaller than 1e-8

   # Avoid epsilon too large (degeneracy)
   eps_proj <- 1e-3  # Not larger than 1e-2
   ```

#### Issue: Slow convergence

**Diagnosis**:
```r
# Plot convergence
plot(res$hist$obj, type="b", log="y")
```

**Solution**:
1. **Increase iterations**:
   ```r
   res <- alt_xbw_cvxr(S, A, B, it_X=5000, it_B=3000)
   ```

2. **Tune lambda** (may be too large):
   ```r
   # Try smaller lambda
   res <- alt_xbw_cvxr(S, A, B, lambda=10)
   ```

3. **Better initialization**:
   ```r
   # Use previous result as warm start
   res2 <- alt_xbw_cvxr(S, A, B, X0_init=res$X)
   ```

### 5. Performance Issues

#### Issue: Very slow optimization

**Diagnosis**:
```r
# Profile code
Rprof("profile.out")
res <- alt_xbw_cvxr(S, A, B, outer_it=2)
Rprof(NULL)
summaryRprof("profile.out")
```

**Solution**:

1. **Disable logging**:
   ```r
   res <- alt_xbw_cvxr(S, A, B, log_every=0)
   ```

2. **Check BLAS**:
   ```r
   sessionInfo()
   # Look for optimized BLAS (OpenBLAS, MKL, Accelerate)
   ```

3. **Sparse matrices**:
   ```r
   # Ensure A is sparse
   class(A)  # Should be "dgCMatrix"
   A <- as(A, "dgCMatrix")
   ```

4. **Reduce graph density** (if using MST or dense Delaunay):
   ```r
   # Use sparser graph
   A <- init_A(coords, method="mst")  # N-1 edges instead of ~3N
   ```

### 6. Constraint Violations

#### Issue: X rows don't sum to 1

**Diagnosis**:
```r
rowsums <- rowSums(res$X)
summary(rowsums)
max(abs(rowsums - 1))
```

**Acceptable**: < 1e-6

**If larger**:
```r
# Check C++ projection function
X_test <- matrix(runif(15), 5, 3)
X_proj <- t(apply(X_test, 1, function(x) {
  # Simplex projection code from projections.R
  source("R/projections.R")
  proj_simplex(x)
}))
rowSums(X_proj)  # Should all be 1.0
```

#### Issue: Negative values in X or B

**Diagnosis**:
```r
min(res$X)
min(res$B)
```

**Should be**: ≥ eps_proj (e.g., 0.001)

**If negative**:
- Check C++ projection implementation
- Re-compile: `sourceCpp("src/FISTA_w.cpp", rebuild=TRUE)`

### 7. Data Issues

#### Issue: "ncol(S) != nrow(B)"

**Cause**: Dimension mismatch

**Solution**:
```r
# Check dimensions
cat(sprintf("S: %d x %d\n", nrow(S), ncol(S)))
cat(sprintf("B: %d x %d\n", nrow(B), ncol(B)))

# B should be K x D where K = ncol(S)
# Re-initialize if needed
B <- init_B(S, A, D=3)
```

#### Issue: "ncol(A) != nrow(S)"

**Cause**: A and S have different number of spots

**Solution**:
```r
# Subset A to match S
barcodes <- rownames(S)
# A should have columns corresponding to barcodes
# If not, rebuild A
A <- init_A(coords[barcodes, ])
```

### 8. Visualization Issues

#### Error: "myRCTD object not found"

**Cause**: Plotting functions require RCTD object for coordinates

**Solution**:

**Option 1**: Load RCTD object
```r
myRCTD <- readRDS("path/to/myRCTD.rds")
plot_types(res$X, res$B, myRCTD, "Title")
```

**Option 2**: Use generic plotting
```r
# Plot mixture components directly
library(ggplot2)
df <- data.frame(
  x = coords$x,
  y = coords$y,
  cluster = max.col(res$X)
)
ggplot(df, aes(x, y, color=factor(cluster))) +
  geom_point()
```

#### Issue: Plot looks noisy/random

**Cause**: Insufficient regularization

**Solution**:
```r
# Increase lambda for more spatial smoothness
res <- alt_xbw_cvxr(S, A, B, lambda=50)

# Or check neighbor agreement
source("R/evaluation.R")
metrics <- eval_run(S, A, res$X, res$B, res$w, lambda=30)
metrics["neigh_agree"]  # Should be > 0.5
```

### 9. Platform-Specific Issues

#### macOS: "xcrun: error: invalid active developer path"

**Cause**: Xcode Command Line Tools not installed

**Solution**:
```bash
xcode-select --install
```

#### Windows: "make not found"

**Cause**: Rtools not in PATH

**Solution**:
1. Reinstall Rtools
2. Check "Add Rtools to PATH" during installation
3. Restart R
4. Verify: `Sys.which("make")`

#### Linux: "cannot find -lstdc++"

**Cause**: C++ standard library not found

**Solution**:
```bash
sudo apt-get install libstdc++-11-dev
```

### 10. Getting More Help

If your issue persists:

1. **Check existing issues**: [GitHub Issues](https://github.com/yourusername/tv-spatial-mixture/issues)

2. **Post new issue** with:
   ```r
   # System info
   sessionInfo()

   # Dimensions
   cat(sprintf("S: %d x %d\n", nrow(S), ncol(S)))
   cat(sprintf("A: %d x %d (class: %s)\n", nrow(A), ncol(A), class(A)[1]))
   cat(sprintf("B: %d x %d\n", nrow(B), ncol(B)))

   # Compilation test
   Rcpp::evalCpp("2 + 2")
   ```

3. **Provide minimal reproducible example**:
   ```r
   # Example that triggers the error
   set.seed(123)
   S <- matrix(runif(100*5), 100, 5)
   # ... minimal code to reproduce
   ```

4. **Include error message** (full text)

5. **What you've tried** (so we don't suggest the same things)

## Quick Diagnostic Checklist

Run this to check for common issues:

```r
# Diagnostic script
cat("=== TV Spatial Mixture Diagnostic ===\n\n")

# 1. R version
cat("R version:", R.version.string, "\n")

# 2. Compiler
cat("C++ compiler:", system("g++ --version", intern=TRUE)[1], "\n")

# 3. Rcpp
cat("Rcpp version:", packageVersion("Rcpp"), "\n")
cat("Rcpp test:", Rcpp::evalCpp("2 + 2"), "\n")

# 4. Key packages
for (pkg in c("Matrix", "CVXR", "ggplot2")) {
  cat(sprintf("%s: %s\n", pkg,
              if (requireNamespace(pkg, quietly=TRUE))
                packageVersion(pkg) else "NOT INSTALLED"))
}

# 5. FISTA compilation
cat("FISTA compiled:", exists("xb_fista_wfixed"), "\n")

# 6. Memory
cat("Available memory:", pryr::mem_used() / 1e9, "GB\n")

# 7. BLAS
cat("BLAS:", sessionInfo()$BLAS, "\n")

cat("\n=== End Diagnostic ===\n")
```

Save this output when reporting issues.
