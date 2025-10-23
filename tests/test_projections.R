#!/usr/bin/env Rscript
# Unit tests for simplex projection functions

cat("Testing simplex projection functions...\n\n")

source("R/projections.R")

# Test 1: proj_simplex basic functionality
cat("Test 1: Basic simplex projection\n")
v <- c(0.2, 0.5, 0.3)
p <- proj_simplex(v, K = 3, eps = 1e-4)

# Check sum = 1
stopifnot(abs(sum(p) - 1.0) < 1e-6)

# Check all >= eps
stopifnot(all(p >= 1e-4 - 1e-10))

cat("  ✓ Basic projection passed\n")

# Test 2: Extreme values
cat("Test 2: Extreme values\n")
v_extreme <- c(100, 0, 0)
p_extreme <- proj_simplex(v_extreme, eps = 1e-3)

stopifnot(abs(sum(p_extreme) - 1.0) < 1e-6)
stopifnot(all(p_extreme >= 1e-3 - 1e-10))
stopifnot(p_extreme[1] > 0.9)  # Should mostly be first element

cat("  ✓ Extreme values passed\n")

# Test 3: Uniform distribution
cat("Test 3: Uniform distribution\n")
v_uniform <- rep(1, 5)
p_uniform <- proj_simplex(v_uniform, eps = 1e-3)

stopifnot(abs(sum(p_uniform) - 1.0) < 1e-6)
stopifnot(all(abs(p_uniform - 1/5) < 1e-6))  # Should all be 1/5

cat("  ✓ Uniform distribution passed\n")

# Test 4: Matrix row projection
cat("Test 4: Matrix row projection\n")
X <- matrix(runif(15), 5, 3)
X_proj <- proj_rows_simplex(X, eps = 1e-4)

# Check all rows sum to 1
rowsums <- rowSums(X_proj)
stopifnot(all(abs(rowsums - 1.0) < 1e-6))

# Check all elements >= eps
stopifnot(all(X_proj >= 1e-4 - 1e-10))

cat("  ✓ Matrix row projection passed\n")

# Test 5: Matrix column projection
cat("Test 5: Matrix column projection\n")
B <- matrix(runif(12), 4, 3)
B_proj <- proj_cols_simplex(B, eps = 1e-4)

# Check all columns sum to 1
colsums <- colSums(B_proj)
stopifnot(all(abs(colsums - 1.0) < 1e-6))

# Check all elements >= eps
stopifnot(all(B_proj >= 1e-4 - 1e-10))

cat("  ✓ Matrix column projection passed\n")

# Test 6: Idempotence (projecting twice = projecting once)
cat("Test 6: Idempotence\n")
v <- runif(4)
p1 <- proj_simplex(v, eps = 1e-4)
p2 <- proj_simplex(p1, eps = 1e-4)

stopifnot(all(abs(p1 - p2) < 1e-6))

cat("  ✓ Idempotence passed\n")

# Test 7: Different epsilon values
cat("Test 7: Different epsilon values\n")
v <- c(0.5, 0.3, 0.2)

p_small <- proj_simplex(v, eps = 1e-5)
p_large <- proj_simplex(v, eps = 1e-2)

stopifnot(abs(sum(p_small) - 1.0) < 1e-6)
stopifnot(abs(sum(p_large) - 1.0) < 1e-6)
stopifnot(all(p_small >= 1e-5 - 1e-10))
stopifnot(all(p_large >= 1e-2 - 1e-10))

cat("  ✓ Different epsilon values passed\n")

cat("\n========================================\n")
cat("All projection tests passed! ✓\n")
cat("========================================\n")
