#!/usr/bin/env Rscript
# Dimension validation tests for the optimization workflow

cat("Testing dimension compatibility...\n\n")

suppressPackageStartupMessages({
  library(Matrix)
  library(Rcpp)
})

# Compile C++ (suppress output)
sink("/dev/null")
sourceCpp("src/FISTA_w.cpp")
sink()

source("R/init_B.R")
source("R/projections.R")

# Test dimensions
N <- 50   # spots
K <- 5    # cell types
D <- 3    # components
m <- 100  # edges

cat(sprintf("Testing with N=%d, K=%d, D=%d, m=%d\n\n", N, K, D, m))

# Test 1: Create compatible data
cat("Test 1: Data creation\n")

set.seed(123)
S <- matrix(runif(N * K), N, K)
S <- sweep(S, 1, rowSums(S), "/")  # Normalize for realism

A <- sparseMatrix(
  i = c(1:m),
  j = sample(1:N, m, replace = TRUE),
  x = sample(c(-1, 1), m, replace = TRUE),
  dims = c(m, N)
)

B <- matrix(runif(K * D), K, D)
B <- sweep(B, 2, colSums(B), "/")  # Column normalize

w <- runif(m, 0.1, 0.5)

cat(sprintf("  S: %d x %d\n", nrow(S), ncol(S)))
cat(sprintf("  A: %d x %d\n", nrow(A), ncol(A)))
cat(sprintf("  B: %d x %d\n", nrow(B), ncol(B)))
cat(sprintf("  w: length %d\n", length(w)))

# Check dimensions
stopifnot(nrow(S) == N)
stopifnot(ncol(S) == K)
stopifnot(nrow(A) == m)
stopifnot(ncol(A) == N)
stopifnot(nrow(B) == K)
stopifnot(ncol(B) == D)
stopifnot(length(w) == m)

cat("  ✓ Dimensions consistent\n")

# Test 2: Check S-B compatibility
cat("\nTest 2: S-B compatibility\n")
stopifnot(ncol(S) == nrow(B))
cat("  ✓ ncol(S) == nrow(B)\n")

# Test 3: Check A-S compatibility
cat("\nTest 3: A-S compatibility\n")
stopifnot(ncol(A) == nrow(S))
cat("  ✓ ncol(A) == nrow(S)\n")

# Test 4: Check A-w compatibility
cat("\nTest 4: A-w compatibility\n")
stopifnot(nrow(A) == length(w))
cat("  ✓ nrow(A) == length(w)\n")

# Test 5: Run C++ function with these dimensions
cat("\nTest 5: C++ function call\n")

res <- xb_fista_wfixed(
  S, A, B, w,
  lambda = 1.0,
  delta = 1e-6,
  it_X = 10,
  it_B = 10,
  log_every_X = 0,
  log_every_B = 0
)

# Check output dimensions
stopifnot(nrow(res$X) == N)
stopifnot(ncol(res$X) == D)
stopifnot(nrow(res$B) == K)
stopifnot(ncol(res$B) == D)

cat(sprintf("  Output X: %d x %d\n", nrow(res$X), ncol(res$X)))
cat(sprintf("  Output B: %d x %d\n", nrow(res$B), ncol(res$B)))
cat("  ✓ Output dimensions correct\n")

# Test 6: Check simplex constraints
cat("\nTest 6: Simplex constraints\n")

X_rowsums <- rowSums(res$X)
B_colsums <- colSums(res$B)

cat(sprintf("  X row sums: [%.6f, %.6f]\n", min(X_rowsums), max(X_rowsums)))
cat(sprintf("  B col sums: [%.6f, %.6f]\n", min(B_colsums), max(B_colsums)))

stopifnot(all(abs(X_rowsums - 1.0) < 1e-3))
stopifnot(all(abs(B_colsums - 1.0) < 1e-3))

cat("  ✓ Simplex constraints satisfied\n")

# Test 7: Check non-negativity
cat("\nTest 7: Non-negativity\n")

cat(sprintf("  X min: %.6f\n", min(res$X)))
cat(sprintf("  B min: %.6f\n", min(res$B)))

stopifnot(all(res$X >= -1e-6))  # Allow tiny numerical errors
stopifnot(all(res$B >= -1e-6))

cat("  ✓ Non-negativity satisfied\n")

# Test 8: Objective value is finite
cat("\nTest 8: Objective value\n")

cat(sprintf("  obj: %.6f\n", res$obj))
stopifnot(is.finite(res$obj))
stopifnot(res$obj > 0)  # Should be positive (NLL + penalty)

cat("  ✓ Objective is finite and positive\n")

# Test 9: init_B dimension compatibility
cat("\nTest 9: init_B function\n")

B_init <- init_B(S, A, D = D, eps = 1e-3)

stopifnot(nrow(B_init) == K)
stopifnot(ncol(B_init) == D)

B_colsums <- colSums(B_init)
stopifnot(all(abs(B_colsums - 1.0) < 1e-3))

cat(sprintf("  B_init: %d x %d\n", nrow(B_init), ncol(B_init)))
cat("  ✓ init_B produces correct dimensions\n")

# Test 10: Different D values
cat("\nTest 10: Different D values\n")

for (D_test in c(2, 4, 5)) {
  B_test <- matrix(runif(K * D_test), K, D_test)
  B_test <- sweep(B_test, 2, colSums(B_test), "/")

  res_test <- xb_fista_wfixed(
    S, A, B_test, w,
    lambda = 1.0,
    it_X = 5,
    it_B = 5,
    log_every_X = 0,
    log_every_B = 0
  )

  stopifnot(ncol(res_test$X) == D_test)
  stopifnot(ncol(res_test$B) == D_test)

  cat(sprintf("  D=%d: ✓\n", D_test))
}

cat("\n========================================\n")
cat("All dimension tests passed! ✓\n")
cat("========================================\n")
