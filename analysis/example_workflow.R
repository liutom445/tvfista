#!/usr/bin/env Rscript
# Example workflow for TV-regularized spatial mixture modeling
# This script demonstrates end-to-end usage from RCTD data to visualization

# ============================================================================
# SETUP
# ============================================================================

# Load libraries
suppressPackageStartupMessages({
  library(spacexr)
  library(interp)
  library(Matrix)
  library(fields)
  library(Rcpp)
  library(ggplot2)
  library(viridis)
  library(CVXR)
})

# Source R functions
source("R/init_A.R")
source("R/init_B.R")
source("R/alt_xbw_cvxr.R")
source("R/evaluation.R")
source("R/visualization.R")

# Load configuration
source("analysis/config.R")

# Compile C++ implementation
Rcpp::sourceCpp("src/FISTA_w.cpp")

# Set seed
set.seed(SEED)

# ============================================================================
# LOAD OR GENERATE DATA
# ============================================================================

# Option 1: Load pre-computed data
cat("Loading data...\n")
S_mat <- as.matrix(read.csv("data/raw/S.csv", header = TRUE, row.names = 1))
A_mat <- as.matrix(read.csv("data/raw/A.csv", header = TRUE, row.names = 1))
B_init <- as.matrix(read.csv("data/raw/B.csv", header = TRUE, row.names = 1))

# Convert A to sparse format
A <- as(A_mat, "dgCMatrix")

cat(sprintf("Dimensions: N=%d, K=%d, D=%d, m=%d\n",
            nrow(S_mat), ncol(S_mat), ncol(B_init), nrow(A)))

# Option 2: Generate from RCTD (uncomment if starting from RCTD object)
# myRCTD <- readRDS("path/to/myRCTD.rds")
#
# # Extract S from RCTD
# n <- myRCTD@cell_type_info$info[[3]]
# S_mat <- matrix(0, nrow(myRCTD@results$results_df), n)
# rownames(S_mat) <- rownames(myRCTD@results$results_df)
# colnames(S_mat) <- myRCTD@cell_type_info$info[[2]]
#
# for (i in seq_len(nrow(S_mat))) {
#   sc <- myRCTD@results$singlet_scores[[i]]
#   S_mat[i, names(sc)] <- exp((min(sc) - sc) / SCALE)
# }
#
# # Build spatial graph
# Xcoords <- myRCTD@spatialRNA@coords[rownames(S_mat), ]
# A <- init_A(Xcoords, method = GRAPH_METHOD, spatial_weight = SPATIAL_WEIGHT)
#
# # Initialize dictionary
# B_init <- init_B(S_mat, A, D = D, eps = EPS_PROJ)

# ============================================================================
# DIMENSION CHECKS
# ============================================================================

cat("\nVerifying dimensions...\n")
cat(sprintf("  S: %d x %d\n", nrow(S_mat), ncol(S_mat)))
cat(sprintf("  A: %d x %d (sparse: %s)\n", nrow(A), ncol(A), class(A)[1]))
cat(sprintf("  B_init: %d x %d\n", nrow(B_init), ncol(B_init)))

stopifnot(ncol(S_mat) == nrow(B_init))
stopifnot(ncol(A) == nrow(S_mat))

# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

cat("\n========================================\n")
cat("Starting optimization\n")
cat("========================================\n")
cat(sprintf("Lambda: %.2f\n", LAMBDA))
cat(sprintf("Delta: %.2e\n", DELTA))
cat(sprintf("Weight bounds: [%.2f, %.2f]\n", WMIN, WMAX))
cat(sprintf("Outer iterations: %d\n", OUTER_ITER))
cat(sprintf("Inner iterations: X=%d, B=%d\n", IT_X, IT_B))
cat("========================================\n\n")

start_time <- Sys.time()

res <- alt_xbw_cvxr(
  S_mat = S_mat,
  A = A,
  B_init = B_init,
  lambda = LAMBDA,
  delta = DELTA,
  outer_it = OUTER_ITER,
  it_X = IT_X,
  it_B = IT_B,
  eps_proj = EPS_PROJ,
  eps_step = EPS_STEP,
  safety = SAFETY,
  log_every = LOG_EVERY,
  wmin = WMIN,
  wmax = WMAX
)

end_time <- Sys.time()
runtime <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\n========================================\n"))
cat(sprintf("Optimization complete in %.2f seconds\n", runtime))
cat(sprintf("========================================\n\n"))

# Extract results
X_opt <- res$X
B_opt <- res$B
w_opt <- res$w
hist <- res$hist

# ============================================================================
# EVALUATE RESULTS
# ============================================================================

cat("Evaluation metrics:\n")
metrics <- eval_run(S_mat, A, X_opt, B_opt, w_opt, lambda = LAMBDA, delta = DELTA)
print(round(metrics, 4))
cat("\n")

# Weight statistics
cat("Weight statistics:\n")
cat(sprintf("  min: %.4f\n", min(w_opt)))
cat(sprintf("  max: %.4f\n", max(w_opt)))
cat(sprintf("  mean: %.4f\n", mean(w_opt)))
cat(sprintf("  median: %.4f\n", median(w_opt)))
cat("\n")

# Constraint satisfaction
cat("Constraint checks:\n")
X_rowsums <- rowSums(X_opt)
B_colsums <- colSums(B_opt)
cat(sprintf("  X row sums: [%.6f, %.6f] (should be 1.0)\n",
            min(X_rowsums), max(X_rowsums)))
cat(sprintf("  X min value: %.6f (should be >= %.3f)\n", min(X_opt), EPS_PROJ))
cat(sprintf("  B col sums: [%.6f, %.6f] (should be 1.0)\n",
            min(B_colsums), max(B_colsums)))
cat(sprintf("  B min value: %.6f (should be >= %.3f)\n\n", min(B_opt), EPS_PROJ))

# ============================================================================
# SAVE RESULTS
# ============================================================================

cat("Saving results...\n")
write.csv(X_opt, "analysis/outputs/x_weighted.csv", row.names = TRUE)
write.csv(B_opt, "analysis/outputs/B_weighted.csv", row.names = TRUE)
write.csv(w_opt, "analysis/outputs/w_weighted.csv", row.names = FALSE)
write.csv(hist, "analysis/outputs/convergence_history.csv", row.names = FALSE)

# Save metrics
metrics_df <- data.frame(
  metric = names(metrics),
  value = as.numeric(metrics)
)
write.csv(metrics_df, "analysis/outputs/metrics.csv", row.names = FALSE)

cat("  - x_weighted.csv\n")
cat("  - B_weighted.csv\n")
cat("  - w_weighted.csv\n")
cat("  - convergence_history.csv\n")
cat("  - metrics.csv\n\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

cat("Generating plots...\n")

# 1. Convergence plot
p_conv <- plot_convergence(hist, title = sprintf("Convergence (lambda=%.1f)", LAMBDA))
ggsave("analysis/outputs/plots/convergence.png", p_conv, width = 8, height = 5)
cat("  - convergence.png\n")

# 2. Weight distribution
p_weights <- plot_weights(w_opt, wmin = WMIN, wmax = WMAX)
ggsave("analysis/outputs/plots/weights.png", p_weights, width = 8, height = 5)
cat("  - weights.png\n")

# 3. Spatial maps (if RCTD object available)
# Uncomment and modify if you have myRCTD loaded:
#
# p_types <- plot_types(X_opt, B_opt, myRCTD,
#                       title = sprintf("TV-weighted (lambda=%.1f)", LAMBDA))
# ggsave("analysis/outputs/plots/spatial_map.png", p_types, width = 10, height = 8)
# cat("  - spatial_map.png\n")

cat("\nDone!\n")

# ============================================================================
# SUMMARY
# ============================================================================

cat("\n========================================\n")
cat("SUMMARY\n")
cat("========================================\n")
cat(sprintf("Spots: %d\n", nrow(S_mat)))
cat(sprintf("Cell types: %d\n", ncol(S_mat)))
cat(sprintf("Components: %d\n", ncol(B_opt)))
cat(sprintf("Edges: %d\n", nrow(A)))
cat(sprintf("Final objective: %.6f\n", tail(hist$obj, 1)))
cat(sprintf("Neighbor agreement: %.2f%%\n", 100 * metrics["neigh_agree"]))
cat(sprintf("Runtime: %.2f min\n", runtime / 60))
cat("========================================\n")
