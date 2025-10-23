#!/usr/bin/env Rscript
# Minimal B initialization via graph-smoothed K-means
# ~50 lines | Requires: Matrix package | Input: S (N×K), A (m×N), D (int)

init_B <- function(S, A, D = 3, alpha = 0.3, eps = 1e-3, seed = 123) {
  #' Initialize dictionary B from scores S and adjacency A
  #' @param S: Score matrix (N spots × K cell types)
  #' @param A: Incidence matrix (m edges × N spots)
  #' @param D: Number of components (default: 3)
  #' @param alpha: Diffusion strength (default: 0.3)
  #' @param eps: Floor constraint (default: 1e-3)
  #' @param seed: Random seed (default: 123)
  #' @return B: Dictionary matrix (K × D) with columns on simplex
  
  require(Matrix)
  set.seed(seed)
  
  K <- ncol(S)
  
  # Build row-normalized adjacency from incidence
  Adj <- abs(crossprod(A))
  diag(Adj) <- 0
  row_sums <- Matrix::rowSums(Adj)
  row_sums[row_sums == 0] <- 1
  W <- Diagonal(x = 1/row_sums) %*% Adj
  
  # One-step graph diffusion smoothing
  S_smooth <- (1 - alpha) * S + alpha * (W %*% S)
  
  # K-means clustering
  km <- kmeans(as.matrix(S_smooth), centers = D, nstart = 25, iter.max = 300)
  B_raw <- t(km$centers)  # K × D
  
  # Project columns to simplex: sum=1, elements≥eps
  proj_col <- function(v) {
    s <- 1 - K * eps
    u <- sort(pmax(v - eps, 0), decreasing = TRUE)
    cssv <- cumsum(u) - s
    rho <- max(which(u > cssv / seq_along(u)))
    theta <- cssv[rho] / rho
    pmax(v - eps - theta, 0) + eps
  }
  
  B <- apply(B_raw, 2, proj_col)
  
  # Verify (optional)
  if (any(abs(colSums(B) - 1) > 1e-6)) 
    warning("Column sums: ", paste(round(colSums(B), 4), collapse=", "))
  
  return(B)
}

# Usage:
# B_init <- init_B(S_mat, A, D = 3)
