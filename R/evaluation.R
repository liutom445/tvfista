#!/usr/bin/env Rscript
# Model evaluation and diagnostics

#' Evaluate optimization run
#'
#' @param S_mat Score matrix (N × K)
#' @param A Incidence matrix (m × N)
#' @param X Mixture matrix (N × D)
#' @param B Dictionary (K × D)
#' @param w Edge weights (length m)
#' @param lambda Regularization parameter
#' @param delta Soft-L1 smoothing (default: 1e-6)
#'
#' @return Named vector with: obj, nll, TVsoft1, neigh_agree, entropy, sharp
#'
#' @details
#' Computes:
#' - obj: Total objective value
#' - nll: Negative log-likelihood
#' - TVsoft1: Soft-L1 TV penalty
#' - neigh_agree: Fraction of edges with matching cluster assignments
#' - entropy: Normalized row entropy (measures mixing)
#' - sharp: Mean max probability per row (measures certainty)
#'
eval_run <- function(S_mat, A, X, B, w, lambda, delta = 1e-6) {
  # Data term (negative log-likelihood)
  SB   <- S_mat %*% B                # N x D
  den  <- pmax(rowSums(SB * X), 1e-12)
  nll  <- -sum(log(den))

  # TV penalty (soft-L1)
  U    <- as.matrix(A %*% X)         # m x D
  tv   <- sum(as.numeric(w) * rowSums(sqrt(U^2 + delta)))

  # Total objective
  obj  <- nll + lambda * tv

  # Cluster assignments
  cls  <- max.col(X, ties.method = "first")

  # Neighbor agreement (fraction of edges connecting same cluster)
  Edges <- summary(abs(A))[, 1:2]
  agree <- mean(cls[Edges[,1]] == cls[Edges[,2]])

  # Entropy (normalized)
  rowent <- function(p) {
    p <- pmax(p, 1e-12)
    -sum(p * log(p)) / log(ncol(X))
  }
  entropy <- mean(apply(X, 1, rowent))

  # Sharpness (mean maximum probability)
  sharp <- mean(apply(X, 1, max))

  c(obj=obj, nll=nll, TVsoft1=tv,
    neigh_agree=agree, entropy=entropy, sharp=sharp)
}

#' Convert component mixtures to cell-type scores
#'
#' @param X Mixture matrix (N × D)
#' @param B Dictionary (K × D)
#' @return Cell-type score matrix (N × K), row-normalized
#'
X_to_types <- function(X, B) {
  XT <- X %*% t(B)                    # N x K
  sweep(XT, 1, rowSums(XT) + 1e-12, "/")  # row-normalize
}
