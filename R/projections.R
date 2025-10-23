#!/usr/bin/env Rscript
# Simplex projection utilities

#' Project vector to floor-simplex
#'
#' @param v Input vector
#' @param K Dimension (default: length(v))
#' @param eps Floor constraint (default: 1e-3)
#' @return Projected vector with sum=1 and all elements >= eps
#'
#' @details
#' Projects v to the simplex Delta^{K-1}_eps = {u >= eps, 1^T u = 1}
#' Uses efficient sorting-based algorithm
#'
proj_simplex <- function(v, K = length(v), eps = 1e-3) {
  s <- 1 - K * eps
  u <- sort(pmax(v - eps, 0), decreasing = TRUE)
  cssv <- cumsum(u) - s
  rho <- max(which(u > cssv / seq_along(u)))
  theta <- cssv[rho] / rho
  pmax(v - eps - theta, 0) + eps
}

#' Project matrix rows to simplex
#'
#' @param X Input matrix (N × D)
#' @param eps Floor constraint (default: 1e-3)
#' @return Matrix with each row on simplex
#'
proj_rows_simplex <- function(X, eps = 1e-3) {
  t(apply(X, 1, proj_simplex, K = ncol(X), eps = eps))
}

#' Project matrix columns to simplex
#'
#' @param B Input matrix (K × D)
#' @param eps Floor constraint (default: 1e-3)
#' @return Matrix with each column on simplex
#'
proj_cols_simplex <- function(B, eps = 1e-3) {
  apply(B, 2, proj_simplex, K = nrow(B), eps = eps)
}
