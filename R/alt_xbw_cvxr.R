#!/usr/bin/env Rscript
# Main optimization wrapper: alternating X,B (FISTA) and w (LP)
# Requires: Rcpp, Matrix, CVXR

#' Alternating X,B,w optimization with adaptive weights
#'
#' @param S_mat Likelihood score matrix (N × K)
#' @param A Sparse incidence matrix (m × N) from spatial graph
#' @param B_init Initial dictionary (K × D)
#' @param lambda TV regularization strength (default: 1.0)
#' @param delta Soft-L1 smoothing parameter (default: 1e-4)
#' @param outer_it Number of outer iterations (default: 5)
#' @param it_X FISTA iterations for X (default: 1500)
#' @param it_B FISTA iterations for B (default: 500)
#' @param eps_proj Floor constraint for simplex projection (default: 1e-4)
#' @param eps_step Step size tolerance (default: 1e-2)
#' @param safety Safety factor for step size (default: 0.6)
#' @param log_every Logging frequency (default: 1000)
#' @param wmin Minimum edge weight (default: 0.01)
#' @param wmax Maximum edge weight (default: 0.50)
#' @param w_init Initial weights (optional, defaults to uniform)
#' @param X0_init Initial X (optional)
#'
#' @return List with X (N×D), B (K×D), w (length m), hist (convergence history)
#'
#' @details
#' Alternates between:
#' - X,B step: FISTA in C++ with fixed w (calls xb_fista_wfixed from FISTA_w.cpp)
#' - w step: Linear program via CVXR to minimize sum of TV jumps subject to
#'   constraints wmin <= w <= wmax and |A^T|w = 1
#'
#' Objective:
#'   min_{X,B} -sum log(S * (X*B^T)) + lambda * sum_e w_e * ||AX||_soft-L1
#'   s.t. X rows on simplex, B columns on simplex
#'
alt_xbw_cvxr <- function(S_mat, A, B_init,
                         lambda = 1.0, delta = 1e-4,
                         outer_it = 5,
                         it_X = 1500, it_B = 500,
                         eps_proj = 1e-4, eps_step = 1e-2,
                         safety = 0.6, log_every = 1000,
                         wmin = 0.01, wmax = 0.50,
                         w_init = NULL,
                         X0_init = NULL) {

  stopifnot(is.matrix(S_mat), inherits(A, "dgCMatrix"), is.matrix(B_init))
  N <- nrow(S_mat)
  m <- nrow(A)
  B <- B_init

  # Initialize weights
  if (!is.null(w_init)) {
    stopifnot(length(w_init) == m)
    w <- as.numeric(w_init)
    cat("Using provided initial weights\n")
    cat(sprintf("  min=%.4f, max=%.4f, mean=%.4f\n",
                min(w), max(w), mean(w)))
  } else {
    w <- rep(1, m)
    cat("Using uniform initial weights (w=1)\n")
  }

  # Initialize X
  X <- X0_init

  hist <- data.frame(iter=integer(0), obj=double(0),
                     mean_w=double(0), med_w=double(0))

  for (t in seq_len(outer_it)) {
    cat(sprintf("\n========== Outer Iteration %d / %d ==========\n", t, outer_it))

    # X,B step in C++ (calls compiled function from FISTA_w.cpp)
    res <- xb_fista_wfixed(S_mat, A, B, w,
                           lambda=lambda, delta=delta,
                           it_X=it_X, eps_proj_X=eps_proj, eps_step_X=eps_step,
                           normalize_for_bound=TRUE, safety_X=safety,
                           log_every_X=log_every,
                           it_B=it_B, eps_proj_B=eps_proj, eps_step_B=eps_step,
                           safety_B=safety, log_every_B=log_every,
                           update_B=TRUE,
                           X0_opt = if (is.null(X)) NULL else X)
    X <- res$X
    B <- res$B

    # w-step LP (skip if wmin>1 → W=I)
    if (wmin <= 1) {
      cvec <- as.numeric(rowSums(abs(A %*% X)))
      wvar <- CVXR::Variable(m)
      constr <- list(wvar >= wmin, wvar <= wmax,
                     Matrix::t(abs(A)) %*% wvar == 1)
      prob <- CVXR::Problem(CVXR::Minimize(t(cvec) %*% wvar), constr)
      sol <- try(CVXR::solve(prob, solver = "ECOS"), silent = TRUE)

      if (!inherits(sol, "try-error") &&
          !(sol$status %in% c("infeasible","unbounded"))) {
        w <- as.vector(sol$getValue(wvar))
        cat(sprintf("  w optimization successful (status: %s)\n", sol$status))
      } else {
        warning("CVXR w-step failed; keeping previous w")
      }
    } else {
      message("w-step skipped (wmin>1 → W=I)")
      w <- rep(1, m)
    }

    hist <- rbind(hist, data.frame(iter=t, obj=res$obj,
                                    mean_w=mean(w), med_w=median(w)))
    cat(sprintf("  Objective: %.6e\n", res$obj))
    cat(sprintf("  Weights: mean=%.3f, median=%.3f, range=[%.3f, %.3f]\n",
                mean(w), median(w), min(w), max(w)))
  }

  list(X=X, B=B, w=w, hist=hist)
}
