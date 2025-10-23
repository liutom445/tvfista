#!/usr/bin/env Rscript
# Visualization utilities for spatial mixture models

#' Get spacexr-compatible color palette
#'
#' @param types Character vector of cell type names
#' @return Named vector of colors
#'
spacexr_pal_fixed <- function(types) {
  n <- length(types)
  if (n > 36) stop("At most 36 cell types supported")
  if (requireNamespace("pals", quietly = TRUE)) {
    pal <- if (n > 21) unname(pals::polychrome(n)) else pals::kelly(n + 1)[2:(n + 1)]
  } else {
    pal <- viridisLite::viridis(n)
  }
  setNames(pal, types)
}

#' Plot argmax spatial map with spacexr palette
#'
#' @param X_score Score matrix (N × K or N × D)
#' @param myRCTD RCTD object (for coordinates and type names)
#' @param title Plot title
#' @return ggplot2 object
#'
#' @details
#' Creates a spatial scatter plot where each spot is colored by its
#' dominant (argmax) cell type. Uses RCTD's color palette for consistency.
#'
plot_argmax_spx <- function(X_score, myRCTD, title) {
  all_types <- myRCTD@cell_type_info$info[[2]]

  if (is.null(colnames(X_score)))
    colnames(X_score) <- all_types[seq_len(ncol(X_score))]
  if (is.null(rownames(X_score)))
    rownames(X_score) <- rownames(myRCTD@spatialRNA@coords)[seq_len(nrow(X_score))]

  bc <- rownames(X_score)
  coords_all <- myRCTD@spatialRNA@coords

  xy <- if (all(bc %in% rownames(coords_all))) {
    coords_all[bc, c("x","y")]
  } else {
    coords_all[seq_len(nrow(X_score)), c("x","y")]
  }

  pal_fixed <- spacexr_pal_fixed(all_types)
  lab <- factor(colnames(X_score)[max.col(X_score, ties.method = "first")],
                levels = all_types)

  ggplot2::ggplot(data.frame(x = xy$x, y = xy$y, label = lab),
                  ggplot2::aes(x, y, color = label)) +
    ggplot2::geom_point(size = 0.15, shape = 19, alpha = 0.85) +
    ggplot2::scale_color_manual(values = pal_fixed, limits = all_types,
                                breaks = all_types, drop = FALSE) +
    ggplot2::guides(color = ggplot2::guide_legend(override.aes = list(size = 3, alpha = 1))) +
    ggplot2::coord_fixed() +
    ggplot2::theme_bw() +
    ggplot2::labs(title = title, color = NULL)
}

#' Plot component mixtures as cell-type map
#'
#' @param X Mixture matrix (N × D)
#' @param B Dictionary (K × D)
#' @param myRCTD RCTD object
#' @param title Plot title
#' @return ggplot2 object
#'
plot_types <- function(X, B, myRCTD, title) {
  XT <- X_to_types(X, B)  # from evaluation.R
  colnames(XT) <- myRCTD@cell_type_info$info[[2]]
  plot_argmax_spx(XT, myRCTD, title)
}

#' Plot convergence history
#'
#' @param hist Data frame with columns: iter, obj, mean_w, med_w
#' @param title Plot title (optional)
#' @return ggplot2 object
#'
plot_convergence <- function(hist, title = "Convergence History") {
  ggplot2::ggplot(hist, ggplot2::aes(x = iter, y = obj)) +
    ggplot2::geom_line(linewidth = 1) +
    ggplot2::geom_point(size = 2) +
    ggplot2::labs(title = title, x = "Iteration", y = "Objective Value") +
    ggplot2::theme_minimal()
}

#' Plot weight distribution
#'
#' @param w Weight vector
#' @param wmin Minimum weight constraint
#' @param wmax Maximum weight constraint
#' @param title Plot title (optional)
#' @return ggplot2 object
#'
plot_weights <- function(w, wmin = 0.01, wmax = 0.50,
                        title = "Edge Weight Distribution") {
  ggplot2::ggplot(data.frame(w = w), ggplot2::aes(x = w)) +
    ggplot2::geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    ggplot2::geom_vline(xintercept = c(wmin, wmax), linetype = "dashed", color = "red") +
    ggplot2::labs(title = title,
                  subtitle = sprintf("Bounds: [%.2f, %.2f]", wmin, wmax),
                  x = "Weight", y = "Count") +
    ggplot2::theme_minimal()
}
