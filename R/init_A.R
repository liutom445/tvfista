#!/usr/bin/env Rscript
# Fast spatial graph initialization for TV regularization
# Handles both small (N<5000) and large (N>5000) datasets efficiently

library(Matrix)
library(interp)
library(igraph)

#' Build spatial graph - universal function
#' 
#' @param coords Spatial coordinates (N x 2)
#' @param S Score matrix (N x K), only needed for MST
#' @param method "delaunay" (default) or "mst"
#' @param spatial_weight Weight for MST (default: 0.3)
#' @return Sparse incidence matrix A (m x N)
init_A <- function(coords, S = NULL, method = "delaunay", spatial_weight = 0.3) {
  
  N <- nrow(coords)
  method <- tolower(method)
  
  if (method == "delaunay") {
    # Fast Delaunay triangulation
    tri <- tri.mesh(coords)
    m <- nrow(tri$arcs)
    A <- Matrix(0, m, N, sparse = TRUE)
    A[cbind(1:m, tri$arcs[,'from'])] <-  1
    A[cbind(1:m, tri$arcs[,'to'])]   <- -1
    cat(sprintf("Built Delaunay graph: %d edges for %d spots\n", m, N))
    
  } else if (method == "mst") {
    if (is.null(S)) stop("S matrix required for MST method")
    
    # Choose fast method based on N
    if (N < 5000) {
      cat("Building exact MST...\n")
      A <- build_mst_exact(coords, S, spatial_weight)
    } else {
      cat("Building fast MST (Delaunay-based)...\n")
      A <- build_mst_fast(coords, S, spatial_weight)
    }
    
  } else {
    stop("method must be 'delaunay' or 'mst'")
  }
  
  return(A)
}


#' Exact MST (for N < 5000)
build_mst_exact <- function(coords, S, spatial_weight) {
  N <- nrow(coords)
  
  # Cell-type dissimilarity
  S_norm <- sweep(S, 1, sqrt(rowSums(S^2)) + 1e-12, '/')
  cell_dissim <- 1 - tcrossprod(S_norm)
  
  # Spatial distance
  spatial_dist <- as.matrix(dist(coords))
  spatial_dist <- spatial_dist / (max(spatial_dist) + 1e-12)
  
  # Combined distance
  combined <- cell_dissim + spatial_weight * spatial_dist
  diag(combined) <- 0
  
  # MST
  g <- graph_from_adjacency_matrix(combined, mode = "undirected", 
                                   weighted = TRUE, diag = FALSE)
  mst_graph <- mst(g)
  
  # Convert to incidence matrix
  edges <- as_edgelist(mst_graph, names = FALSE)
  m <- nrow(edges)
  A <- Matrix(0, m, N, sparse = TRUE)
  A[cbind(1:m, edges[,1])] <-  1
  A[cbind(1:m, edges[,2])] <- -1
  
  cat(sprintf("Built MST: %d edges\n", m))
  return(A)
}


#' Fast MST via Delaunay (for N >= 5000)
build_mst_fast <- function(coords, S, spatial_weight) {
  N <- nrow(coords)
  
  # Step 1: Delaunay triangulation (spatial structure)
  tri <- tri.mesh(coords)
  edges_del <- tri$arcs
  m_del <- nrow(edges_del)
  
  # Step 2: Cell-type dissimilarity
  S_norm <- sweep(S, 1, sqrt(rowSums(S^2)) + 1e-12, '/')
  cell_dissim <- 1 - tcrossprod(S_norm)
  
  # Step 3: Weight Delaunay edges only (not full N×N!)
  edge_coords_from <- coords[edges_del[,'from'], ]
  edge_coords_to <- coords[edges_del[,'to'], ]
  spatial_dists <- sqrt(rowSums((edge_coords_from - edge_coords_to)^2))
  spatial_dists <- spatial_dists / max(spatial_dists)
  
  edge_weights <- numeric(m_del)
  for (e in 1:m_del) {
    i <- edges_del[e, 'from']
    j <- edges_del[e, 'to']
    edge_weights[e] <- cell_dissim[i, j] + spatial_weight * spatial_dists[e]
  }
  
  # Step 4: MST on Delaunay graph
  g <- graph_from_edgelist(edges_del, directed = FALSE)
  E(g)$weight <- edge_weights
  mst_graph <- mst(g)
  
  # Step 5: Convert to incidence matrix
  edges <- as_edgelist(mst_graph, names = FALSE)
  m <- nrow(edges)
  A <- Matrix(0, m, N, sparse = TRUE)
  A[cbind(1:m, edges[,1])] <-  1
  A[cbind(1:m, edges[,2])] <- -1
  
  cat(sprintf("Built MST: %d edges (%.1f%% of Delaunay)\n", m, 100*m/m_del))
  return(A)
}


# ============ USAGE ============

if (FALSE) {
  # Example 1: Delaunay (fast, standard)
  A <- init_A(coords = Xcoords)
  
  # Example 2: MST (adaptive boundaries)
  A <- init_A(coords = Xcoords, S = S_mat, method = "mst")
  
  # Example 3: MST with custom weight
  A <- init_A(coords = Xcoords, S = S_mat, method = "mst", spatial_weight = 0.5)
}