#!/usr/bin/env Rscript
# Configuration and hyperparameters for TV-regularized mixture models

# ============================================================================
# OPTIMIZATION HYPERPARAMETERS
# ============================================================================

# Regularization strength
LAMBDA <- 30.0          # TV penalty weight (higher = more spatial smoothness)

# Soft-L1 smoothing parameter
DELTA <- 1e-6           # Smoothing for ||u||_soft = sqrt(u^2 + delta)

# Simplex floor constraint
EPS_PROJ <- 1e-3        # Minimum element value for simplex projection

# Edge weight bounds (for LP)
WMIN <- 0.01            # Minimum edge weight (set >1 to freeze W=I)
WMAX <- 0.80            # Maximum edge weight

# ============================================================================
# ALGORITHM PARAMETERS
# ============================================================================

# Outer loop (alternating X,B,w)
OUTER_ITER <- 15        # Number of alternating iterations

# FISTA iterations
IT_X <- 3000            # Inner iterations for X optimization
IT_B <- 2500            # Inner iterations for B optimization

# Step size control
EPS_STEP <- 1e-2        # Relative step size tolerance
SAFETY <- 0.8           # Safety factor for Lipschitz estimate (0.6-0.9)

# Logging
LOG_EVERY <- 0          # Log frequency (0 = no inner logging, 1000 = every 1000 iters)

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Number of mixture components
D <- 3                  # Latent dimensions (typically 3-5)

# Spatial graph type
GRAPH_METHOD <- "delaunay"  # "delaunay" or "mst"
SPATIAL_WEIGHT <- 0.3       # Weight for MST (if using MST)

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# RCTD score transformation
SCALE <- 10             # Temperature for score transformation

# Random seed
SEED <- 123

# ============================================================================
# NOTES
# ============================================================================

# Tuning guidelines:
# - LAMBDA: Start at 1-10, increase for more spatial coherence (10-100 for strong smoothing)
# - DELTA: Keep at 1e-6 (smaller values cause numerical issues)
# - EPS_PROJ: Typically 1e-3 to 1e-4
# - WMIN/WMAX: Bounds for adaptive weights (0.01-0.5 is reasonable)
#              Set WMIN > 1 to use uniform weights (W=I)
# - IT_X, IT_B: Increase if convergence is slow (check convergence plots)
# - SAFETY: Lower (0.5-0.7) for more conservative steps, higher (0.8-0.95) for faster convergence

# Example configurations:
#
# 1. Strong spatial smoothing:
#    LAMBDA <- 50; WMIN <- 0.01; WMAX <- 0.80
#
# 2. Weak spatial smoothing:
#    LAMBDA <- 5; WMIN <- 0.1; WMAX <- 0.5
#
# 3. No adaptive weights (uniform):
#    WMIN <- 2; WMAX <- 2  # (or any value > 1)
#
# 4. Fast prototyping:
#    OUTER_ITER <- 5; IT_X <- 1000; IT_B <- 500
