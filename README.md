# TV-Regularized Spatial Mixture Models

**Fast, scalable spatial deconvolution with total variation regularization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This package implements **total variation (TV) regularized mixture models** for spatial transcriptomics data. It learns spatially coherent cell-type mixtures by:

1. **Low-rank factorization**: Decomposes per-spot cell-type scores into mixture coefficients (X) × dictionary (B)
2. **Spatial regularization**: Penalizes rapid changes in mixture proportions across neighboring spots
3. **Adaptive weighting**: Learns edge-specific weights to allow sharp boundaries at biological interfaces

The method combines:
- **Fast FISTA** (C++) for X and B optimization
- **Linear programming** (CVXR) for adaptive edge weights
- **Sparse graph representations** for scalability to large datasets (N > 10,000 spots)

## Key Features

- **Scalable**: Handles datasets with 10,000+ spots (FISTA is ~100× faster than CVX)
- **Spatially aware**: Uses Delaunay or MST graphs to model tissue geometry
- **Adaptive**: Learns edge weights to preserve sharp boundaries
- **Validated**: Matches MATLAB/CVX solutions within 1-5%
- **Modular**: Separate components for initialization, optimization, evaluation, visualization

## Quick Start

```r
# Load libraries
library(Rcpp)
library(Matrix)
library(CVXR)

# Source functions
source("R/init_A.R")
source("R/init_B.R")
source("R/alt_xbw_cvxr.R")
sourceCpp("src/FISTA_w.cpp")

# Load data
S <- as.matrix(read.csv("data/raw/S.csv", row.names = 1))
A <- as(read.csv("data/raw/A.csv", row.names = 1), "dgCMatrix")
B_init <- as.matrix(read.csv("data/raw/B.csv", row.names = 1))

# Run optimization
res <- alt_xbw_cvxr(S, A, B_init, lambda = 30, outer_it = 15)

# Results
X <- res$X  # Mixture coefficients (N × D)
B <- res$B  # Dictionary (K × D)
w <- res$w  # Edge weights (length m)
```

See [analysis/example_workflow.R](analysis/example_workflow.R) for complete workflow.

## Installation

### Prerequisites

- R ≥ 4.0
- C++ compiler (for Rcpp)
- Required R packages: `Rcpp`, `Matrix`, `CVXR`, `ggplot2`

### Quick Install

```r
# Install CRAN dependencies
install.packages(c("Rcpp", "Matrix", "CVXR", "ggplot2", "viridis"))

# Compile C++ code (first time only)
Rcpp::sourceCpp("src/FISTA_w.cpp")
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed instructions.

## Method

Given:
- **S** (N × K): Cell-type likelihood scores from RCTD
- **A** (m × N): Incidence matrix from spatial graph (Delaunay/MST)
- **D**: Number of mixture components (typically 3-5)

We minimize:

```
min_{X,B} -Σ log(S ⊙ (XB^T)) + λ Σ_e w_e ||AX||_soft-L1
s.t. X rows on simplex, B columns on simplex
```

where `||·||_soft-L1` is the soft-L1 norm (smooth approximation to L1).

### Algorithm

**Alternating optimization**:
1. **X-step**: FISTA with fixed B, w
2. **B-step**: FISTA with fixed X, w
3. **w-step**: Linear program with fixed X

Converges in 10-20 outer iterations.

See [docs/algorithm.md](docs/algorithm.md) for mathematical details.

## Repository Structure

```
tv-spatial-mixture/
├── README.md                  # This file
├── LICENSE
│
├── src/
│   ├── FISTA_w.cpp           # Core C++ optimization
│   └── README.md             # Compilation notes
│
├── R/
│   ├── alt_xbw_cvxr.R        # Main wrapper function
│   ├── init_A.R              # Spatial graph construction
│   ├── init_B.R              # Dictionary initialization
│   ├── evaluation.R          # Metrics and diagnostics
│   └── visualization.R       # Plotting utilities
│
├── data/
│   ├── raw/
│   │   ├── S.csv             # Likelihood scores
│   │   ├── A.csv             # Incidence matrix
│   │   └── B.csv             # Initial dictionary
│   └── README.md             # Data schema
│
├── analysis/
│   ├── example_workflow.R    # Reproducible example
│   ├── config.R              # Hyperparameter defaults
│   └── outputs/              # Results (gitignored)
│
├── tests/
│   ├── test_projections.R    # Unit tests
│   └── test_dimensions.R     # Dimension checks
│
├── docs/
│   ├── algorithm.md          # Mathematical details
│   ├── INSTALL.md            # Installation guide
│   └── TROUBLESHOOTING.md    # Common issues
│
└── matlab/
    ├── tvspatm.m             # Reference implementation
    └── README.md             # MATLAB vs R comparison
```

## Usage

### 1. Prepare Data from RCTD

```r
library(spacexr)
myRCTD <- readRDS("myRCTD.rds")

# Extract scores
source("R/init_A.R")
source("R/init_B.R")

n <- myRCTD@cell_type_info$info[[3]]
S <- matrix(0, nrow(myRCTD@results$results_df), n)
rownames(S) <- rownames(myRCTD@results$results_df)
colnames(S) <- myRCTD@cell_type_info$info[[2]]

SCALE <- 10
for (i in seq_len(nrow(S))) {
  sc <- myRCTD@results$singlet_scores[[i]]
  S[i, names(sc)] <- exp((min(sc) - sc) / SCALE)
}

# Build spatial graph
coords <- myRCTD@spatialRNA@coords[rownames(S), ]
A <- init_A(coords, method = "delaunay")

# Initialize dictionary
B_init <- init_B(S, A, D = 3)
```

### 2. Run Optimization

```r
source("R/alt_xbw_cvxr.R")
Rcpp::sourceCpp("src/FISTA_w.cpp")

res <- alt_xbw_cvxr(
  S_mat = S,
  A = A,
  B_init = B_init,
  lambda = 30,        # TV strength
  delta = 1e-6,       # Soft-L1 smoothing
  outer_it = 15,      # Alternating iterations
  it_X = 3000,        # FISTA iterations for X
  it_B = 2500,        # FISTA iterations for B
  wmin = 0.01,        # Min edge weight
  wmax = 0.80         # Max edge weight
)
```

### 3. Evaluate and Visualize

```r
source("R/evaluation.R")
source("R/visualization.R")

# Metrics
metrics <- eval_run(S, A, res$X, res$B, res$w, lambda = 30)
print(metrics)

# Convergence plot
plot_convergence(res$hist)

# Weight distribution
plot_weights(res$w)

# Spatial map (if myRCTD available)
plot_types(res$X, res$B, myRCTD, title = "TV-weighted mixtures")
```

## Hyperparameter Tuning

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lambda` | 30 | 1-100 | TV strength (higher = smoother) |
| `delta` | 1e-6 | 1e-8 to 1e-4 | Soft-L1 smoothing (keep small) |
| `wmin`, `wmax` | 0.01, 0.80 | [0, 1] | Edge weight bounds |
| `outer_it` | 15 | 5-30 | Alternating iterations |
| `it_X`, `it_B` | 3000, 2500 | 500-5000 | FISTA inner iterations |

**Tips**:
- Start with `lambda = 10-30`, increase for more spatial coherence
- Use `wmin = 2` to disable adaptive weighting (uniform weights)
- Increase `it_X`, `it_B` if convergence is slow (check `res$hist`)

## Performance

**FISTA vs CVX benchmark** (N=1000, K=10, D=3):

| Method | Time/iter | Memory | Scalability |
|--------|-----------|--------|-------------|
| MATLAB CVX | ~60s | 2 GB | N < 2000 |
| R FISTA | ~0.5s | 50 MB | N > 10000 |

**Speedup**: ~100× faster than CVX

## Citation

If you use this method, please cite:

```
@article{tv-spatial-mixture-2024,
  title={TV-Regularized Spatial Mixture Models for Transcriptomics},
  author={Liu, Hongyi},
  year={2024}
}
```

## Related Methods

- [RCTD](https://github.com/dmcable/spacexr): Cell-type deconvolution (provides S matrix)
- [SPOTlight](https://github.com/MarcElosua/SPOTlight): NMF-based deconvolution
- [cell2location](https://github.com/BayraktarLab/cell2location): Bayesian spatial mapping

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For bugs or feature requests, open an issue.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- **Author**: Hongyi Liu
- **Issues**: [GitHub Issues](https://github.com/yourusername/tv-spatial-mixture/issues)

## Acknowledgments

- RCTD team for spatial deconvolution framework
- Rcpp developers for seamless R/C++ integration
- CVXR team for convex optimization in R
