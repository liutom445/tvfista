# TV-Regularized Spatial Deconvolution

**Spatially-aware cell-type mixture estimation using Total Variation regularization and accelerated optimization**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project performs **low-rank spatial deconvolution** of cell-type mixtures in spatial transcriptomics data. Starting from cell-type likelihood scores (e.g., from RCTD), we learn:

- **Per-spot mixture weights** ($X \in \mathbb{R}^{N \times D}$) with spatial smoothness
- **Low-rank dictionary** ($B \in \mathbb{R}^{K \times D}$) capturing $D$ mixture archetypes from $K$ cell types
- **Adaptive edge weights** ($w \in \mathbb{R}^m$) that identify tissue boundaries

The key innovation: **soft-$\ell_1$ Total Variation** with learned edge weights automatically preserves sharp boundaries while smoothing within homogeneous regions.

## Method

### Objective Function

$$
\min_{X,B,w} \; -\sum_{i=1}^N \log(z_i) + \lambda \sum_{e=1}^m w_e \sum_{d=1}^D \sqrt{(AX)_{ed}^2 + \delta}
$$

where:
- $z_i = \sum_k S_{ik}(XB^\top)_{ik}$ is the evidence at spot $i$
- $A \in \mathbb{R}^{m \times N}$ is the spatial graph incidence matrix (Delaunay triangulation)
- $\delta = 10^{-6}$ provides smooth approximation to $|\cdot|$

**Constraints**: Rows of $X$ and columns of $B$ lie on the probability simplex ($\geq \epsilon$, sum to 1).

### Algorithm: Alternating FISTA + LP

We alternate three steps until convergence:

1. **X-step (FISTA)**: Update mixture weights with TV regularization
   - Fast Iterative Shrinkage-Thresholding Algorithm (Beck & Teboulle 2009)
   - Convergence rate: $O(1/k^2)$ vs $O(1/k)$ for standard gradient descent
   - 1500-2000 iterations, ~1.2 seconds

2. **B-step (FISTA)**: Update dictionary (data term only, no TV)
   - 500 iterations, ~0.5 seconds

3. **w-step (Linear Program)**: Learn edge weights
   - Minimizes $w^\top c$ subject to $w_{\min} \leq w \leq w_{\max}$ and $|A|^\top w = \mathbf{1}$
   - Solved via CVXR/ECOS, ~0.2 seconds

**Total runtime**: ~4 minutes for 777 spots × 19 cell types × 3 components (3 outer iterations)

## Results: Mouse Cerebellum

| Metric | Value | Description |
|--------|-------|-------------|
| **Objective reduction** | 59.7% | From 22,714 → 9,146 over 3 iterations |
| **Sharpness** | 0.80 | Mean max weight (most spots have dominant component) |
| **Edge weights** | Bimodal | 40% at $w_{\min}=0.01$ (boundaries), 60% at 0.10-0.40 (interior) |
| **Spatial coherence** | High | Recovers cerebellar layer structure (granular/molecular/Purkinje) |

**Visualization**: The learned mixtures reveal spatially coherent regions corresponding to known cerebellar architecture, with sharp transitions between cortical layers preserved by adaptive edge weighting.

## Key Features

✅ **Fast**: C++/Armadillo implementation with O(1/k²) convergence  
✅ **Scalable**: Handles N > 10K spots with sparse matrix operations  
✅ **Adaptive**: Learns which edges to smooth (interior) vs preserve (boundaries)  
✅ **Interpretable**: Low-rank factorization reduces K cell types → D mixture components  
✅ **Validated**: Matches CVX reference implementation (15× faster)

## Installation

```r
# Install dependencies
install.packages(c("Rcpp", "RcppArmadillo", "Matrix", "CVXR"))

# Compile C++ code
Rcpp::sourceCpp("FISTA_w.cpp")

# Load R wrapper
source("tv_withW.Rmd")
```

**Requirements**:
- R ≥ 4.0
- C++11 compiler
- Armadillo library (header-only, auto-installed by RcppArmadillo)

## Quick Start

```r
# Load your data
S_mat <- load_RCTD_scores(myRCTD)              # N × K score matrix
A     <- build_delaunay_graph(coords)          # m × N incidence matrix  
B_init <- initialize_dictionary(S_mat, D=3)    # K × D initial dictionary

# Run optimization
result <- alt_xbw_cvxr(
    S_mat, A, B_init,
    lambda = 2.0,        # TV regularization strength
    outer_it = 3,        # Alternating iterations
    it_X = 2000,         # FISTA iterations for X
    it_B = 500,          # FISTA iterations for B
    wmin = 0.01,         # Min edge weight (boundaries)
    wmax = 0.40          # Max edge weight (interior)
)

# Extract results
X_final <- result$X    # N × D mixture weights
B_final <- result$B    # K × D dictionary
w_final <- result$w    # m edge weights

# Visualize
plot_types(X_final, B_final, myRCTD, title="Spatial Mixtures")
```

## Files

| File | Description | Lines |
|------|-------------|-------|
| `FISTA_w.cpp` | Core optimization (C++/Rcpp) | 400 |
| `tv_withW.Rmd` | R wrapper + analysis pipeline | 150 |
| `tvspatm.m` | Matlab/CVX reference | 100 |
| `tvspat_blog_post.md` | Detailed methodology | 1000+ |

## Why FISTA?

For spatial transcriptomics researchers unfamiliar with optimization, **FISTA** offers:

- **Simplicity**: Only requires gradient computation + projection
- **Speed**: 10-50× faster than second-order methods for this problem size
- **Convergence**: Provable $O(1/k^2)$ rate with momentum
- **Flexibility**: Easy to modify objective (change TV norm, add constraints)

The soft-$\ell_1$ smoothing ($\sqrt{u^2 + \delta}$ instead of $|u|$) makes the TV penalty differentiable while preserving edge-preserving properties.

## Extensions

- **Multi-sample analysis**: Learn shared dictionary across tissue sections
- **Uncertainty quantification**: Bootstrap or variational inference
- **Alternative spatial priors**: Replace TV with geodesic distance on manifolds
- **GPU acceleration**: For whole-slide imaging (N > 100K spots)

## Citation

If you use this code, please cite:

```bibtex
@article{beck2009fista,
  title={A fast iterative shrinkage-thresholding algorithm for linear inverse problems},
  author={Beck, Amir and Teboulle, Marc},
  journal={SIAM Journal on Imaging Sciences},
  volume={2},
  number={1},
  pages={183--202},
  year={2009}
}

@article{cable2022rctd,
  title={Robust decomposition of cell type mixtures in spatial transcriptomics},
  author={Cable, Dylan M and others},
  journal={Nature Biotechnology},
  volume={40},
  number={4},
  pages={517--526},
  year={2022}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

**Author**: Tom Liu  
**Email**: liutom@umich.edu
**Issues**: Please report bugs via GitHub Issues

---

*For detailed methodology, see [tvspat_blog_post.md](tvspat_blog_post.md)*
