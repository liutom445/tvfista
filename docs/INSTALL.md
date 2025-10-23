# Installation Guide

## Quick Start

```r
# Install CRAN dependencies
install.packages(c("Rcpp", "Matrix", "CVXR", "ggplot2", "viridis", "interp"))

# Clone repository
git clone https://github.com/yourusername/tv-spatial-mixture.git
cd tv-spatial-mixture

# Compile C++ code
Rcpp::sourceCpp("src/FISTA_w.cpp")

# Test installation
source("R/alt_xbw_cvxr.R")
```

## System Requirements

### Operating Systems
- **Linux**: Ubuntu 18.04+, CentOS 7+
- **macOS**: 10.14+ (Mojave or later)
- **Windows**: 10+ (with Rtools)

### Software Dependencies
- **R**: Version 4.0.0 or later
- **C++ compiler**:
  - Linux: g++ 7.0+ or clang 9.0+
  - macOS: Xcode Command Line Tools
  - Windows: Rtools 4.0+

### Hardware Requirements
- **RAM**: 4 GB minimum, 8 GB recommended
- **Storage**: 500 MB for package + data
- **CPU**: Modern multi-core processor recommended (optimization can utilize multiple cores via BLAS)

## Detailed Installation

### 1. R Installation

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install r-base r-base-dev
```

**macOS** (via Homebrew):
```bash
brew install r
```

**Windows**: Download from [CRAN](https://cran.r-project.org/bin/windows/base/)

### 2. C++ Compiler Setup

**Ubuntu/Debian**:
```bash
sudo apt-get install build-essential
```

**macOS**:
```bash
xcode-select --install
```

**Windows**: Download [Rtools](https://cran.r-project.org/bin/windows/Rtools/) matching your R version

### 3. R Package Dependencies

#### Required Packages

```r
# Core dependencies
install.packages("Rcpp")         # C++ integration
install.packages("Matrix")       # Sparse matrices
install.packages("CVXR")         # Convex optimization

# Visualization
install.packages("ggplot2")
install.packages("viridis")

# Utilities
install.packages("interp")       # Delaunay triangulation
```

#### Optional Packages

```r
# For spatial transcriptomics
install.packages("spacexr")      # RCTD integration

# For enhanced color palettes
install.packages("pals")

# For graph operations
install.packages("igraph")

# For testing
install.packages("testthat")
```

### 4. Compile C++ Code

```r
library(Rcpp)
setwd("path/to/tv-spatial-mixture")
sourceCpp("src/FISTA_w.cpp")
```

**Expected output**:
```
Functions exported:
- xb_fista_wfixed
- eval_softL1_w_cpp
```

### 5. Verify Installation

```r
# Test basic functions
source("R/init_A.R")
source("R/init_B.R")
source("R/alt_xbw_cvxr.R")

# Check C++ compilation
if (exists("xb_fista_wfixed")) {
  cat("✓ C++ compilation successful\n")
} else {
  cat("✗ C++ compilation failed\n")
}

# Test small optimization
set.seed(123)
N <- 100; K <- 5; D <- 3
S <- matrix(runif(N*K), N, K)
A <- Matrix::sparseMatrix(i=1:99, j=1:99, x=1, dims=c(99,N))
B <- matrix(runif(K*D), K, D)
B <- sweep(B, 2, colSums(B), "/")

res <- alt_xbw_cvxr(S, A, B, lambda=1, outer_it=2, it_X=100, it_B=50)
cat("✓ Basic optimization test passed\n")
```

## Platform-Specific Notes

### Linux

**OpenMP support** (optional, for future parallelization):
```bash
sudo apt-get install libomp-dev
```

**Optimized BLAS** (recommended for speed):
```bash
sudo apt-get install libopenblas-dev
```

### macOS

**Apple Silicon (M1/M2)**:
- Use R 4.1+ (native arm64 build)
- Rcpp works natively, no Rosetta needed

**Intel Macs**:
- Standard installation works
- Consider upgrading to latest Xcode CLT

**Common issue**: Clang vs. g++
```bash
# If compilation fails, check compiler
R CMD config CXX
```

### Windows

**Rtools configuration**:
1. Install Rtools matching your R version
2. Verify PATH: `Sys.getenv("PATH")` should include Rtools
3. Test compilation: `Rcpp::evalCpp("2 + 2")`

**Common issue**: Missing make
```r
# Check if make is available
Sys.which("make")
```

If empty, reinstall Rtools and check "Add to PATH" option.

## Troubleshooting

### Issue: C++ compilation fails

**Symptoms**: `sourceCpp()` errors

**Solutions**:
1. Check compiler: `system("g++ --version")` or `system("clang++ --version")`
2. Update Rcpp: `install.packages("Rcpp")`
3. Check Makevars: `file.path("~", ".R", "Makevars")`
4. Try clean rebuild:
   ```r
   Rcpp::sourceCpp("src/FISTA_w.cpp", rebuild=TRUE, cleanupCacheDir=TRUE)
   ```

### Issue: CVXR solver fails

**Symptoms**: LP for w returns "solver failed"

**Solutions**:
1. Install ECOS solver (should be automatic with CVXR)
2. Try alternative solver:
   ```r
   solve(prob, solver = "SCS")
   ```
3. Check problem is feasible (wmin < wmax, wmin > 0)

### Issue: Out of memory

**Symptoms**: R crashes during optimization

**Solutions**:
1. Reduce data size (subsample spots)
2. Use sparse matrices consistently
3. Increase system swap space
4. Reduce IT_X, IT_B (fewer iterations)

### Issue: Slow performance

**Check**:
```r
# Check if BLAS is optimized
sessionInfo()
# Look for "BLAS: /usr/lib/..." (system) vs "BLAS: /opt/OpenBLAS/..." (optimized)
```

**Optimize**:
1. Install optimized BLAS (OpenBLAS, MKL)
2. Reduce logging: `log_every = 0`
3. Use fewer outer iterations
4. Profile code: `Rprof("profile.out")`

### Issue: Convergence problems

**Symptoms**: Objective increases, or NaN values

**Solutions**:
1. Lower safety factor: `safety = 0.5`
2. Increase epsilon: `eps_proj = 1e-3`
3. Check data: `any(S < 0)`, `any(is.na(S))`
4. Reduce lambda (may be too large)

## Advanced Configuration

### Custom BLAS

**Linux** (OpenBLAS):
```bash
sudo apt-get install libopenblas-dev
sudo update-alternatives --config libblas.so.3
```

**macOS** (Accelerate framework is default, already optimized)

### Compiler Flags

Create `~/.R/Makevars`:
```makefile
CXX = g++
CXXFLAGS = -O3 -march=native -ffast-math
```

For debugging:
```makefile
CXXFLAGS = -g -O0 -Wall -Wextra
```

### CMake Build (Alternative)

For production use, you can build a standalone shared library:

```bash
cd src
mkdir build && cd build
cmake ..
make
```

See `src/CMakeLists.txt` for configuration.

## Docker Installation (Reproducible Environment)

```dockerfile
FROM rocker/r-ver:4.3.0

RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libopenblas-dev

RUN R -e "install.packages(c('Rcpp', 'Matrix', 'CVXR', 'ggplot2', 'viridis', 'interp'))"

COPY . /app
WORKDIR /app

RUN R -e "Rcpp::sourceCpp('src/FISTA_w.cpp')"

CMD ["R"]
```

Build and run:
```bash
docker build -t tv-spatial-mixture .
docker run -it tv-spatial-mixture
```

## Testing Installation

Run the test suite:
```r
source("tests/test_projections.R")
source("tests/test_dimensions.R")
```

Expected output: All tests pass.

## Getting Help

If installation fails:
1. Check [Troubleshooting](TROUBLESHOOTING.md)
2. Search [GitHub Issues](https://github.com/yourusername/tv-spatial-mixture/issues)
3. Post a new issue with:
   - OS and R version: `sessionInfo()`
   - Error message
   - Output of `Rcpp::evalCpp("2 + 2")`

## Next Steps

After installation:
1. Read [algorithm.md](algorithm.md) for method details
2. Run [example_workflow.R](../analysis/example_workflow.R)
3. Explore your own data!
