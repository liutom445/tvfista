# Data Directory

## Overview

This directory contains input data for the TV-regularized spatial mixture model.

## Files in `raw/`

### S.csv
**Likelihood score matrix** (N × K)
- **N**: Number of spatial spots (rows)
- **K**: Number of cell types (columns)
- **Values**: Non-negative likelihood scores from RCTD singlet analysis
- **Processing**: Scores are transformed via `exp((min(sc) - sc) / SCALE)` to create soft likelihoods
- **Usage**: Input to model as observation matrix

### A.csv
**Incidence matrix** (m × N)
- **m**: Number of edges in spatial graph
- **N**: Number of spots
- **Values**: {-1, 0, +1} indicating edge connectivity
- **Structure**: Each row represents one edge; has exactly two non-zero entries (+1 and -1) at incident spots
- **Graph type**: Typically Delaunay triangulation or MST (minimum spanning tree)
- **Usage**: Defines spatial neighborhood structure for TV regularization

### B.csv
**Initial dictionary** (K × D)
- **K**: Number of cell types
- **D**: Number of mixture components (latent dimensions)
- **Values**: Non-negative, columns sum to 1 (simplex constraint)
- **Initialization**: From k-means clustering on graph-smoothed S
- **Usage**: Starting point for dictionary optimization

## Schema

### Coordinate Information
The spatial coordinates (x, y) are stored in the RCTD object and referenced via row names (barcodes) in S.csv.

### Dimensions
Typical sizes:
- N ≈ 1000-5000 (spots)
- K ≈ 10-20 (cell types)
- D ≈ 3-5 (components)
- m ≈ 3N (for Delaunay) or N-1 (for MST)

## Data Generation

To regenerate these files from RCTD:

```r
# Extract S from RCTD
myRCTD <- readRDS("path/to/myRCTD.rds")
n <- myRCTD@cell_type_info$info[[3]]
S <- matrix(0, nrow(myRCTD@results$results_df), n)
rownames(S) <- rownames(myRCTD@results$results_df)
colnames(S) <- myRCTD@cell_type_info$info[[2]]

SCALE <- 10
for (i in seq_len(nrow(S))) {
  sc <- myRCTD@results$singlet_scores[[i]]
  S[i, names(sc)] <- exp((min(sc) - sc) / SCALE)
}
write.csv(S, "S.csv")

# Build spatial graph
source("R/init_A.R")
Xcoords <- myRCTD@spatialRNA@coords[rownames(S), ]
A <- init_A(Xcoords, method = "delaunay")
write.csv(as.matrix(A), "A.csv")

# Initialize dictionary
source("R/init_B.R")
B <- init_B(S, A, D = 3)
write.csv(B, "B.csv")
```

## Notes

- All CSV files include row and column names for traceability
- Data should be read with `read.csv(..., row.names = 1)`
- A should be converted to sparse format: `A_sparse <- as(A, "dgCMatrix")`
