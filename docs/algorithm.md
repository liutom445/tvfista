# Algorithm Details

## Mathematical Formulation

### Problem Statement

Given spatial transcriptomics data, we estimate per-spot cell-type mixtures with spatial regularization.

**Input data**:
- **S** ∈ ℝ^(N×K)_≥0: Cell-type likelihood scores from RCTD
- **Coordinates**: (x_i, y_i) for i = 1..N spatial spots
- **A** ∈ ℝ^(m×N): Sparse incidence matrix (one row per graph edge)
- **D**: Number of mixture components (latent dimensions)

**Variables to learn**:
- **X** ∈ ℝ^(N×D): Per-spot mixture coefficients (rows on simplex)
- **B** ∈ ℝ^(K×D): Dictionary matrix (columns on simplex)
- **w** ∈ ℝ^m: Edge weights

### Objective Function

We minimize:

```
min_{X,B} -Σ_{i=1}^N log(z_i) + λ Σ_{e=1}^m w_e Σ_{d=1}^D √((AX)_{ed}² + δ)

subject to:
  - X_{i:} ∈ Δ^{D-1}_ε  (row i on floor-simplex)
  - B_{:d} ∈ Δ^{K-1}_ε  (column d on floor-simplex)
```

where:
- **z_i** = Σ_k S_{ik} (XB^T)_{ik} is the per-spot evidence
- **Δ^{p-1}_ε** = {u ≥ ε, 1^T u = 1} is the floor-simplex
- **δ** = 10^{-6} is the soft-L1 smoothing parameter
- **λ** > 0 is the regularization strength

### Data Term

The negative log-likelihood encourages X and B to explain the observed scores S:

```
f(X,B) = -Σ_{i=1}^N log(z_i)
       = -Σ_{i=1}^N log(Σ_k S_{ik} Σ_d X_{id} B_{kd})
```

This is the log-evidence for mixture model S ≈ X B^T.

### Regularization Term

The soft-L1 total variation penalty encourages spatial smoothness:

```
φ(X;w) = Σ_{e=1}^m w_e Σ_{d=1}^D √((AX)_{ed}² + δ)
```

where:
- **(AX)_{ed}** measures the difference in component d across edge e
- **w_e** is the learned weight for edge e
- **√(u² + δ)** smoothly approximates |u| (soft-L1 norm)

**Why soft-L1?** The standard L1 norm |u| is non-differentiable at u=0. We use √(u² + δ) which:
- Is differentiable everywhere
- Approximates |u| for |u| >> √δ
- Avoids numerical issues near zero

## Gradients

### Data Term Gradients

Let Z = XB^T ∈ ℝ^(N×K) and z_i = Σ_k S_{ik} Z_{ik}.

```
∇_X f = -(S ⊘ z) B

∇_B f = -(S ⊘ z)^T X
```

where ⊘ denotes element-wise division by the vector z (broadcast to columns).

**Derivation**:
```
∂f/∂X_{id} = -Σ_i (Σ_k S_{ik} B_{kd}) / z_i
           = -(Σ_k (S_{ik}/z_i) B_{kd})
```

### TV Penalty Gradient

Let U = AX ∈ ℝ^(m×D).

```
∇_X φ = A^T (diag(w) · (U ⊘ √(U² + δ)))
```

where:
- **U² + δ** is element-wise
- **⊘** is element-wise division
- **diag(w)** scales each row e by w_e

**Derivation**:
```
∂φ/∂X_{id} = Σ_e w_e · (AX)_{ed} / √((AX)_{ed}² + δ) · A_{ei}
           = Σ_e A^T_{ie} · w_e · (AX)_{ed} / √((AX)_{ed}² + δ)
```

## FISTA Updates

### X-step (with fixed B, w)

FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) uses momentum acceleration:

```
1. Compute gradient: g = ∇_X f + λ ∇_X φ
2. Gradient step: X_new = Y_X - η_X · g
3. Project: X_new = Π_row-simplex(X_new)
4. Momentum: Y_X = X_new + ω_X (X_new - X_old)
```

**Parameters**:
- **η_X** = (safety) / L_X: Step size
- **L_X** = L_data + L_pen: Lipschitz constant estimate
- **ω_X** = (k-1)/(k+2): Nesterov momentum coefficient
- **safety** ≈ 0.6-0.9: Safety factor for backtracking

**Lipschitz Bound**:
```
L_data ≈ ||S||_max · ||B||_F² / ε_min

L_pen ≤ λ ||A^T A||_2 · max(w) / √δ
```

### B-step (with fixed X, w)

Similar to X-step but:
- Gradient: g = ∇_B f (no penalty on B)
- Projection: column-wise simplex projection
- Step size: η_B = (safety) / L_B

```
L_B ≈ ||S||_max · ||X||_F² / ε_min
```

### w-step (with fixed X)

Linear program solved via CVXR:

```
min_w  Σ_e c_e w_e

subject to:
  - w_min ≤ w_e ≤ w_max  for all e
  - |A^T| w = 1  (normalize by node degree)
```

where **c_e** = Σ_d √((AX)_{ed}² + δ) is the soft-L1 TV contribution of edge e.

**Constraint normalization**: |A^T|w = 1 ensures that node degrees sum to 1, preventing trivial w=0.

## Simplex Projection

### Row-wise Simplex (for X)

Project each row x ∈ ℝ^D to Δ^{D-1}_ε:

```
Algorithm (Euclidean projection):
1. Shift: v = x - ε
2. Sort: u = sort(max(v, 0), decreasing)
3. Find threshold:
   ρ = max{j : u_j > (Σ_{i≤j} u_i - s) / j}
   where s = 1 - D·ε
4. Threshold: θ = (Σ_{i≤ρ} u_i - s) / ρ
5. Project: x_new = max(v - θ, 0) + ε
```

**Complexity**: O(D log D) per row

### Column-wise Simplex (for B)

Apply the same projection to each column.

## Alternating Algorithm

```
Initialize: X, B, w
Repeat for t = 1..T_outer:
  1. X-step: Run FISTA for T_X iterations
  2. B-step: Run FISTA for T_B iterations
  3. w-step: Solve LP (unless w_min > 1)
  4. Record: obj, mean(w), median(w)
Until convergence or max iterations
```

**Typical parameters**:
- T_outer = 10-20
- T_X = 1500-5000
- T_B = 500-2500

**Convergence**: Objective typically decreases monotonically (small non-monotonicity possible due to alternation).

## Computational Complexity

### Per iteration costs:

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Gradient ∇_X f | O(N·K·D) | Dense matrix multiply |
| Gradient ∇_X φ | O(nnz(A)·D) | Sparse A^T (AX) |
| Simplex projection X | O(N·D log D) | Sort per row |
| Simplex projection B | O(K·D log D) | Sort per column |
| LP for w | O(m³) | ECOS solver (m << N) |

**Overall**:
- X-step: O(N·K·D + m·D) per FISTA iteration
- B-step: O(N·K·D) per FISTA iteration
- w-step: O(m³) but m ≈ 3N for Delaunay

**Scalability**: Dominated by O(N·K·D) matrix operations, which are highly vectorizable.

## Initialization

### Dictionary B

1. **Graph smoothing**: S_smooth = (1-α)S + α W S
   - W: row-normalized adjacency from A
   - α ≈ 0.3
2. **K-means**: Cluster S_smooth into D clusters
3. **Extract**: B = cluster_centers^T (K × D)
4. **Project**: Column-wise simplex projection

### Spatial graph A

**Delaunay triangulation** (default):
- Fast: O(N log N)
- Sparse: m ≈ 3N edges
- Respects spatial geometry

**MST** (optional):
- Adaptive to cell-type boundaries
- Uses combined distance: d_cell + α·d_spatial
- Sparser: m = N-1 edges
- Slower for large N

### Weights w

Start with uniform: w = 1 (all edges equally weighted).

## Numerical Considerations

### Stability

1. **Floor constraint**: ε ≈ 10^{-3} prevents division by zero in log(z_i)
2. **Soft-L1**: δ = 10^{-6} smooths gradients near zero
3. **Step size safety**: Factor 0.6-0.9 prevents divergence

### Precision

- Objective values: relative tolerance ~ 10^{-6}
- Constraints: ||row_sums(X) - 1|| < 10^{-6}
- CVXR solver: default tolerance (ECOS)

## Validation

The implementation has been validated against MATLAB/CVX:
- **Objective difference**: < 2%
- **Solution error** (Frobenius norm): < 5%
- **Constraint satisfaction**: Both satisfy to solver tolerance

See `analysis/example_workflow.R` for comparison workflow.

## References

1. **FISTA**: Beck & Teboulle (2009), "A Fast Iterative Shrinkage-Thresholding Algorithm"
2. **Simplex projection**: Condat (2016), "Fast Projection onto the Simplex and the l1 Ball"
3. **Total variation**: Rudin et al. (1992), "Nonlinear total variation based noise removal"
4. **Spatial deconvolution**: Cable et al. (2022), "Robust decomposition of cell type mixtures in spatial transcriptomics"
