// FISTA_w_cleaned.cpp
// Fast Iterative Shrinkage-Thresholding Algorithm with weighted soft-L1 Total Variation
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]

#include <RcppArmadillo.h>
#include <algorithm>


using namespace Rcpp;
using namespace arma;

namespace {  // Anonymous namespace for internal functions

// ============================================================================
// Projection Operations
// ============================================================================

/**
 * Project a vector onto the probability simplex {x: sum(x) = s, x >= 0}
 * Uses efficient O(n log n) algorithm based on sorting
 * Optimized to minimize memory allocations
 */
inline void projectOntoSimplex(const rowvec& v, double targetSum, rowvec& result) {
  const uword n = v.n_elem;
  result = v;  // Copy input

  // Sort in descending order for numerical stability
  std::sort(result.begin(), result.end(), std::greater<double>());

  // Find the index where the KKT conditions are satisfied
  double cumSum = 0.0;
  uword rho = 0;
  for (uword j = 0; j < n; ++j) {
    cumSum += result[j];
    double threshold = (cumSum - targetSum) / double(j + 1);
    if (result[j] > threshold) {
      rho = j + 1;
    } else {
      break;  // Early exit optimization
    }
  }

  // Compute the Lagrange multiplier
  double theta = 0.0;
  if (rho > 0) {
    cumSum = 0.0;
    for (uword j = 0; j < rho; ++j) {
      cumSum += result[j];
    }
    theta = (cumSum - targetSum) / double(rho);
  }

  // Apply soft thresholding on original vector
  result = v;
  for (uword j = 0; j < n; ++j) {
    result[j] = std::max(0.0, v[j] - theta);
  }
}

/**
 * Project matrix rows onto simplex with floor constraint
 * Each row sums to 1 with minimum element value eps
 * Optimized to reuse buffers and avoid allocations
 */
inline void projectRowsOntoSimplexWithFloor(mat& X, double floorEps) {
  const uword numRows = X.n_rows;
  const uword numCols = X.n_cols;
  const double adjustedSum = 1.0 - double(numCols) * floorEps;

  // Reuse buffers to avoid repeated allocations
  rowvec shifted(numCols);
  rowvec projected(numCols);

  for (uword i = 0; i < numRows; ++i) {
    // Shift by floor value - FIX: access elements directly
    for (uword j = 0; j < numCols; ++j) {
      shifted[j] = std::max(0.0, X(i, j) - floorEps);
    }

    // Project onto simplex and add floor back
    projectOntoSimplex(shifted, adjustedSum, projected);
    for (uword j = 0; j < numCols; ++j) {
      X(i, j) = projected[j] + floorEps;
    }
  }
}

/**
 * Project matrix columns onto simplex with floor constraint
 * Each column sums to 1 with minimum element value eps
 * Optimized to reuse buffers and avoid allocations
 */
inline void projectColsOntoSimplexWithFloor(mat& B, double floorEps) {
  const uword numRows = B.n_rows;
  const uword numCols = B.n_cols;
  const double adjustedSum = 1.0 - double(numRows) * floorEps;

  // Reuse buffers to avoid repeated allocations
  rowvec shifted(numRows);
  rowvec projected(numRows);

  for (uword d = 0; d < numCols; ++d) {
    double* b_ptr = B.colptr(d);

    // Shift by floor value
    for (uword j = 0; j < numRows; ++j) {
      shifted[j] = std::max(0.0, b_ptr[j] - floorEps);
    }

    // Project onto simplex and add floor back
    projectOntoSimplex(shifted, adjustedSum, projected);
    for (uword j = 0; j < numRows; ++j) {
      b_ptr[j] = projected[j] + floorEps;
    }
  }
}

// ============================================================================
// Linear Algebra Utilities
// ============================================================================

/**
 * Compute row-wise dot product between two matrices
 * Returns vector where element i = dot(X.row(i), S.row(i))
 * Optimized for cache-friendly access
 */
inline vec computeRowwiseDotProduct(const mat& X, const mat& S) {
  const uword n_rows = X.n_rows;
  const uword n_cols = X.n_cols;
  vec result(n_rows, fill::zeros);

  // Cache-friendly row-wise computation - FIX: direct element access
  for (uword i = 0; i < n_rows; ++i) {
    double sum = 0.0;
    for (uword j = 0; j < n_cols; ++j) {
      sum += X(i, j) * S(i, j);
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Estimate largest eigenvalue of sparse matrix using power iteration
 * Used for computing Lipschitz constant bounds
 */
inline double estimateLargestEigenvalue(const sp_mat& M, int maxIterations = 60) {
  vec v = randn<vec>(M.n_rows);
  double normV = norm(v, 2);
  
  if (normV > 0) {
    v /= normV;
  }
  
  for (int iter = 0; iter < maxIterations; ++iter) {
    vec w = M * v;
    normV = norm(w, 2);
    
    if (normV == 0.0) {
      break;
    }
    
    v = w / normV;
  }
  
  return std::max(0.0, as_scalar(v.t() * (M * v)));
}

// ============================================================================
// Objective Function Components
// ============================================================================

/**
 * Compute negative log-likelihood term
 * -sum(log(rowSums(S .* (X * B^T))))
 */
inline double computeNegativeLogLikelihood(const mat& X, const mat& S, const mat& B, 
                                          double minDenominator = 1e-12) {
  mat Z = X * B.t();  // N x K
  vec denominators = sum(S % Z, 1);
  
  // Ensure numerical stability
  denominators.transform([minDenominator](double d) { 
    return std::max(d, minDenominator); 
  });
  
  return -accu(log(denominators));
}

/**
 * Compute weighted soft-L1 Total Variation term
 * sum_i w_i * sum_j sqrt((AX)_{ij}^2 + delta)
 */
inline double computeWeightedSoftL1TV(const mat& X, const sp_mat& A, 
                                     const vec& weights, double delta) {
  mat U = mat(A * X);  // m x D
  mat softL1 = sqrt(square(U) + delta);  // Element-wise soft-L1
  vec edgeContributions = softL1 * ones<vec>(softL1.n_cols);
  return dot(weights, edgeContributions);
}

/**
 * Compute full objective function value
 */
inline double computeObjective(const mat& X, const mat& S, const sp_mat& A,
                              const vec& weights, double lambda, 
                              const mat& B, double delta) {
  return computeNegativeLogLikelihood(X, S, B) + 
    lambda * computeWeightedSoftL1TV(X, A, weights, delta);
}

/**
 * Compute gradient mapping norm for convergence checking
 */
inline double computeGradientMappingNorm(const mat& X, const mat& S,
                                        const sp_mat& A, const sp_mat& AtA,
                                        const vec& weights, double lambda,
                                        double stepSize, double projEps,
                                        const mat& B, double delta) {
  const double min_denom = 1e-12;

  // Compute data gradient
  mat Z = X * B.t();
  vec denominators(S.n_rows);
  denominators.zeros();

  // FIX: Use direct element access instead of memptr()
  for (uword i = 0; i < S.n_rows; ++i) {
    double sum = 0.0;
    for (uword j = 0; j < S.n_cols; ++j) {
      sum += S(i, j) * Z(i, j);
    }
    denominators[i] = std::max(sum, min_denom);
  }

  mat dataGrad = -S;
  dataGrad.each_col() /= denominators;
  mat gradX = dataGrad * B;  // N x D

  // Compute TV gradient: A^T(diag(w) * U/sqrt(U^2 + delta))
  mat U = mat(A * X);
  mat V = U;  // Reuse memory
  const double* w_ptr = weights.memptr();
  
  // FIX: Use direct element access for row operations
  for (uword i = 0; i < V.n_rows; ++i) {
    const double w_val = w_ptr[i];
    for (uword j = 0; j < V.n_cols; ++j) {
      const double u_val = V(i, j);
      V(i, j) = w_val * u_val / std::sqrt(u_val * u_val + delta);
    }
  }

  mat tvGrad = lambda * mat(A.t() * V);
  mat totalGrad = gradX + tvGrad;

  // Compute gradient mapping
  mat Xtest = X - stepSize * totalGrad;
  projectRowsOntoSimplexWithFloor(Xtest, projEps);
  mat gradMapping = (X - Xtest) / stepSize;

  return norm(gradMapping, "fro");
}

// ============================================================================
// FISTA Optimization Functions
// ============================================================================

/**
 * FISTA optimization for X with fixed B
 * Minimizes NLL + weighted soft-L1 TV subject to row-wise simplex constraints
 */
void optimizeX_FISTA(const mat& S, const sp_mat& A, const sp_mat& AtA,
                     const vec& weights, double lambda,
                     double projEps, double stepEps, bool normalizeForBound,
                     int maxIterations, double safetyFactor, int logFrequency,
                     mat& X, const mat& B, double delta, double& stepSizeOut) {
  
  const double min_denom = 1e-12;
  
  // Compute Lipschitz constant for smooth part (data term)
  mat BtB = B.t() * B;
  double dataNorm = norm(BtB, 2);
  
  // Compute TV part contribution
  double tvNorm = estimateLargestEigenvalue(AtA);
  if (!std::isfinite(tvNorm)) {
    tvNorm = 0.0;
  }
  
  // Combine into total Lipschitz constant
  double lipschitz;
  if (normalizeForBound) {
    double nPoints = double(S.n_rows);
    lipschitz = dataNorm / nPoints + lambda * tvNorm;
  } else {
    lipschitz = dataNorm + lambda * tvNorm;
  }
  
  if (!std::isfinite(lipschitz) || lipschitz <= 0) {
    lipschitz = accu(BtB) + lambda * accu(AtA);
  }
  
  double stepSize = safetyFactor / std::max(1e-12, lipschitz / (stepEps * stepEps));
  stepSizeOut = stepSize;
  
  // Initialize FISTA variables
  mat Xk = X, Yk = Xk;
  
  if (logFrequency > 0) {
    Rprintf("  X-step: step=%.2e, L_data=%.2e, L_tv=%.2e\n", 
            stepSize, dataNorm, lambda * tvNorm);
  }
  
  // Pre-allocate matrices
  mat Z(S.n_rows, B.n_rows);
  vec denominators(S.n_rows);
  mat dataGrad(S.n_rows, S.n_cols);
  mat gradX(X.n_rows, X.n_cols);
  mat U(A.n_rows, X.n_cols);
  mat V(A.n_rows, X.n_cols);
  mat tvGrad(X.n_rows, X.n_cols);
  
  double lastObj = std::numeric_limits<double>::infinity();

  // FISTA iterations
  for (int t = 1; t <= maxIterations; ++t) {
    // === Data Gradient ===
    Z = Yk * B.t();
    
    // Compute denominators - FIX: direct element access
    denominators.zeros();
    for (uword i = 0; i < S.n_rows; ++i) {
      double sum = 0.0;
      for (uword j = 0; j < S.n_cols; ++j) {
        sum += S(i, j) * Z(i, j);
      }
      denominators[i] = std::max(sum, min_denom);
    }
    
    dataGrad = -S;
    dataGrad.each_col() /= denominators;
    gradX = dataGrad * B;

    // === TV Gradient ===
    U = mat(A * Yk);
    V = U;
    
    // FIX: Use direct element access
    const double* w_ptr = weights.memptr();
    for (uword i = 0; i < V.n_rows; ++i) {
      const double w_val = w_ptr[i];
      for (uword j = 0; j < V.n_cols; ++j) {
        const double u_val = V(i, j);
        V(i, j) = w_val * u_val / std::sqrt(u_val * u_val + delta);
      }
    }
    
    tvGrad = lambda * mat(A.t() * V);

    // === FISTA Update ===
    mat Xnew = Yk - stepSize * (gradX + tvGrad);
    projectRowsOntoSimplexWithFloor(Xnew, projEps);

    double momentum = double(t - 1) / double(t + 2);
    Yk = Xnew + momentum * (Xnew - Xk);
    Xk = Xnew;
    
    // Logging
    if (logFrequency > 0 && (t % logFrequency == 0 || t == maxIterations)) {
      double currentObj = computeObjective(Xk, S, A, weights, lambda, B, delta);
      
      double sumViolation = abs(sum(Xk, 1) - 1.0).max();
      double floorViolation = std::max(0.0, projEps - Xk.min());
      double primalResidual = std::max(sumViolation, floorViolation);
      
      double dualResidual = computeGradientMappingNorm(
        Xk, S, A, AtA, weights, lambda, stepSize, projEps, B, delta);
      
      double relativeGap = std::abs(currentObj - lastObj) / 
        std::max(1.0, std::abs(currentObj));
      
      Rprintf("  X iter %5d: primal=%.2e, dual=%.2e, gap=%.2e, obj=%.6e\n", 
              t, primalResidual, dualResidual, relativeGap, currentObj);
      
      lastObj = currentObj;
    }
  }
  
  X = Xk;
}

/**
 * FISTA optimization for B with fixed X
 * Minimizes NLL subject to column-wise simplex constraints
 */
void optimizeB_FISTA(const mat& S, const mat& X, mat& B,
                     double projEps, double stepEps, int maxIterations,
                     double safetyFactor, int logFrequency,
                     double& stepSizeOut) {
  
  // Compute Lipschitz constant
  mat XtX = X.t() * X;
  double lipschitz = norm(XtX, 2);
  if (!std::isfinite(lipschitz) || lipschitz <= 0) {
    lipschitz = accu(XtX);
  }
  
  double stepSize = safetyFactor / std::max(1e-12, lipschitz / (stepEps * stepEps));
  stepSizeOut = stepSize;
  
  // Initialize FISTA variables
  mat Bk = B, Yk = Bk;
  
  if (logFrequency > 0) {
    Rprintf("  B-step: step=%.2e, L=%.2e\n", stepSize, lipschitz);
  }
  
  // Pre-allocate matrices to avoid repeated allocations in loop
  mat Z(S.n_rows, B.n_rows);
  vec denominators(S.n_rows);
  mat gradZ(S.n_rows, S.n_cols);
  mat gradB(B.n_rows, B.n_cols);
  const double min_denom = 1e-12;

  // FISTA iterations
  for (int t = 1; t <= maxIterations; ++t) {
    // Compute gradient: Z = X * Yk.t()
    Z = X * Yk.t();

    // Compute denominators with stability check - FIX: direct element access
    denominators.zeros();
    for (uword i = 0; i < S.n_rows; ++i) {
      double sum = 0.0;
      for (uword j = 0; j < S.n_cols; ++j) {
        sum += S(i, j) * Z(i, j);
      }
      denominators[i] = std::max(sum, min_denom);
    }

    // Compute gradZ = -S / denominators
    gradZ = -S;
    gradZ.each_col() /= denominators;
    gradB = gradZ.t() * X;  // K x D

    // FISTA update with momentum
    mat Bnew = Yk - stepSize * gradB;
    projectColsOntoSimplexWithFloor(Bnew, projEps);

    double momentum = double(t - 1) / double(t + 2);
    Yk = Bnew + momentum * (Bnew - Bk);
    Bk = Bnew;
    
    // Logging
    if (logFrequency > 0 && (t % logFrequency == 0 || t == maxIterations)) {
      Rprintf("  B iter %5d: ||grad||_F=%.2e\n", t, norm(gradB, "fro"));
    }
  }
  
  B = Bk;
}

} // end anonymous namespace

// ============================================================================
// Exported Functions
// ============================================================================

/**
 * Main optimization function: alternating FISTA for X and B
 * 
 * @param S_mat Data matrix (N x K)
 * @param A Adjacency/incidence matrix for TV (m x N, sparse)
 * @param B Dictionary matrix (K x D, columns on simplex)
 * @param w Edge weights for TV (m x 1)
 * @param lambda Regularization parameter
 * @param delta Soft-L1 smoothing parameter
 * @param it_X Max iterations for X optimization
 * @param eps_proj_X Floor constraint for X rows
 * @param eps_step_X Step size denominator bound for X
 * @param normalize_for_bound Use normalized bound for Lipschitz constant
 * @param safety_X Safety factor for X step size
 * @param log_every_X Logging frequency for X optimization
 * @param it_B Max iterations for B optimization
 * @param eps_proj_B Floor constraint for B columns
 * @param eps_step_B Step size denominator bound for B
 * @param safety_B Safety factor for B step size
 * @param log_every_B Logging frequency for B optimization
 * @param update_B Whether to update B matrix
 * @param X0_opt Optional initial X matrix
 * 
 * @return List containing optimized X, B, objective value, and components
 */
// [[Rcpp::export]]
Rcpp::List xb_fista_wfixed(const arma::mat& S_mat,
                           const arma::sp_mat& A,
                           arma::mat B,
                           const arma::vec& w,
                           const double lambda = 1.0,
                           const double delta = 1e-6,
                           const int it_X = 1500,
                           const double eps_proj_X = 1e-6,
                           const double eps_step_X = 1e-3,
                           const bool normalize_for_bound = true,
                           const double safety_X = 0.95,
                           const int log_every_X = 250,
                           const int it_B = 500,
                           const double eps_proj_B = 1e-6,
                           const double eps_step_B = 1e-3,
                           const double safety_B = 0.95,
                           const int log_every_B = 250,
                           const bool update_B = true,
                           Rcpp::Nullable<arma::mat> X0_opt = R_NilValue) {
  
  // Validate dimensions
  if (S_mat.n_cols != B.n_rows) {
    stop("Dimension mismatch: ncol(S_mat) must equal nrow(B).");
  }
  
  const uword N = S_mat.n_rows;
  const uword D = B.n_cols;
  const uword m = A.n_rows;
  
  if (w.n_elem != m) {
    stop("Length of w must equal nrow(A).");
  }
  
  // Initialize X
  arma::mat X;
  if (X0_opt.isNotNull()) {
    X = Rcpp::as<arma::mat>(X0_opt);
    if (X.n_rows != N || X.n_cols != D) {
      stop("X0 has wrong shape.");
    }
  } else {
    X = arma::mat(N, D, arma::fill::ones) / double(D);
  }
  
  // Project initial values onto constraints
  projectRowsOntoSimplexWithFloor(X, eps_proj_X);
  projectColsOntoSimplexWithFloor(B, eps_proj_B);
  
  // Precompute A^T * A for efficiency
  arma::sp_mat AtA = A.t() * A;
  
  // Optimize X with fixed B
  double stepX = NA_REAL;
  optimizeX_FISTA(S_mat, A, AtA, w, lambda,
                  eps_proj_X, eps_step_X, normalize_for_bound,
                  it_X, safety_X, log_every_X, 
                  X, B, delta, stepX);
  
  // Optionally optimize B with fixed X
  double stepB = NA_REAL;
  if (update_B && it_B > 0) {
    optimizeB_FISTA(S_mat, X, B, 
                    eps_proj_B, eps_step_B, it_B, 
                    safety_B, log_every_B, stepB);
  }
  
  // Compute final objective and components
  double nll = computeNegativeLogLikelihood(X, S_mat, B);
  double tv = computeWeightedSoftL1TV(X, A, w, delta);
  double obj = nll + lambda * tv;
  
  return Rcpp::List::create(
    Named("X") = X,
    Named("B") = B,
    Named("obj") = obj,
    Named("nll") = nll,
    Named("tv") = tv,
    Named("stepX") = stepX,
    Named("stepB") = stepB
  );
}

/**
 * Evaluate objective function without optimization
 * 
 * @param S_mat Data matrix (N x K)
 * @param A Adjacency/incidence matrix for TV (m x N, sparse)
 * @param X Solution matrix (N x D)
 * @param B Dictionary matrix (K x D)
 * @param w Edge weights for TV (m x 1)
 * @param lambda Regularization parameter
 * @param delta Soft-L1 smoothing parameter
 * 
 * @return List containing objective value and components
 */
// [[Rcpp::export]]
Rcpp::List eval_softL1_w_cpp(const arma::mat& S_mat,
                             const arma::sp_mat& A,
                             const arma::mat& X,
                             const arma::mat& B,
                             const arma::vec& w,
                             const double lambda = 1.0,
                             const double delta = 1e-6) {
  
  // Validate dimensions
  if (S_mat.n_cols != B.n_rows) {
    stop("Dimension mismatch: ncol(S_mat) != nrow(B).");
  }
  if (X.n_cols != B.n_cols) {
    stop("Dimension mismatch: ncol(X) != ncol(B).");
  }
  if (S_mat.n_rows != X.n_rows) {
    stop("Dimension mismatch: nrow(S_mat) != nrow(X).");
  }
  if (w.n_elem != A.n_rows) {
    stop("Dimension mismatch: length(w) != nrow(A).");
  }
  
  // Compute objective components
  double nll = computeNegativeLogLikelihood(X, S_mat, B);
  double tv = computeWeightedSoftL1TV(X, A, w, delta);
  double obj = nll + lambda * tv;
  
  return Rcpp::List::create(
    Named("obj") = obj,
    Named("nll") = nll,
    Named("tv") = tv
  );
}