#pragma once
#include <torch/torch.h>
#include <vector>

struct IDResult {
    std::vector<int> cols;      /// k selected column indices (0-based)
    torch::Tensor P;            /// k×n matrix: A ≈ A[:, cols] * P
    std::vector<double> sv;     /// singular values
};

// Column interpolative decomposition — fixed rank k.
IDResult id_cols(const torch::Tensor& A, int64_t k);

// Column interpolative decomposition — rank chosen by tolerance.
// Keeps columns while |R[i,i]| / |R[0,0]| >= tol  (R from pivoted QR).
IDResult id_cols(const torch::Tensor& A, double tol);

struct RowIDResult {
    std::vector<int> rows;      /// k selected row indices (0-based)
    torch::Tensor P;            /// m×k matrix: A ≈ P * A[rows, :]
    std::vector<double> sv;     /// singular values
};

// Row interpolative decomposition — fixed rank k.
RowIDResult id_rows(const torch::Tensor& A, int64_t k);

// Row interpolative decomposition — rank chosen by tolerance.
RowIDResult id_rows(const torch::Tensor& A, double tol);
