#pragma once
#include <torch/torch.h>
#include <cstdint>
#include <vector>

namespace ttid {

struct ColID {
    std::vector<int> cols;  // k selected column indices (0-based)
    torch::Tensor P;        // k×n:  A ≈ A[:,cols] * P
    std::vector<double> sv; // singular value estimates
};

struct RowID {
    std::vector<int> rows;  // k selected row indices (0-based)
    torch::Tensor P;        // m×k:  A ≈ P * A[rows,:]
    std::vector<double> sv;
};

ColID interp_decomp_cols(const torch::Tensor& A, double tol = 0.0, int64_t k = INT64_MAX);
RowID interp_decomp_rows(const torch::Tensor& A, double tol = 0.0, int64_t k = INT64_MAX);

} // namespace ttid
