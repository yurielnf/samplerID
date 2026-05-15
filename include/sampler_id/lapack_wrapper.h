#pragma once
#include <torch/torch.h>
#include <tuple>
#include <vector>

// Rank-revealing QR: A[:,P] = Q R
// Returns (R upper-triangular min(m,n)×n, P permutation 0-indexed).
// Supported dtypes: float32, float64, complex64, complex128.
std::tuple<torch::Tensor, std::vector<int>> rrQR(const torch::Tensor& A);
