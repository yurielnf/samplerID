#include "tt.h"
#include "id.h"

TensorTrain::TensorTrain(std::vector<torch::Tensor> cores)
    : cores(std::move(cores)) {}

// TT-ID decomposition.
//
// At each step k the remaining unfolded tensor C has shape [r, d_k * ... * d_{n-1}].
// We reshape to M = [r * d_k, d_{k+1} * ... * d_{n-1}], run column ID on M, store
// M[:, J] as core k, and carry forward P (the k×rest interpolation matrix) as C.
TensorTrain::TensorTrain(const torch::Tensor& tensor, double tol) {
    auto shape = tensor.sizes().vec();
    int64_t n = static_cast<int64_t>(shape.size());
    TORCH_CHECK(n >= 2, "tensor must have at least 2 dimensions");

    int64_t r = 1;
    auto C = tensor.reshape({1, tensor.numel()});  // [1, d_0 * ... * d_{n-1}]

    for (int64_t k = 0; k < n - 1; ++k) {
        int64_t dk   = shape[k];
        int64_t rest = C.numel() / (r * dk);

        auto M = C.reshape({r * dk, rest});
        auto [rows, Q, sv] = (tol > 0.0)
            ? id_rows(M, tol)
            : id_rows(M, std::min(r * dk, rest));

        int64_t r_new = static_cast<int64_t>(rows.size());
        cores.push_back(Q.reshape({r, dk, r_new}));
        C = M.index({torch::tensor(rows, torch::kLong), torch::indexing::Slice()});
        r = r_new;
    }

    cores.push_back(C.reshape({r, shape[n - 1], 1}));
}

// Reconstruction contracts cores left to right.
// After step k, `result` has shape [d_0, ..., d_k, r_{k+1}].
// Contracting with core k+1 (shape [r_{k+1}, d_{k+1}, r_{k+2}]):
//   reshape result to [batch, r_{k+1}], multiply by core reshaped to
//   [r_{k+1}, d_{k+1} * r_{k+2}], then unflatten.
torch::Tensor TensorTrain::to_tensor() const {
    TORCH_CHECK(!cores.empty(), "TensorTrain has no cores");

    auto result = cores[0].squeeze(0);  // [d_0, r_1]

    for (size_t k = 1; k < cores.size(); ++k) {
        auto shape  = result.sizes().vec();
        int64_t r_k    = shape.back();
        int64_t dk     = cores[k].size(1);
        int64_t r_next = cores[k].size(2);
        int64_t batch  = result.numel() / r_k;

        auto prod = torch::mm(result.reshape({batch, r_k}),
                              cores[k].reshape({r_k, dk * r_next}));

        auto new_shape = std::vector<int64_t>(shape.begin(), shape.end() - 1);
        new_shape.push_back(dk);
        new_shape.push_back(r_next);
        result = prod.reshape(new_shape);
    }

    return result.squeeze(-1);
}

