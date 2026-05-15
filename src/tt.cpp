#include "sampler_id/tt.h"

TensorTrain::TensorTrain(std::vector<torch::Tensor> c) : cores(std::move(c)) {}

// TT-ID decomposition: at each step unmatrix C into [r*d_k, rest], apply row ID,
// store the interpolation matrix as core k, carry the selected rows forward.
TensorTrain::TensorTrain(const torch::Tensor& tensor, double tol) {
    auto shape = tensor.sizes().vec();
    int64_t n = (int64_t)shape.size();
    TORCH_CHECK(n >= 2, "tensor must have at least 2 dimensions");

    int64_t r = 1;
    auto C = tensor.reshape({1, tensor.numel()});

    for (int64_t k = 0; k < n - 1; ++k) {
        int64_t dk = shape[k], rest = C.numel() / (r * dk);
        auto M = C.reshape({r * dk, rest});
        auto [rows, Q, sv] = (tol > 0.0)
            ? interp_decomp_rows(M, tol)
            : interp_decomp_rows(M, std::min(r * dk, rest));
        int64_t r_new = (int64_t)rows.size();
        cores.push_back(Q.reshape({r, dk, r_new}));
        C = M.index({torch::tensor(rows, torch::kLong), torch::indexing::Slice()});
        r = r_new;
    }
    cores.push_back(C.reshape({r, shape[n - 1], 1}));
}

torch::Tensor TensorTrain::to_tensor() const {
    TORCH_CHECK(!cores.empty(), "TensorTrain has no cores");
    auto result = cores[0].squeeze(0);  // [d_0, r_1]
    for (size_t k = 1; k < cores.size(); ++k) {
        auto shape  = result.sizes().vec();
        int64_t r_k = shape.back(), dk = cores[k].size(1), r_next = cores[k].size(2);
        auto prod   = torch::mm(result.reshape({result.numel() / r_k, r_k}),
                                cores[k].reshape({r_k, dk * r_next}));
        auto new_shape = std::vector<int64_t>(shape.begin(), shape.end() - 1);
        new_shape.push_back(dk); new_shape.push_back(r_next);
        result = prod.reshape(new_shape);
    }
    return result.squeeze(-1);
}
