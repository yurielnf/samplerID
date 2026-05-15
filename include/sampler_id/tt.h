#pragma once
#include "interp_decomp.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>

class TensorTrain {
public:
    // cores[k] has shape [r_k, d_k, r_{k+1}] with r_0 = r_n = 1.
    std::vector<torch::Tensor> cores;

    explicit TensorTrain(int ncores) : cores(ncores) {}
    explicit TensorTrain(std::vector<torch::Tensor> cores);

    // TT-ID decomposition; tol=0 keeps full rank.
    explicit TensorTrain(const torch::Tensor& tensor, double tol = 0.0);

    torch::Tensor to_tensor() const;

    template<typename T = double>
    T eval(const std::vector<int>& idx) const {
        TORCH_CHECK(idx.size() == cores.size(), "index size mismatch");
        auto v = cores[0].select(0, 0).select(0, (int64_t)idx[0]);
        for (size_t k = 1; k < cores.size(); ++k)
            v = torch::mv(cores[k].select(1, (int64_t)idx[k]).t(), v);
        return v[0].item<T>();
    }

    // Sum over all multi-indices: sum_i TT(i).
    template<typename T = double>
    T sum() const {
        auto v = cores[0].sum(1).squeeze(0);  // sum phys dim → [r_1]
        for (size_t k = 1; k < cores.size(); ++k)
            v = torch::mv(cores[k].sum(1).t(), v);
        return v[0].item<T>();
    }

    // Inner product: sum_i conj(other(i)) * this(i)  (for real: dot product).
    template<typename T = double>
    T overlap(const TensorTrain& other) const {
        TORCH_CHECK(cores.size() == other.cores.size(), "TT length mismatch");
        auto env = torch::ones({1, 1}, cores[0].options());
        for (size_t k = 0; k < cores.size(); ++k)
            env = torch::einsum("ab,asc,bsd->cd", {env, cores[k], other.cores[k].conj()});
        return env[0][0].item<T>();
    }

    // Frobenius norm: sqrt(|<this|this>|).
    template<typename T = double>
    double norm() const {
        return std::sqrt(std::abs(overlap<T>(*this)));
    }
};
