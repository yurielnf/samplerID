#pragma once
#include "interp_decomp.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <type_traits>

template<typename T = double>
class TensorTrain {
    static constexpr auto dtype_v = c10::CppTypeToScalarType<T>::value;
public:
    // cores[k] has shape [r_k, d_k, r_{k+1}] with r_0 = r_n = 1.
    std::vector<torch::Tensor> cores;

    explicit TensorTrain(int ncores) : cores(ncores) {}
    explicit TensorTrain(std::vector<torch::Tensor> c) : cores(std::move(c)) {}

    // TT-ID decomposition; tol=0 keeps full rank.
    explicit TensorTrain(const torch::Tensor& tensor, double tol = 0.0) {
        auto t = tensor.to(dtype_v);
        auto shape = t.sizes().vec();
        int64_t n = (int64_t)shape.size();
        TORCH_CHECK(n >= 2, "tensor must have at least 2 dimensions");

        int64_t r = 1;
        auto C = t.reshape({1, t.numel()});

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

    torch::Tensor to_tensor() const {
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

    T eval(const std::vector<int>& idx) const {
        TORCH_CHECK(idx.size() == cores.size(), "index size mismatch");
        auto v = cores[0].select(0, 0).select(0, (int64_t)idx[0]);
        for (size_t k = 1; k < cores.size(); ++k)
            v = torch::mv(cores[k].select(1, (int64_t)idx[k]).t(), v);
        return v[0].item<T>();
    }

    T sum() const {
        auto v = cores[0].sum(1).squeeze(0);
        for (size_t k = 1; k < cores.size(); ++k)
            v = torch::mv(cores[k].sum(1).t(), v);
        return v[0].item<T>();
    }

    T overlap(const TensorTrain& other) const {
        TORCH_CHECK(cores.size() == other.cores.size(), "TT length mismatch");
        auto env = torch::ones({1, 1}, cores[0].options());
        for (size_t k = 0; k < cores.size(); ++k)
            env = torch::einsum("ab,asc,bsd->cd", {env, cores[k], other.cores[k].conj()});
        return env[0][0].item<T>();
    }

    double norm() const {
        T ov = overlap(*this);
        if constexpr (std::is_floating_point_v<T>)
            return std::sqrt(std::abs((double)ov));
        else
            return std::sqrt(std::abs((double)ov.real()));
    }
};

// Deduction guides so TensorTrain tt(tensor) → TensorTrain<double>
TensorTrain(const torch::Tensor&)        -> TensorTrain<double>;
TensorTrain(const torch::Tensor&, double) -> TensorTrain<double>;
TensorTrain(int)                          -> TensorTrain<double>;
TensorTrain(std::vector<torch::Tensor>)   -> TensorTrain<double>;
