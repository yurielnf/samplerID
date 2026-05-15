#pragma once
#include <torch/torch.h>
#include <vector>

class TensorTrain {
public:
    // cores[k] has shape [r_k, d_k, r_{k+1}] with r_0 = r_n = 1.
    std::vector<torch::Tensor> cores;

    explicit TensorTrain(int ncores) : cores(ncores) {}
    explicit TensorTrain(std::vector<torch::Tensor> cores);

    // TT-ID decomposition of an n-dimensional tensor.
    // tol: relative R-diagonal cutoff for rank selection (0 = exact, keep all).
    explicit TensorTrain(const torch::Tensor& tensor, double tol = 0.0);

    // Reconstruct the full dense tensor by contracting all cores.
    torch::Tensor to_tensor() const;

    // Evaluate the TT at a single multi-index (accepts any integer vector type).
    template<typename IndexType>
    double eval(const std::vector<IndexType>& indices) const {
        TORCH_CHECK(indices.size() == cores.size(),
                    "indices size must match number of cores");
        auto v = cores[0].select(0, 0).select(0, (int64_t)indices[0]);
        for (size_t k = 1; k < cores.size(); ++k) {
            TORCH_CHECK((int64_t)indices[k] < cores[k].size(1),
                        "index out of range at dim ", k);
            v = torch::mv(cores[k].select(1, (int64_t)indices[k]).t(), v);
        }
        return v[0].item<double>();
    }
};
