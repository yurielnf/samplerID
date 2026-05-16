#pragma once
#include "interp_decomp.h"
#include "index_set.h"
#include "tt.h"
#include <functional>
#include <map>
#include <vector>

namespace ttid {

using namespace xfac;

struct tt_id_param {
    int bondDim = 30;
    double reltol = 1e-12;
    std::vector<int> pivot1;
    bool use_cache = false;
};

// Interpolative decomposition of a tensor given as a function f:(i_0,...,i_{n-1})->T.
// Supported scalar types T: float, double, c10::complex<float>, c10::complex<double>.
template<typename T = double>
struct tt_id {
    using tensor_fun = std::function<T(std::vector<int>)>;

    tensor_fun f;
    tt_id_param param;
    std::vector<double> pivotError;
    std::vector<IndexSet<MultiIndex>> Iset, localSet, Jset;
    TensorTrain<T> tt;
    int cIter = 0, center = 0;

    tt_id() = default;

    tt_id(tensor_fun const& f_, std::vector<int> localDim, tt_id_param param_ = {})
        : f{f_}, param(param_), pivotError(1),
          Iset(localDim.size()), localSet(localDim.size()), Jset(localDim.size()),
          tt(localDim.size())
    {
        if (param.pivot1.empty()) param.pivot1.resize(localDim.size(), 0);
        pivotError[0] = std::abs(f(param.pivot1));
        if (pivotError[0] == 0.0)
            throw std::invalid_argument("f(pivot1)=0; provide a better first pivot");

        for (auto p = 0u; p < localDim.size(); p++)
            for (auto i = 0; i < localDim[p]; i++)
                localSet[p].push_back({char32_t(i)});

        for (auto b = 0u; b < localDim.size(); b++) {
            Iset[b].push_back({param.pivot1.begin(), param.pivot1.begin() + b});
            Jset[b].push_back({param.pivot1.begin() + b + 1, param.pivot1.end()});
        }
        iterate(1, 1);
    }

    void addPivotsAllBonds(std::vector<std::vector<int>> const& pivots)
    {
        for (auto i : {1, 2}) {
            (void)i;
            if (cIter % 2 == 0)
                for (auto b = 0u; b < len() - 1; b++) addPivotsAt(pivots, b);
            else
                for (int b = (int)len() - 2; b >= 0; b--) addPivotsAt(pivots, b);
            iterate(1, 2);
        }
    }

    void addPivotsAt(std::vector<std::vector<int>> const& pivots, int b)
    {
        for (const auto& pg : pivots) {
            if (cIter % 2 == 0)
                Jset[b].push_back({pg.begin() + b + 1, pg.end()});
            else
                Iset[b + 1].push_back({pg.begin(), pg.begin() + b + 1});
        }
    }

    size_t len() const { return localSet.size(); }

    void iterate(int nIter = 1, int dmrg_type = 2)
    {
        for (auto i = 0; i < nIter; i++) {
            if (cIter % 2 == 0)
                for (auto b = 0u; b < len() - 1; b++) { center = b + 1; updatePivotAt(b, dmrg_type); }
            else
                for (int b = (int)len() - 2; b >= 0; b--) { center = b; updatePivotAt(b, dmrg_type); }
            cIter++;
        }
    }

protected:
    static constexpr auto dtype_v = c10::CppTypeToScalarType<T>::value;

    void updatePivotAt(int b, int dmrg = 2)
    {
        if (dmrg == 1) dmrg1_updatePivotAt(b);
        else           dmrg2_updatePivotAt(b);
    }

    void dmrg2_updatePivotAt(int b)
    {
        IndexSet<MultiIndex> Ib = kron(Iset[b], localSet[b]);
        IndexSet<MultiIndex> Jb = kron(localSet[b + 1], Jset[b + 1]);
        auto M = tensor_fun_to_mat(Ib, Jb);

        if (b < center) {
            auto res = interp_decomp_rows(M, param.reltol);
            Iset[b + 1] = Ib.at(res.rows);
            Jset[b] = Iset[b + 1];
            tt.cores[b] = res.P;
            tt.cores[b + 1] = M.index({torch::tensor(res.rows, torch::kLong), torch::indexing::Slice()});
            collectPivotError(res.sv);
        } else {
            auto res = interp_decomp_cols(M, param.reltol);
            Jset[b] = Jb.at(res.cols);
            Iset[b + 1] = Jset[b];
            tt.cores[b] = M.index({torch::indexing::Slice(), torch::tensor(res.cols, torch::kLong)});
            tt.cores[b + 1] = res.P;
            collectPivotError(res.sv);
        }
        reshape_site_tensor(b);
        reshape_site_tensor(b + 1);
    }

    void dmrg1_updatePivotAt(int b)
    {
        bool isLeft = b < center;
        IndexSet<MultiIndex> Ib = isLeft ? kron(Iset[b], localSet[b]) : Iset[b + 1].from_int();
        IndexSet<MultiIndex> Jb = isLeft ? Jset[b].from_int() : kron(localSet[b + 1], Jset[b + 1]);
        auto M = tensor_fun_to_mat(Ib, Jb);

        if (isLeft) {
            auto res = interp_decomp_rows(M, param.reltol);
            Iset[b + 1] = Ib.at(res.rows);
            Jset[b] = Iset[b + 1];
            tt.cores[b] = res.P;
            tt.cores[b + 1] = M.index({torch::tensor(res.rows, torch::kLong), torch::indexing::Slice()});
            collectPivotError(res.sv);
        } else {
            auto res = interp_decomp_cols(M, param.reltol);
            Jset[b] = Jb.at(res.cols);
            tt.cores[b] = M.index({torch::indexing::Slice(), torch::tensor(res.cols, torch::kLong)});
            tt.cores[b + 1] = res.P;
            collectPivotError(res.sv);
        }
        reshape_site_tensor(b);
    }

    torch::Tensor tensor_fun_to_mat(const std::vector<MultiIndex>& I, const std::vector<MultiIndex>& J) const
    {
        auto M = torch::empty({(int64_t)I.size(), (int64_t)J.size()},
                              torch::TensorOptions().dtype(dtype_v));
        auto ptr = M.template data_ptr<T>();
        for (size_t i = 0; i < I.size(); i++)
            for (size_t j = 0; j < J.size(); j++) {
                MultiIndex ij = I[i] + J[j];
                std::vector<int> key(ij.begin(), ij.end());
                ptr[i * J.size() + j] = cached_eval(key);
            }
        return M;
    }

    void reshape_site_tensor(int i)
    {
        int64_t a = Iset[i].size(), b = localSet[i].size(), c = Jset[i].size();
        tt.cores[i] = tt.cores[i].reshape({a, b, c});
    }

    void collectPivotError(std::vector<double> const& pe)
    {
        if (pe.size() > pivotError.size()) pivotError.resize(pe.size(), 0);
        for (size_t i = 0; i < pe.size(); i++)
            if (pe[i] > pivotError[i]) pivotError[i] = pe[i];
    }

    T cached_eval(const std::vector<int>& key) const
    {
        if (!param.use_cache) return f(key);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        return cache[key] = f(key);
    }

private:
    mutable std::map<std::vector<int>, T> cache;
};

} // namespace ttid
