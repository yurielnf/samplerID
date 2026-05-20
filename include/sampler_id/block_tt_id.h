#pragma once
#include "tt_id.h"

namespace ttid {

using namespace xfac;

// Interpolative decomposition of n functions simultaneously.
// All functions share the same Iset/Jset (bond structure).
// The TT has one "fat" center core of local dim n*d_center (carrying the function index alpha).
// eval(alpha, idx) evaluates function alpha at multi-index idx.
template<typename T = double>
struct block_tt_id {
    using tensor_fun = std::function<T(std::vector<int>)>;

    std::vector<tensor_fun> funs;
    tt_id_param param;
    std::vector<double> pivotError;
    std::vector<IndexSet<MultiIndex>> Iset, localSet, Jset;
    TensorTrain<T> tt;
    int cIter = 0, center = 0;

    block_tt_id() = default;

    block_tt_id(std::vector<tensor_fun> funs_,
                std::vector<int> localDim,
                tt_id_param param_ = {})
        : funs(std::move(funs_)), param(param_), pivotError(1),
          Iset(localDim.size()), localSet(localDim.size()), Jset(localDim.size()),
          tt((int)localDim.size())
    {
        if (param.pivot1.empty()) param.pivot1.resize(localDim.size(), 0);
        pivotError[0] = std::abs(funs[0](param.pivot1));
        if (pivotError[0] == 0.0)
            throw std::invalid_argument("funs[0](pivot1)=0; provide a better first pivot");

        for (auto p = 0u; p < localDim.size(); p++)
            for (auto i = 0; i < localDim[p]; i++)
                localSet[p].push_back({char32_t(i)});

        for (auto b = 0u; b < localDim.size(); b++) {
            Iset[b].push_back({param.pivot1.begin(), param.pivot1.begin() + (int)b});
            Jset[b].push_back({param.pivot1.begin() + (int)b + 1, param.pivot1.end()});
        }
        iterate(1);
    }

    size_t len()    const { return localSet.size(); }
    int    nFuncs() const { return (int)funs.size(); }

    void iterate(int nIter = 1) {
        for (int i = 0; i < nIter; i++) {
            if (cIter % 2 == 0)
                for (auto b = 0u; b < len() - 1; b++) { center = (int)b + 1; updateAt((int)b); }
            else
                for (int b = (int)len() - 2; b >= 0; b--) { center = b; updateAt(b); }
            cIter++;
        }
    }

    // Evaluate function alpha at multi-index idx.
    // At the center site: local index = alpha * d_center + idx[center].
    T eval(int alpha, const std::vector<int>& idx) const {
        TORCH_CHECK(idx.size() == len(), "index size mismatch");
        auto local_idx = [&](int k) -> int64_t {
            if (k == center)
                return (int64_t)(alpha * (int)localSet[k].size() + idx[k]);
            return (int64_t)idx[k];
        };
        auto v = tt.cores[0].select(0, 0).select(0, local_idx(0));
        for (size_t k = 1; k < len(); k++)
            v = torch::mv(tt.cores[k].select(1, local_idx((int)k)).t(), v);
        return v[0].template item<T>();
    }

private:
    static constexpr auto dtype_v = c10::CppTypeToScalarType<T>::value;

    // Returns shape m x (n*n_j); col = alpha*n_j + j
    torch::Tensor eval_block_right(const IndexSet<MultiIndex>& Ib,
                                   const IndexSet<MultiIndex>& Jb) const {
        int64_t m = Ib.size(), n_j = Jb.size(), n = nFuncs();
        auto M = torch::empty({m, n * n_j}, torch::TensorOptions().dtype(dtype_v));
        auto ptr = M.template data_ptr<T>();
        for (int64_t i = 0; i < m; i++)
            for (int64_t a = 0; a < n; a++)
                for (int64_t j = 0; j < n_j; j++)
                    ptr[i*(n*n_j) + a*n_j + j] = funs[a](multiIndex_as_vec(Ib.at((int)i) + Jb.at((int)j)));
        return M;
    }

    // Returns shape (n*m) x n_j; row = alpha*m + i
    torch::Tensor eval_block_left(const IndexSet<MultiIndex>& Ib,
                                  const IndexSet<MultiIndex>& Jb) const {
        int64_t m = Ib.size(), n_j = Jb.size(), n = nFuncs();
        auto M = torch::empty({n * m, n_j}, torch::TensorOptions().dtype(dtype_v));
        auto ptr = M.template data_ptr<T>();
        for (int64_t a = 0; a < n; a++)
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < n_j; j++)
                    ptr[(a*m + i)*n_j + j] = funs[a](multiIndex_as_vec(Ib.at((int)i) + Jb.at((int)j)));
        return M;
    }

    void updateAt(int b) {
        if (b < center) updateRight(b);
        else            updateLeft(b);
    }

    // Right sweep: alpha on right side of center bond.
    // row_id on M_block[i, (alpha,j)] => Iset[b+1] is alpha-free.
    // core[b] = P (normal),  core[b+1] = fat (carries alpha).
    void updateRight(int b) {
        IndexSet<MultiIndex> Ib = kron(Iset[b],       localSet[b]);
        IndexSet<MultiIndex> Jb = kron(localSet[b+1], Jset[b+1]);
        int64_t m = (int64_t)Ib.size(), n_j = (int64_t)Jb.size();
        int64_t n = nFuncs();

        auto M_block = eval_block_right(Ib, Jb);  // m x (n*n_j)

        // Primary: row ID => Iset[b+1], P = normal core[b]
        auto rr = interp_decomp_rows(M_block, param.reltol, param.bondDim);
        int64_t k = (int64_t)rr.rows.size();
        Iset[b+1] = Ib.at(rr.rows);

        // Secondary: col ID on (alpha stacked as rows) => Jset[b], alpha-free
        // M_block [m, n, n_j] -> permute [n, m, n_j] -> reshape (n*m) x n_j
        auto M_col = M_block.reshape({m, n, n_j})
                            .permute({1, 0, 2})
                            .reshape({n * m, n_j})
                            .contiguous();
        Jset[b] = Jb.at(interp_decomp_cols(M_col, 0.0, k).cols);

        // core[b] = rr.P reshaped [|Iset[b]|, d_b, k]
        tt.cores[b] = rr.P;
        reshape_site_tensor(b);

        // core[b+1] = M_block[rows, :] reshaped [k, n*d_{b+1}, |Jset[b+1]|]  (fat)
        int64_t d_b1 = (int64_t)localSet[b+1].size();
        int64_t r_b2 = (int64_t)Jset[b+1].size();
        tt.cores[b+1] = M_block.index_select(0, torch::tensor(rr.rows, torch::kLong))
                               .reshape({k, n * d_b1, r_b2})
                               .contiguous();
        collectPivotError(rr.sv);
    }

    // Left sweep: alpha on left side of center bond.
    // col_id on M_block[(alpha,i), j] => Jset[b] is alpha-free.
    // core[b] = fat (carries alpha),  core[b+1] = P (normal).
    void updateLeft(int b) {
        IndexSet<MultiIndex> Ib = kron(Iset[b],       localSet[b]);
        IndexSet<MultiIndex> Jb = kron(localSet[b+1], Jset[b+1]);
        int64_t m = (int64_t)Ib.size(), n_j = (int64_t)Jb.size();
        int64_t n = nFuncs();

        auto M_block = eval_block_left(Ib, Jb);  // (n*m) x n_j

        // Primary: col ID => Jset[b], P = normal core[b+1]
        auto cr = interp_decomp_cols(M_block, param.reltol, param.bondDim);
        int64_t k = (int64_t)cr.cols.size();
        Jset[b] = Jb.at(cr.cols);

        // Secondary: row ID on (alpha stacked as cols) => Iset[b+1], alpha-free
        // M_block [n, m, n_j] -> permute [m, n, n_j] -> reshape m x (n*n_j)
        auto M_row = M_block.reshape({n, m, n_j})
                            .permute({1, 0, 2})
                            .reshape({m, n * n_j})
                            .contiguous();
        Iset[b+1] = Ib.at(interp_decomp_rows(M_row, 0.0, k).rows);

        // core[b+1] = cr.P reshaped [k, d_{b+1}, |Jset[b+1]|]
        tt.cores[b+1] = cr.P;
        reshape_site_tensor(b+1);

        // core[b] = M_block[:,cols] reshaped [|Iset[b]|, n*d_b, k]  (fat)
        // (n*m) x k -> reshape [n, r_b, d_b, k] -> permute [r_b, n, d_b, k] -> [r_b, n*d_b, k]
        int64_t r_b = (int64_t)Iset[b].size();
        int64_t d_b = (int64_t)localSet[b].size();
        tt.cores[b] = M_block.index_select(1, torch::tensor(cr.cols, torch::kLong))
                             .reshape({n, r_b, d_b, k})
                             .permute({1, 0, 2, 3})
                             .reshape({r_b, n * d_b, k})
                             .contiguous();
        collectPivotError(cr.sv);
    }

    void reshape_site_tensor(int i) {
        int64_t a = Iset[i].size(), b = localSet[i].size(), c = Jset[i].size();
        tt.cores[i] = tt.cores[i].reshape({a, b, c});
    }

    void collectPivotError(const std::vector<double>& pe) {
        if (pe.size() > pivotError.size()) pivotError.resize(pe.size(), 0);
        for (size_t i = 0; i < pe.size(); i++)
            if (pe[i] > pivotError[i]) pivotError[i] = pe[i];
    }
};

} // namespace ttid
