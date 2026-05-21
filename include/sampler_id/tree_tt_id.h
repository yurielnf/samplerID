#pragma once
#include "tensor_tree.h"
#include "interp_decomp.h"
#include "index_set.h"
#include <functional>
#include <map>
#include <utility>
#include <vector>
#include <stdexcept>

namespace ttid {

using xfac::MultiIndex;

struct tree_tt_id_param {
    int bondDim = 30;
    double reltol = 1e-12;
    std::vector<int> pivot1;          // size = tree.size(); virtual-node entries ignored
    bool use_cache = false;
};

// Interpolative-decomposition tensor-tree construction of f:(i_0,...,i_{N-1})->T.
// localDim[u] is required for every physical node u.
template<typename T = double>
struct tree_tt_id {
    using tensor_fun = std::function<T(std::vector<int>)>;
    static constexpr auto dtype_v = c10::CppTypeToScalarType<T>::value;

    tensor_fun f;
    TreeTopology tree;
    tree_tt_id_param param;
    std::vector<double> pivotError;
    std::map<int, xfac::IndexSet<MultiIndex>> localSet;                       // per physical node
    std::map<std::pair<int,int>, xfac::IndexSet<MultiIndex>> Iset;            // per directed edge: u-side pivots
    TensorTree<T> tt;
    int cIter = 0;

    tree_tt_id() = default;

    tree_tt_id(tensor_fun f_, TreeTopology tree_,
               std::map<int,int> const& localDim, tree_tt_id_param param_ = {})
        : f(std::move(f_)), tree(std::move(tree_)), param(std::move(param_)),
          pivotError(1), tt(tree)
    {
        tree.validate();
        for (int p : tree.phys)
            if (!localDim.count(p))
                throw std::invalid_argument("tree_tt_id: localDim missing for a physical node");

        int N = (int)tree.size();
        if (param.pivot1.empty()) param.pivot1.assign(N, 0);
        if ((int)param.pivot1.size() != N)
            throw std::invalid_argument("tree_tt_id: pivot1.size() != tree.size()");

        pivotError[0] = std::abs(f(param.pivot1));
        if (pivotError[0] == 0.0)
            throw std::invalid_argument("tree_tt_id: f(pivot1)=0, provide a better first pivot");

        // Build localSet: one rank-1 MultiIndex per local value, zero on all other coordinates.
        for (int p : tree.phys) {
            int d = localDim.at(p);
            for (int i = 0; i < d; ++i) {
                MultiIndex mi(N, char32_t(0));
                mi[p] = char32_t(i);
                localSet[p].push_back(mi);
            }
        }

        // Initialize Iset[{u,v}] with the partition of pivot1 restricted to u's physical side.
        for (auto const& [u, ns] : tree.neigh) {
            for (int v : ns.from_int()) {
                auto [u_phys, _] = tree.split(u, v);
                MultiIndex mi(N, char32_t(0));
                for (int p : u_phys) mi[p] = char32_t(param.pivot1[p]);
                Iset[{u, v}].push_back(mi);
            }
        }

        rebuild_all_cores();
    }

    // Push every pivot into every directed edge's Iset (as the u-side partial),
    // then iterate to let row-ID select the most informative pivots.
    void addPivotsAllBonds(std::vector<std::vector<int>> const& pivots) {
        int N = (int)tree.size();
        for (auto const& pv : pivots) {
            if ((int)pv.size() != N)
                throw std::invalid_argument("addPivotsAllBonds: pivot size != tree.size()");
            for (auto const& [u, ns] : tree.neigh) {
                for (int v : ns.from_int()) {
                    auto [u_phys, _] = tree.split(u, v);
                    MultiIndex mi((std::size_t)N, char32_t(0));
                    for (int p : u_phys) mi[p] = char32_t(pv[p]);
                    Iset[{u, v}].push_back(mi);
                }
            }
        }
        iterate(2);
    }

    // One iterate = one leaves->root + one root->leaves half-sweep.
    void iterate(int nIter = 1) {
        for (int it = 0; it < nIter; ++it) {
            for (auto [u, v] : tree.leaves_to_root()) updateEdge(u, v);   // u = child of v
            for (auto [u, v] : tree.root_to_leaves()) updateEdge(v, u);   // v = child of u
            cIter++;
        }
        rebuild_all_cores();
    }

protected:
    // updateEdge(child, parent): row-ID with rows on child's side, col-ID with cols on parent's side.
    void updateEdge(int child, int parent) {
        auto Ic = build_side_indices(child, parent);
        auto Ip = build_side_indices(parent, child);
        auto M = eval_matrix(Ic, Ip);

        auto res_r = interp_decomp_rows(M, param.reltol, param.bondDim);
        int64_t k = (int64_t)res_r.rows.size();
        auto res_c = interp_decomp_cols(M, 0.0, k);

        xfac::IndexSet<MultiIndex> new_Iset_cp, new_Iset_pc;
        for (int r : res_r.rows) new_Iset_cp.push_back(Ic[r]);
        for (int c : res_c.cols) new_Iset_pc.push_back(Ip[c]);
        Iset[{child, parent}] = std::move(new_Iset_cp);
        Iset[{parent, child}] = std::move(new_Iset_pc);

        collectPivotError(res_r.sv);
    }

    // Rebuild every core from the current Isets.
    // Each non-root node u stores the row-interp matrix at edge (u, parent(u)).
    // The root stores the raw f-evaluation at the kron of its neighbor pivots.
    void rebuild_all_cores() {
        // parent map from a root traversal.
        std::map<int,int> parent;
        parent[tree.root] = -1;
        for (auto [u, v] : tree.root_to_leaves()) parent[v] = u;

        for (auto const& [u, _] : tree.neigh) {
            if (u == tree.root) {
                tt.M[u] = build_anchor_core(u);
            } else {
                tt.M[u] = build_interp_core(u, parent[u]);
            }
        }
    }

    // Side indices for node u with neighbor `excl` removed: kron of Iset[{c,u}] for
    // c in neigh(u)\{excl}, in neigh[u] order (first slowest), then localSet[u] if phys.
    std::vector<MultiIndex> build_side_indices(int u, int excl) const {
        std::vector<std::vector<MultiIndex>> stacks;
        for (int c : tree.neigh.at(u).from_int()) {
            if (c == excl) continue;
            stacks.push_back(Iset.at({c, u}).from_int());
        }
        if (tree.is_phys(u)) stacks.push_back(localSet.at(u).from_int());
        return kron_ordered(stacks, (int)tree.size());
    }

    // Kron of multi-index sets: sets[0] varies slowest, sets.back() varies fastest.
    // Matches a torch::reshape with shape [|sets[0]|, ..., |sets.back()|] (row-major).
    static std::vector<MultiIndex> kron_ordered(
        std::vector<std::vector<MultiIndex>> const& sets, int N)
    {
        std::vector<MultiIndex> R{ MultiIndex((std::size_t)N, char32_t(0)) };
        for (auto const& S : sets) {
            std::vector<MultiIndex> next;
            next.reserve(R.size() * S.size());
            for (auto const& a : R)
                for (auto const& b : S)
                    next.push_back(xfac::add(a, b));
            R = std::move(next);
        }
        return R;
    }

    torch::Tensor eval_matrix(std::vector<MultiIndex> const& I,
                              std::vector<MultiIndex> const& J) const
    {
        auto M = torch::empty({(int64_t)I.size(), (int64_t)J.size()},
                              torch::TensorOptions().dtype(dtype_v));
        auto ptr = M.template data_ptr<T>();
        for (size_t i = 0; i < I.size(); ++i)
            for (size_t j = 0; j < J.size(); ++j) {
                MultiIndex ij = xfac::add(I[i], J[j]);
                std::vector<int> key(ij.begin(), ij.end());
                ptr[i * J.size() + j] = cached_eval(key);
            }
        return M;
    }

    // The row-interpolation core for non-root u (parent = its parent under tree.root).
    // Output axes match neigh[u] order, with a trailing physical axis if u is phys.
    torch::Tensor build_interp_core(int u, int parent) {
        int deg = (int)tree.neigh.at(u).size();
        int pp = tree.leg_pos(u, parent);
        bool phys = tree.is_phys(u);

        // Π-matrix at the (u,parent) edge: same logic as updateEdge, freshly computed.
        auto Ic = build_side_indices(u, parent);
        auto Ip = build_side_indices(parent, u);
        auto M = eval_matrix(Ic, Ip);

        int64_t k = (int64_t)Iset.at({u, parent}).size();
        auto res = interp_decomp_rows(M, 0.0, k);   // exactly k rows so the dim matches Iset
        auto P = res.P;                              // shape (|Ic|, k)

        // Reshape P: source axes are [neigh[u][i] for i!=pp in order, (phys?), k].
        std::vector<int64_t> src_shape;
        for (int i = 0; i < deg; ++i) {
            if (i == pp) continue;
            int c = tree.neigh.at(u).from_int()[i];
            src_shape.push_back((int64_t)Iset.at({c, u}).size());
        }
        if (phys) src_shape.push_back((int64_t)localSet.at(u).size());
        src_shape.push_back(k);
        auto Tn = P.reshape(src_shape);

        // Target axes: [neigh[u][0], ..., neigh[u][deg-1], (phys?)].
        // Build a permutation from target axis -> source axis.
        int total = (int)src_shape.size();
        std::vector<int64_t> perm(total);
        int src_idx = 0;
        for (int i = 0; i < deg; ++i) {
            if (i == pp) perm[i] = total - 1;       // parent-axis source slot is last
            else         perm[i] = src_idx++;
        }
        if (phys) perm[deg] = src_idx;               // phys source slot is just before k
        return Tn.permute(perm).contiguous();
    }

    // Raw f-eval at the kron of all neighbor pivots (and localSet if phys).
    // Used only at the root, where there is no parent direction to interpolate along.
    torch::Tensor build_anchor_core(int u) {
        bool phys = tree.is_phys(u);
        std::vector<std::vector<MultiIndex>> stacks;
        std::vector<int64_t> shape;
        for (int c : tree.neigh.at(u).from_int()) {
            stacks.push_back(Iset.at({c, u}).from_int());
            shape.push_back((int64_t)Iset.at({c, u}).size());
        }
        if (phys) {
            stacks.push_back(localSet.at(u).from_int());
            shape.push_back((int64_t)localSet.at(u).size());
        }
        auto I = kron_ordered(stacks, (int)tree.size());
        std::vector<MultiIndex> J = { MultiIndex((std::size_t)tree.size(), char32_t(0)) };
        auto Mmat = eval_matrix(I, J);              // shape [|I|, 1]
        return Mmat.reshape(shape);
    }

    void collectPivotError(std::vector<double> const& pe) {
        if (pe.size() > pivotError.size()) pivotError.resize(pe.size(), 0);
        for (size_t i = 0; i < pe.size(); ++i)
            if (pe[i] > pivotError[i]) pivotError[i] = pe[i];
    }

    T cached_eval(std::vector<int> const& key) const {
        if (!param.use_cache) return f(key);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        return cache[key] = f(key);
    }

private:
    mutable std::map<std::vector<int>, T> cache;
};

} // namespace ttid
