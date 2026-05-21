#pragma once
#include "tree.h"
#include <torch/torch.h>
#include <map>
#include <vector>
#include <stdexcept>
#include <type_traits>

namespace ttid {

// Tensor tree data structure. M[u] has `deg(u)` legs in neigh[u] order, plus a
// trailing physical leg if u is physical. Bond u-v dimensions are consistent:
// M[u].size(leg_pos(u,v)) == M[v].size(leg_pos(v,u)).
//
// All contraction algorithms below are chain-like over tree.leaves_to_root():
// the tree recursion is owned by the iterator, the loop body sees only one
// edge at a time. We keep a length-1 axis at each contracted position so that
// neigh[u].pos(c) keeps indexing the same axis throughout the sweep.
template<typename T = double>
struct TensorTree {
    static constexpr auto dtype_v = c10::CppTypeToScalarType<T>::value;

    TreeTopology tree;
    std::map<int, torch::Tensor> M;

    TensorTree() = default;
    explicit TensorTree(TreeTopology tree_) : tree(std::move(tree_)) {
        for (auto const& [u, _] : tree.neigh) M[u] = torch::Tensor();
    }

    T eval(std::vector<int> const& id) const {
        if ((int)id.size() != (int)tree.size())
            throw std::invalid_argument("TensorTree::eval id size != tree size");
        std::map<int, torch::Tensor> prod;
        for (auto const& [u, m] : M)
            prod[u] = tree.is_phys(u) ? m.select(tree.deg(u), (int64_t)id[u]) : m;
        for (auto [from, to] : tree.leaves_to_root()) {
            int p = tree.leg_pos(to, from);
            absorb_vector(prod[to], prod[from].reshape({-1}), p);
        }
        return prod[tree.root].template item<T>();
    }
    T operator()(std::vector<int> const& id) const { return eval(id); }

    T sum() const {
        std::map<int, torch::Tensor> prod;
        for (auto const& [u, m] : M)
            prod[u] = tree.is_phys(u) ? m.sum(tree.deg(u)) : m;
        for (auto [from, to] : tree.leaves_to_root()) {
            int p = tree.leg_pos(to, from);
            absorb_vector(prod[to], prod[from].reshape({-1}), p);
        }
        return prod[tree.root].template item<T>();
    }

    // <this | other>, conjugating `other` (matches the chain TensorTrain).
    T overlap(TensorTree const& other) const {
        if (tree != other.tree)
            throw std::invalid_argument("TensorTree::overlap: incompatible trees");
        std::map<int, torch::Tensor> prod;
        for (auto const& [u, m] : M) {
            int deg = tree.deg(u);
            auto B = other.M.at(u).conj();
            prod[u] = tree.is_phys(u)
                ? torch::tensordot(m, B, {deg}, {deg})   // contract physical axis
                : torch::tensordot(m, B, {}, {});        // outer product
            // axes of prod[u]: [m's deg neighbor axes, B's deg neighbor axes]
        }
        for (auto [from, to] : tree.leaves_to_root()) {
            int df = tree.deg(from), dt = tree.deg(to);
            int pf = tree.leg_pos(from, to);
            int pt = tree.leg_pos(to, from);
            int64_t k_this  = prod[from].size(pf);
            int64_t k_other = prod[from].size(df + pf);
            // prod[from] now has size 1 on every axis except pf and df+pf, so
            // reshape gives the (this_bond, other_bond) matrix directly.
            auto mat = prod[from].reshape({k_this, k_other});
            absorb_matrix(prod[to], mat, pt, dt + pt);
        }
        return prod[tree.root].template item<T>();
    }

    double norm() const {
        T ov = overlap(*this);
        if constexpr (std::is_floating_point_v<T>)
            return std::sqrt(std::abs((double)ov));
        else
            return std::sqrt(std::abs((double)ov.real()));
    }

private:
    // In-place: T *= v broadcast on axis `axis`, then sum out `axis` keeping its slot.
    static void absorb_vector(torch::Tensor& T_, torch::Tensor const& v, int axis) {
        std::vector<int64_t> bshape(T_.dim(), 1);
        bshape[axis] = v.numel();
        T_ = (T_ * v.reshape(bshape)).sum(axis, /*keepdim=*/true);
    }

    // In-place: contract matrix `mat[a,b]` into T_ at axes (a_axis, b_axis), keepdim.
    static void absorb_matrix(torch::Tensor& T_, torch::Tensor const& mat,
                              int a_axis, int b_axis) {
        std::vector<int64_t> bshape(T_.dim(), 1);
        bshape[a_axis] = mat.size(0);
        bshape[b_axis] = mat.size(1);
        T_ = (T_ * mat.reshape(bshape)).sum(a_axis, true).sum(b_axis, true);
    }
};

} // namespace ttid
