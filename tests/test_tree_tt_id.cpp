#include <catch2/catch_test_macros.hpp>
#include "sampler_id/tree_tt_id.h"
#include <cmath>
#include <numeric>
#include <random>

using namespace ttid;

TEST_CASE("TreeTopology basics", "[tree]") {
    auto t = TreeTopology::chain(5);
    REQUIRE(t.size() == 5);
    REQUIRE(t.n_phys() == 5);
    t.validate();

    auto rtl = t.root_to_leaves();
    REQUIRE(rtl.size() == 4);
    REQUIRE(rtl.front() == std::make_pair(0, 1));
    REQUIRE(rtl.back()  == std::make_pair(3, 4));

    auto ltr = t.leaves_to_root();
    REQUIRE(ltr.front() == std::make_pair(4, 3));
    REQUIRE(ltr.back()  == std::make_pair(1, 0));

    auto [a, b] = t.split(2, 3);
    REQUIRE(a == std::set<int>{0, 1, 2});
    REQUIRE(b == std::set<int>{3, 4});
}

TEST_CASE("tree_tt_id chain reproduces sum+cos", "[tree_tt_id]") {
    int dim = 5, d = 10;
    auto myf = [&](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return s + std::cos(s);
    };
    auto tree = TreeTopology::chain(dim);
    std::map<int,int> ld;
    for (int p : tree.phys) ld[p] = d;

    tree_tt_id_param p; p.use_cache = true;
    auto ci = tree_tt_id<double>(myf, tree, ld, p);
    ci.iterate(8);

    std::vector<int> ids = {3, 5, 1, 5, 1};
    REQUIRE(std::abs(ci.tt.eval(ids) - myf(ids)) < 1e-5);

    // The middle bond dimension of the rank-2 cos+sum tensor should be small (theory: 4).
    int mid_bond = (int)ci.tt.M.at(2).size(0);
    REQUIRE(mid_bond <= 6);
}

TEST_CASE("tree_tt_id Y-tree on separable function", "[tree_tt_id]") {
    // Y-tree: virtual center 3 connected to physical leaves 0, 1, 2.
    TreeTopology t(3);
    t.neigh[0]; t.neigh[1]; t.neigh[2]; t.neigh[3];
    t.add_edge(3, 0);
    t.add_edge(3, 1);
    t.add_edge(3, 2);
    t.add_phys(0); t.add_phys(1); t.add_phys(2);
    t.validate();

    int d = 4;
    auto myf = [&](std::vector<int> id) -> double {
        // separable: cos(i0) * cos(i1) * cos(i2); virtual node id[3] is ignored.
        return std::cos((double)id[0]) * std::cos((double)id[1]) * std::cos((double)id[2]);
    };
    std::map<int,int> ld = {{0, d}, {1, d}, {2, d}};
    tree_tt_id_param p; p.pivot1 = {0, 0, 0, 0};
    auto ci = tree_tt_id<double>(myf, t, ld, p);
    ci.iterate(5);

    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b)
            for (int c = 0; c < d; ++c) {
                std::vector<int> id = {a, b, c, 0};
                REQUIRE(std::abs(ci.tt.eval(id) - myf(id)) < 1e-10);
            }
}

TEST_CASE("tree_tt_id global pivots on chain", "[tree_tt_id]") {
    int dim = 6, d = 10, nPivots = 200;
    long count = 0;
    auto myf = [&](std::vector<int> id) -> double {
        count++;
        double s = 0; for (int x : id) s += x;
        return std::cos(s);
    };
    auto tree = TreeTopology::chain(dim);
    std::map<int,int> ld;
    for (int p : tree.phys) ld[p] = d;

    tree_tt_id_param p; p.use_cache = true;
    auto ci = tree_tt_id<double>(myf, tree, ld, p);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, d - 1);
    std::vector<std::vector<int>> pivots(nPivots, std::vector<int>(dim));
    for (auto& pv : pivots)
        for (auto& x : pv) x = dist(rng);

    count = 0;
    ci.addPivotsAllBonds(pivots);

    // Far cheaper than the full tensor of d^dim = 10^6 entries.
    REQUIRE(count < (long)std::pow(d, dim) / 2);

    for (const auto& pv : pivots)
        REQUIRE(std::abs(ci.tt.eval(pv) - myf(pv)) < 1e-10);
}

TEST_CASE("TensorTree sum and overlap on chain", "[tensor_tree]") {
    int dim = 4, d = 6;
    auto myf = [&](std::vector<int> id) -> double {
        double s = 0; for (int x : id) s += x;
        return std::cos(s);
    };
    auto tree = TreeTopology::chain(dim);
    std::map<int,int> ld; for (int p : tree.phys) ld[p] = d;

    auto ci = tree_tt_id<double>(myf, tree, ld);
    ci.iterate(6);

    // Exact sum by brute force.
    double exact = 0;
    std::vector<int> id(dim, 0);
    auto recurse = [&](auto&& self, int k) -> void {
        if (k == dim) { exact += myf(id); return; }
        for (int v = 0; v < d; ++v) { id[k] = v; self(self, k + 1); }
    };
    recurse(recurse, 0);
    REQUIRE(std::abs(ci.tt.sum() - exact) < 1e-6);

    // Self-overlap == sum of squares.
    double sq = 0;
    auto rec2 = [&](auto&& self, int k) -> void {
        if (k == dim) { double v = myf(id); sq += v * v; return; }
        for (int v = 0; v < d; ++v) { id[k] = v; self(self, k + 1); }
    };
    rec2(rec2, 0);
    REQUIRE(std::abs(ci.tt.overlap(ci.tt) - sq) < 1e-6);
    REQUIRE(std::abs(ci.tt.norm() - std::sqrt(sq)) < 1e-6);
}

TEST_CASE("TensorTree overlap between two trees on Y-tree", "[tensor_tree]") {
    TreeTopology t(3);
    t.neigh[0]; t.neigh[1]; t.neigh[2]; t.neigh[3];
    t.add_edge(3, 0); t.add_edge(3, 1); t.add_edge(3, 2);
    t.add_phys(0); t.add_phys(1); t.add_phys(2);

    int d = 5;
    auto f = [&](std::vector<int> id) -> double {
        return std::cos((double)id[0] + 0.3 * id[1] - 0.5 * id[2]);
    };
    auto g = [&](std::vector<int> id) -> double {
        return std::exp(-0.1 * (id[0] + id[1] + id[2]));
    };
    std::map<int,int> ld = {{0,d},{1,d},{2,d}};
    tree_tt_id_param pp; pp.pivot1 = {0,0,0,0};

    auto ttF = tree_tt_id<double>(f, t, ld, pp); ttF.iterate(6);
    auto ttG = tree_tt_id<double>(g, t, ld, pp); ttG.iterate(6);

    // Compare against the brute-force sum of the actual tt evaluations:
    // this isolates the overlap contraction from any TT representation error.
    double via_eval = 0;
    std::vector<int> id(4, 0);
    for (int a = 0; a < d; ++a)
        for (int b = 0; b < d; ++b)
            for (int c = 0; c < d; ++c) {
                id = {a, b, c, 0};
                via_eval += ttF.tt.eval(id) * ttG.tt.eval(id);
            }
    REQUIRE(std::abs(ttF.tt.overlap(ttG.tt) - via_eval) < 1e-10);
}

TEST_CASE("tree_tt_id global pivots on Y-tree", "[tree_tt_id]") {
    // Y-tree with virtual center 4 and physical leaves 0,1,2,3.
    TreeTopology t(4);
    t.neigh[0]; t.neigh[1]; t.neigh[2]; t.neigh[3]; t.neigh[4];
    t.add_edge(4, 0); t.add_edge(4, 1); t.add_edge(4, 2); t.add_edge(4, 3);
    t.add_phys(0); t.add_phys(1); t.add_phys(2); t.add_phys(3);
    t.validate();

    int d = 8;
    auto myf = [&](std::vector<int> id) -> double {
        double s = id[0] + id[1] + id[2] + id[3];
        return std::cos(s);
    };
    std::map<int,int> ld = {{0,d},{1,d},{2,d},{3,d}};
    tree_tt_id_param pp; pp.use_cache = true; pp.pivot1 = {0,0,0,0,0};
    auto ci = tree_tt_id<double>(myf, t, ld, pp);

    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dist(0, d - 1);
    std::vector<std::vector<int>> pivots(80, std::vector<int>(5, 0));
    for (auto& pv : pivots) for (int k = 0; k < 4; ++k) pv[k] = dist(rng);

    ci.addPivotsAllBonds(pivots);
    for (const auto& pv : pivots)
        REQUIRE(std::abs(ci.tt.eval(pv) - myf(pv)) < 1e-10);
}
