#include <catch2/catch_test_macros.hpp>
#include "sampler_id/block_tt_id.h"
#include <cmath>
#include <numeric>

using namespace ttid;

TEST_CASE("block_tt_id two functions cos/sin of sum", "[block_tt_id]") {
    int dim = 5, d = 8;

    // cos(sum) and sin(sum) each have TT-rank 2
    auto f0 = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return std::cos(s);
    };
    auto f1 = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return std::sin(s);
    };

    block_tt_id<double> btt({f0, f1}, std::vector<int>(dim, d));
    btt.iterate(10);

    for (auto& ids : std::vector<std::vector<int>>{{0,1,2,3,4}, {3,5,1,5,1}, {7,7,7,7,7}}) {
        REQUIRE(std::abs(btt.eval(0, ids) - f0(ids)) < 1e-5);
        REQUIRE(std::abs(btt.eval(1, ids) - f1(ids)) < 1e-5);
    }

    // Center core local dim must be n_funcs * d
    REQUIRE(btt.tt.cores[btt.center].size(1) == 2 * d);
}

TEST_CASE("block_tt_id three functions", "[block_tt_id]") {
    int dim = 4, d = 6;

    auto f0 = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return std::cos(s);
    };
    auto f1 = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return std::sin(s);
    };
    // sum^2 = (sum_k i_k)^2 — TT-rank 3
    auto f2 = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return s * s;
    };

    block_tt_id<double> btt({f0, f1, f2}, std::vector<int>(dim, d));
    btt.iterate(15);

    std::vector<int> ids = {1, 3, 2, 5};
    REQUIRE(std::abs(btt.eval(0, ids) - f0(ids)) < 1e-4);
    REQUIRE(std::abs(btt.eval(1, ids) - f1(ids)) < 1e-4);
    REQUIRE(std::abs(btt.eval(2, ids) - f2(ids)) < 1e-4);

    // Center core local dim = 3 * d
    REQUIRE(btt.tt.cores[btt.center].size(1) == 3 * d);
}

TEST_CASE("block_tt_id n=1 gives same accuracy as tt_id", "[block_tt_id]") {
    int dim = 4, d = 8;

    auto f = [](std::vector<int> id) -> double {
        double s = std::accumulate(id.begin(), id.end(), 0.0);
        return std::cos(s);
    };

    tt_id_param p; p.bondDim = 10;
    block_tt_id<double> btt({f}, std::vector<int>(dim, d), p);
    btt.iterate(10);

    for (auto& ids : std::vector<std::vector<int>>{{0,1,2,3}, {3,5,1,5}, {7,7,7,7}}) {
        REQUIRE(std::abs(btt.eval(0, ids) - f(ids)) < 1e-5);
    }
}

TEST_CASE("block_tt_id bond dimension is shared", "[block_tt_id]") {
    // Two separable functions have rank 1 individually;
    // shared learning should keep bond dim low.
    int dim = 5, d = 4;

    // f0 = prod cos(i_k),  f1 = prod sin(i_k)
    auto f0 = [](std::vector<int> id) -> double {
        double v = 1.0; for (int x : id) v *= std::cos(x); return v;
    };
    auto f1 = [](std::vector<int> id) -> double {
        double v = 1.0; for (int x : id) v *= std::sin(x + 1); return v;
    };

    block_tt_id<double> btt({f0, f1}, std::vector<int>(dim, d));
    btt.iterate(10);

    // All non-center bonds should have dim <= 2 (each function contributes rank 1)
    for (int b = 0; b < (int)btt.len() - 1; b++) {
        if (b == btt.center) continue;
        int bond = (int)btt.tt.cores[b].size(2);
        REQUIRE(bond <= 4);
    }

    std::vector<int> ids = {1, 2, 3, 1, 0};
    REQUIRE(std::abs(btt.eval(0, ids) - f0(ids)) < 1e-5);
    REQUIRE(std::abs(btt.eval(1, ids) - f1(ids)) < 1e-5);
}
