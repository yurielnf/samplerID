#include <catch2/catch_test_macros.hpp>
#include "sampler_id/tt.h"

static double rel_err(const torch::Tensor& A, const torch::Tensor& B) {
    return (A - B).norm().item().toDouble() / A.norm().item().toDouble();
}

TEST_CASE("TT cores have correct shapes", "[tt]") {
    torch::manual_seed(0);
    auto T = torch::randn({4, 5, 6, 3}, torch::kFloat64);
    TensorTrain tt(T);

    REQUIRE(tt.cores.size() == 4);
    REQUIRE(tt.cores.front().size(0) == 1);
    REQUIRE(tt.cores.back().size(2) == 1);
    for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
        REQUIRE(tt.cores[k].size(2) == tt.cores[k + 1].size(0));
}

TEST_CASE("TT exact reconstruction", "[tt]") {
    torch::manual_seed(1);
    for (auto shape : std::vector<std::vector<int64_t>>{{5, 6}, {4, 5, 6}, {3, 4, 5, 6}}) {
        auto T = torch::randn(shape, torch::kFloat64);
        REQUIRE(rel_err(T, TensorTrain(T).to_tensor()) < 1e-10);
    }
}

TEST_CASE("TT tolerance-based compression", "[tt]") {
    torch::manual_seed(2);
    auto outer = [](auto u, auto v, auto w) {
        return u.unsqueeze(1).unsqueeze(2) *
               v.unsqueeze(0).unsqueeze(2) *
               w.unsqueeze(0).unsqueeze(0);
    };
    auto u1 = torch::randn({8}, torch::kFloat64), v1 = torch::randn({7}, torch::kFloat64),
         w1 = torch::randn({6}, torch::kFloat64);
    auto u2 = torch::randn({8}, torch::kFloat64), v2 = torch::randn({7}, torch::kFloat64),
         w2 = torch::randn({6}, torch::kFloat64);
    const double eps = 1e-6;
    auto T = outer(u1, v1, w1) + eps * outer(u2, v2, w2);

    SECTION("tight tol → bond dim 2") {
        TensorTrain tt(T, 1e-8);
        for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
            REQUIRE(tt.cores[k].size(2) == 2);
        REQUIRE(rel_err(T, tt.to_tensor()) < 1e-8);
    }
    SECTION("loose tol → bond dim 1") {
        TensorTrain tt(T, 1e-4);
        for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
            REQUIRE(tt.cores[k].size(2) == 1);
        REQUIRE(rel_err(T, tt.to_tensor()) < 10 * eps);
    }
}

TEST_CASE("TT sum", "[tt]") {
    torch::manual_seed(3);
    // sum of a rank-1 TT = product of sums of each mode
    auto u = torch::randn({5}, torch::kFloat64);
    auto v = torch::randn({4}, torch::kFloat64);
    auto T = u.unsqueeze(1) * v.unsqueeze(0);  // outer product, shape [5,4]
    TensorTrain tt(T);
    double expected = u.sum().item().toDouble() * v.sum().item().toDouble();
    REQUIRE(std::abs(tt.sum() - expected) < 1e-10);
}

TEST_CASE("TT norm and overlap", "[tt]") {
    torch::manual_seed(4);
    auto T = torch::randn({4, 5, 6}, torch::kFloat64);
    TensorTrain tt(T);
    double tt_norm = tt.norm();
    double ref_norm = T.norm().item().toDouble();
    REQUIRE(std::abs(tt_norm - ref_norm) / ref_norm < 1e-10);

    // overlap with itself = norm^2
    double ov = tt.overlap(tt);
    REQUIRE(std::abs(ov - ref_norm * ref_norm) / (ref_norm * ref_norm) < 1e-10);
}
