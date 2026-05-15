#include <catch2/catch_test_macros.hpp>
#include "../tt.h"

static double rel_err(const torch::Tensor& A, const torch::Tensor& B) {
    return (A - B).norm().item().toDouble() / A.norm().item().toDouble();
}

TEST_CASE("TT cores have correct shapes", "[tt]") {
    torch::manual_seed(0);
    auto T = torch::randn({4, 5, 6, 3}, torch::kFloat64);
    TensorTrain tt(T);

    REQUIRE(tt.cores.size() == 4);
    REQUIRE(tt.cores.front().size(0) == 1);  // r_0 = 1
    REQUIRE(tt.cores.back().size(2) == 1);   // r_n = 1
    for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
        REQUIRE(tt.cores[k].size(2) == tt.cores[k + 1].size(0));  // bond dims match
}

TEST_CASE("TT exact reconstruction", "[tt]") {
    torch::manual_seed(1);
    for (auto shape : std::vector<std::vector<int64_t>>{{5, 6}, {4, 5, 6}, {3, 4, 5, 6}}) {
        auto T = torch::randn(shape, torch::kFloat64);
        REQUIRE(rel_err(T, TensorTrain(T).to_tensor()) < 1e-10);
    }
}

TEST_CASE("TT tolerance-based compression of a rank-2 tensor", "[tt]") {
    torch::manual_seed(2);
    // T = u1⊗v1⊗w1  +  eps * u2⊗v2⊗w2
    // The second term is O(eps) smaller, so a tight tolerance should drop it.
    auto u1 = torch::randn({8}, torch::kFloat64);
    auto v1 = torch::randn({7}, torch::kFloat64);
    auto w1 = torch::randn({6}, torch::kFloat64);
    auto u2 = torch::randn({8}, torch::kFloat64);
    auto v2 = torch::randn({7}, torch::kFloat64);
    auto w2 = torch::randn({6}, torch::kFloat64);

    auto outer = [](auto u, auto v, auto w) {
        return u.unsqueeze(1).unsqueeze(2) *
               v.unsqueeze(0).unsqueeze(2) *
               w.unsqueeze(0).unsqueeze(0);
    };

    const double eps = 1e-6;
    auto T = outer(u1, v1, w1) + eps * outer(u2, v2, w2);

    SECTION("tight tol keeps both terms: bond dim 2, near-exact reconstruction") {
        TensorTrain tt(T, /*tol=*/1e-8);
        // All inner bond dims should be 2 (rank-2 TT)
        for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
            REQUIRE(tt.cores[k].size(2) == 2);
        REQUIRE(rel_err(T, tt.to_tensor()) < 1e-8);
    }

    SECTION("loose tol drops the small term: bond dim 1, error ~ eps") {
        TensorTrain tt(T, /*tol=*/1e-4);
        for (size_t k = 0; k + 1 < tt.cores.size(); ++k)
            REQUIRE(tt.cores[k].size(2) == 1);
        // Reconstruction misses the eps-sized component
        REQUIRE(rel_err(T, tt.to_tensor()) < 10 * eps);
    }
}
