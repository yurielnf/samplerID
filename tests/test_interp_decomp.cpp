#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "sampler_id/interp_decomp.h"
using namespace ttid;

static double rel_err(const torch::Tensor& A, const torch::Tensor& B) {
    return (A - B).norm().item().toDouble() / A.norm().item().toDouble();
}

static torch::Tensor reconstruct(const torch::Tensor& A, const ColID& r) {
    auto C = A.index({torch::indexing::Slice(), torch::tensor(r.cols, torch::kLong)});
    return torch::mm(C, r.P);
}

TEST_CASE("interp_decomp_cols exact rank-k", "[id]") {
    torch::manual_seed(0);
    const int64_t m = 30, n = 20, r = 4;

    SECTION("float64") {
        auto A = torch::mm(torch::randn({m, r}, torch::kFloat64),
                           torch::randn({n, r}, torch::kFloat64).t());
        auto res = interp_decomp_cols(A, 0.0, r);
        REQUIRE((int64_t)res.cols.size() == r);
        REQUIRE(res.P.sizes() == torch::IntArrayRef({r, n}));
        REQUIRE(rel_err(A, reconstruct(A, res)) < 1e-10);
    }

    SECTION("float32") {
        auto A = torch::mm(torch::randn({m, r}, torch::kFloat32),
                           torch::randn({n, r}, torch::kFloat32).t());
        REQUIRE(rel_err(A, reconstruct(A, interp_decomp_cols(A, 0.0, r))) < 1e-4f);
    }

    SECTION("complex128") {
        auto A = torch::mm(torch::randn({m, r}, torch::kComplexDouble),
                           torch::randn({n, r}, torch::kComplexDouble).t());
        REQUIRE(rel_err(A, reconstruct(A, interp_decomp_cols(A, 0.0, r))) < 1e-10);
    }
}

TEST_CASE("interp_decomp_cols selected columns are actual columns of A", "[id]") {
    torch::manual_seed(1);
    auto A   = torch::randn({15, 12}, torch::kFloat64);
    auto res = interp_decomp_cols(A, 0.0, INT64_C(5));

    for (int64_t i = 0; i < 5; ++i) {
        int64_t c = res.cols[i];
        REQUIRE(c >= 0); REQUIRE(c < 12);
        double off = (res.P.select(1, c) - torch::eye(5, torch::kFloat64).select(1, i))
                         .norm().item().toDouble();
        REQUIRE(off < 1e-12);
    }
}

TEST_CASE("interp_decomp_cols distinct indices", "[id]") {
    torch::manual_seed(2);
    auto res = interp_decomp_cols(torch::randn({25, 18}, torch::kFloat64), 0.0, INT64_C(6));
    auto s   = res.cols; std::sort(s.begin(), s.end());
    REQUIRE(std::adjacent_find(s.begin(), s.end()) == s.end());
}

TEST_CASE("interp_decomp_cols quality improves with rank", "[id]") {
    torch::manual_seed(3);
    auto A = torch::randn({40, 30}, torch::kFloat64);
    double prev = 1.0;
    for (int64_t k : {3, 6, 10, 15}) {
        double e = rel_err(A, reconstruct(A, interp_decomp_cols(A, 0.0, k)));
        REQUIRE(e < prev); prev = e;
    }
}

TEST_CASE("interp_decomp_cols full rank → near-zero error", "[id]") {
    torch::manual_seed(4);
    auto A = torch::randn({10, 8}, torch::kFloat64);
    REQUIRE(rel_err(A, reconstruct(A, interp_decomp_cols(A, 0.0, INT64_C(8)))) < 1e-10);
}

TEST_CASE("interp_decomp_cols input validation", "[id]") {
    auto good = torch::randn({5, 4});
    REQUIRE_THROWS(interp_decomp_cols(torch::randn({3, 3, 3}), 0.0, INT64_C(2)));
    REQUIRE_THROWS(interp_decomp_cols(good, 0.0, INT64_C(0)));
    REQUIRE_NOTHROW(interp_decomp_cols(good, 0.0, INT64_C(5)));  // k > max_k silently caps
}

TEST_CASE("interp_decomp_cols singular values", "[id]") {
    torch::manual_seed(5);
    const int64_t m = 25, n = 20, k = 5;
    auto [V_orth, _] = torch::linalg_qr(torch::randn({m, m}, torch::kFloat64));
    std::vector<double> sv_exp = {10.0, 5.0, 2.0, 1.0, 0.5};
    auto A = torch::zeros({m, n}, torch::kFloat64);
    for (int64_t i = 0; i < k; ++i)
        A.select(1, i).copy_(V_orth.select(1, i) * sv_exp[i]);
    A.slice(1, k, n) = torch::randn({m, n - k}, torch::kFloat64) * 0.1;

    SECTION("fixed rank") {
        auto res = interp_decomp_cols(A, 0.0, k);
        auto sv  = res.sv; std::sort(sv.rbegin(), sv.rend());
        for (int64_t i = 0; i < k; ++i)
            REQUIRE(sv[i] == Catch::Approx(sv_exp[i]).epsilon(0.1).margin(0.5));
    }
    SECTION("tolerance-based") {
        auto res = interp_decomp_cols(A, 0.25);
        REQUIRE((int64_t)res.cols.size() < k);
    }
}
