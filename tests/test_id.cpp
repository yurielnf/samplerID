#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../id.h"

// Helper: relative Frobenius error ||A - B|| / ||A||
static double rel_err(const torch::Tensor& A, const torch::Tensor& B) {
    return (A - B).norm().item().toDouble() / A.norm().item().toDouble();
}

// Reconstruct A from an IDResult
static torch::Tensor reconstruct(const torch::Tensor& A, const IDResult& r) {
    auto C = A.index({torch::indexing::Slice(),
                      torch::tensor(r.cols, torch::kLong)});
    return torch::mm(C, r.P);
}

TEST_CASE("ID reconstructs an exact rank-k matrix", "[id]") {
    torch::manual_seed(0);
    const int64_t m = 30, n = 20, r = 4;

    SECTION("float64") {
        auto U = torch::randn({m, r}, torch::kFloat64);
        auto V = torch::randn({n, r}, torch::kFloat64);
        auto A = torch::mm(U, V.t());

        auto result = id_cols(A, r);

        REQUIRE(static_cast<int64_t>(result.cols.size()) == r);
        REQUIRE(result.P.sizes() == torch::IntArrayRef({r, n}));
        REQUIRE(rel_err(A, reconstruct(A, result)) < 1e-10);
    }

    SECTION("float32") {
        auto U = torch::randn({m, r}, torch::kFloat32);
        auto V = torch::randn({n, r}, torch::kFloat32);
        auto A = torch::mm(U, V.t());

        auto result = id_cols(A, r);

        REQUIRE(rel_err(A, reconstruct(A, result)) < 1e-4f);
    }
}

TEST_CASE("ID selected columns are actual columns of A", "[id]") {
    torch::manual_seed(1);
    auto A = torch::randn({15, 12}, torch::kFloat64);
    auto result = id_cols(A, INT64_C(5));

    for (int64_t i = 0; i < 5; ++i) {
        int64_t c = result.cols[i];
        REQUIRE(c >= 0);
        REQUIRE(c < 12);
        // Column c of P should be a standard basis vector e_i
        double off = (result.P.select(1, c) - torch::eye(5, torch::kFloat64).select(1, i))
                         .norm().item().toDouble();
        REQUIRE(off < 1e-12);
    }
}

TEST_CASE("ID selected column indices are distinct", "[id]") {
    torch::manual_seed(2);
    auto A = torch::randn({25, 18}, torch::kFloat64);
    auto result = id_cols(A, INT64_C(6));

    auto sorted = result.cols;
    std::sort(sorted.begin(), sorted.end());
    auto it = std::adjacent_find(sorted.begin(), sorted.end());
    REQUIRE(it == sorted.end());
}

TEST_CASE("ID quality improves with rank", "[id]") {
    torch::manual_seed(3);
    const int64_t m = 40, n = 30;
    auto A = torch::randn({m, n}, torch::kFloat64);

    double prev_err = 1.0;
    for (int64_t k : {3, 6, 10, 15}) {
        double err = rel_err(A, reconstruct(A, id_cols(A, k)));
        REQUIRE(err < prev_err);
        prev_err = err;
    }
}

TEST_CASE("ID with k == min(m,n) gives near-zero error", "[id]") {
    torch::manual_seed(4);
    const int64_t m = 10, n = 8;
    auto A = torch::randn({m, n}, torch::kFloat64);

    auto result = id_cols(A, n);  // full column rank
    REQUIRE(rel_err(A, reconstruct(A, result)) < 1e-10);
}

TEST_CASE("ID input validation", "[id]") {
    auto good = torch::randn({5, 4});
    REQUIRE_THROWS(id_cols(torch::randn({3, 3, 3}), INT64_C(2)));  // not 2D
    REQUIRE_THROWS(id_cols(good, INT64_C(0)));                     // k too small
    REQUIRE_THROWS(id_cols(good, INT64_C(5)));                     // k > min(m,n)
}

TEST_CASE("ID returns singular values", "[id]") {
    torch::manual_seed(5);
    const int64_t m = 25, n = 20, k = 5;

    // Create orthonormal basis
    auto V_full = torch::randn({m, m}, torch::kFloat64);
    auto [V_orth, _] = torch::linalg_qr(V_full);

    // Expected singular values (column norms)
    std::vector<double> sv_expected = {10.0, 5.0, 2.0, 1.0, 0.5};

    // Construct matrix with columns having explicit norms
    // A = [v1*sv[0], v2*sv[1], v3*sv[2], v4*sv[3], v5*sv[4], small_random_columns]
    auto A = torch::zeros({m, n}, torch::kFloat64);
    for (int64_t i = 0; i < k; ++i) {
        auto col = V_orth.select(1, i) * sv_expected[i];
        A.select(1, i).copy_(col);
    }
    // Fill remaining columns with small random noise
    auto remaining = torch::randn({m, n-k}, torch::kFloat64) * 0.1;
    A.slice(1, k, n) = remaining;

    SECTION("rank == k") {
        auto result = id_cols(A, k);
        int64_t actual_rank = (int64_t)result.cols.size();
        REQUIRE(actual_rank <= k);
        REQUIRE(result.sv.size() == (size_t)actual_rank);

        // Compare returned SV against expected (order may differ due to pivoting)
        std::vector<double> sv_sorted = result.sv;
        std::sort(sv_sorted.rbegin(), sv_sorted.rend());
        std::vector<double> sv_exp_sorted(sv_expected.begin(), sv_expected.begin() + actual_rank);
        std::sort(sv_exp_sorted.rbegin(), sv_exp_sorted.rend());

        for (int64_t i = 0; i < actual_rank; ++i) {
            REQUIRE(sv_sorted[i] == Catch::Approx(sv_exp_sorted[i]).epsilon(0.1).margin(0.5));
        }
    }

    SECTION("rank < k (tolerance-based)") {
        auto result = id_cols(A, 0.25);  // tolerance to select fewer columns
        int64_t actual_rank = (int64_t)result.cols.size();
        REQUIRE(actual_rank < k);
        REQUIRE(result.sv.size() == (size_t)actual_rank);

        // Verify returned SV are subset of expected (sorted)
        std::vector<double> sv_sorted = result.sv;
        std::sort(sv_sorted.rbegin(), sv_sorted.rend());
        for (int64_t i = 0; i < actual_rank; ++i) {
            REQUIRE(sv_sorted[i] == Catch::Approx(sv_expected[i]).epsilon(0.1).margin(0.5));
        }
    }
}
