#include <catch2/catch_test_macros.hpp>
#include "sampler_id/tt_id.h"
#include <numeric>
#include <cmath>

TEST_CASE("tt_id<double> convergence on sum+cos", "[tt_id]") {
    SECTION("cos and sum are rank 2") {
        int dim = 5, d = 10;
        long count = 0;
        auto myf = [&](std::vector<int> id) -> double {
            count++;
            double s = std::accumulate(id.begin(), id.end(), 0.0);
            return s + std::cos(s);
        };
        tt_id_param p; p.use_cache = true;
        auto ci = tt_id<double>(myf, std::vector<int>(dim, d), p);
        ci.iterate(20);

        int bond_dim = ci.tt.cores[dim / 2 - 1].size(0);
        REQUIRE(bond_dim == 4);

        std::vector<int> ids = {3, 5, 1, 5, 1};
        REQUIRE(std::abs(ci.tt.eval(ids) - myf(ids)) < 1e-5);
        REQUIRE(count < (long)std::pow(d * bond_dim, 2) * (dim - 1));
    }

    SECTION("rational function fast convergence") {
        int dim = 5, d = 15;
        auto myf = [&](std::vector<int> id) -> double {
            double s = 0;
            for (int i = 0; i < dim; ++i) s += id[i] / 14.0;
            return std::pow(2.0, dim) / (1.0 + 2.0 * s);
        };
        auto ci = tt_id<double>(myf, std::vector<int>(dim, d));
        ci.iterate(11);
        REQUIRE(ci.tt.cores[1].sizes()[2] == 11);
        REQUIRE(ci.pivotError.at(10) < 1e-9);
    }
}

TEST_CASE("tt_id<float> on linear function", "[tt_id]") {
    int dim = 3, d = 5;
    auto ci = tt_id<float>([](std::vector<int> id) -> float {
        float s = 0; for (int x : id) s += x; return s;
    }, std::vector<int>(dim, d));
    ci.iterate(5);

    std::vector<int> ids = {1, 2, 3};
    REQUIRE(std::abs(ci.tt.eval<float>(ids) - 6.0f) < 1e-4f);
}

TEST_CASE("tt_id<complex<double>> on separable function", "[tt_id]") {
    using T = c10::complex<double>;
    int dim = 3, d = 5;

    auto myf = [&](std::vector<int> id) -> T {
        T val = 1.0;
        for (int x : id) val *= T(std::cos(x), std::sin(x));
        return val;
    };
    tt_id_param p; p.reltol = 1e-10;
    auto ci = tt_id<T>(myf, std::vector<int>(dim, d), p);
    ci.iterate(10);

    std::vector<int> ids = {2, 3, 1};
    T expected = T(std::cos(2), std::sin(2)) * T(std::cos(3), std::sin(3)) * T(std::cos(1), std::sin(1));
    auto got = ci.tt.eval<T>(ids);
    REQUIRE(std::abs(got - expected) < 1e-8);
}
