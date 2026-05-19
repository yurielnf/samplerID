#include <catch2/catch_test_macros.hpp>
#include "sampler_id/q_tt_id.h"
#include <cmath>
#include <iostream>
#include <set>
using namespace ttid;
using namespace std;

TEST_CASE("scale separation", "[q_tt_id]")
{
    auto ft = [](double t) { return (10*exp(-t*t)+1)*cos(t); };
    auto ci = q_tt_id<double>(ft, grid::Quantics{-100, 100, 20}, {.bondDim=12});
    ci.addPivotsAllBonds({vector<int>(20, 1)});
    ci.iterate(5);

    // cout << "cIter=" << ci.cIter << ", rank=" << ci.pivotError.size()-1 << "\n";
    // for (auto i = 0u; i < ci.pivotError.size(); i++)
    //     cout << i << " " << ci.pivotError[i] << "\n";
    REQUIRE(ci.pivotError.size()==12);
    REQUIRE(ci.pivotError.at(11)<1e-10);
}

TEST_CASE("Hiroshi example", "[q_tt_id]")
{
    int nBit = 20;
    double abstol = 1e-4, delta = 10.0/(1<<nBit);
    cout << "delta=" << delta << "\n";
    for (auto t = 0; t < 100; t++) {
        set<double> rpoint {0};
        for (auto i = 0; i < 20; i++)
            rpoint.insert(1.0*(rand()%(1<<nBit))/(1<<nBit));
        auto ft = [&](double x) {
            double y = exp(-x);
            for (auto r : rpoint)
                if (abs(r-x) < delta) y += 2*abstol;
            return y;
        };
        auto ci = q_tt_id<double>(ft, grid::Quantics{0, 1, nBit}, {.bondDim=200});
        ci.iterate(2);
        ci.addPivotValues({rpoint.begin(), rpoint.end()});
        ci.iterate(4);
        auto qtt = ci.get_qtt();

        double errorMax=-1;
        for (double x : rpoint) {
            double errorx = abs(qtt.eval({x}) - ft(x));
            if (errorx>errorMax) errorMax=errorx;
        }
        REQUIRE(errorMax < 1e-8);

        for (auto i = 0; i < 200; i++) {
            auto x = 1.0*(rand()%(1<<nBit))/(1<<nBit);
            double errorx = abs(qtt.eval({x}) - ft(x));
            if (errorx>errorMax) errorMax=errorx;
        }
        REQUIRE(errorMax < 1e-8);
    }
}
