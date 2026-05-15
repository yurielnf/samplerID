#include <catch2/catch_test_macros.hpp>
#include "../tt_ci.h"
#include <iostream>
#include <random>

using namespace std;

TEST_CASE( "Test tensor CI", "[tt_ci]" )
{
    SECTION( "cos and sum are rank 2" )
    {
        int dim=5, d=10;
        long count=0;
        function myf=[&](vector<int> id) {
            count++;
            auto sum=accumulate(id.begin(), id.end(), 0.0);
            return sum+cos(sum);
        };

        tt_ci_param p;
        p.use_cache = true;  // Enable caching for this test to minimize function evaluations
        auto ci=tt_ci(myf, vector(dim,d), p);
        ci.iterate(20);
        int bond_dim = ci.tt.cores[dim/2-1].size(0);
        REQUIRE( bond_dim==4 );

        vector<int> ids={3,5,1,5,1};
        auto tt=ci.tt;
        REQUIRE( abs(tt.eval(ids)-myf(ids))<1e-5 );
        REQUIRE( abs(tt.eval(ids)-myf(ids))<1e-5 );

        REQUIRE( count < int(pow(d * bond_dim, 2)*(dim-1)) );
    }

    SECTION( "interpolating 2^n/(1+2*sum(x)) fast convergence" )
    {
        int dim=5, d=15;
        function myf=[&](vector<int> id) {
            // Map indices to x values in [0,1]
            double sum_x = 0.0;
            for (int i = 0; i < dim; ++i) {
                double x_i = id[i] / 14.0;  // 15 points: 0, 1/14, 2/14, ..., 14/14
                sum_x += x_i;
            }
            return pow(2.0, dim) / (1.0 + 2.0 * sum_x);
        };

        auto ci=tt_ci(myf, vector(dim,d));
        ci.iterate(11);
        REQUIRE(ci.tt.cores[1].sizes()[2]==11);
        REQUIRE(ci.pivotError.at(10) < 1e-9);
    }
}

TEST_CASE("global pivots with linear cost", "[tt_ci]")
{
    SECTION( "interpolating 2^n/(1+2*sum(x)) for a lot of points" )
    {
        int dim=10, d=15;
        function myf=[&](vector<int> id) {
            // Map indices to x values in [0,1]
            double sum_x = 0.0;
            for (int i = 0; i < dim; ++i) {
                double x_i = id[i] / 14.0;  // 15 points: 0, 1/14, 2/14, ..., 14/14
                sum_x += x_i;
            }
            return pow(2.0, dim) / (1.0 + 2.0 * sum_x);
        };

        int nPivots=2000;
        std::cout<<"nPivot=";
        std::cin>>nPivots;
        vector<vector<int>> pivots(nPivots); // list of pivots
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, d-1);
        for (auto& pivot : pivots) {
            pivot.resize(dim);
            for (auto& idx : pivot) idx = dist(rng);
        }

        auto ci=tt_ci(myf, vector(dim,d));
        // ci.iterate(2);
        ci.addPivotsAllBonds(pivots);
        
        std::cout << "bond dim = " << ci.tt.cores[1].sizes()[2] << "\n";

        for( auto & id:pivots) {
            auto y = myf(id);
            auto y_ci = ci.tt.eval(id);
            REQUIRE(abs(y-y_ci) <= 1e-10 * abs(y));
        }
    }

}
