#ifndef TT_CI_H
#define TT_CI_H

#include "id.h"
#include "index_set.h"
#include "tt.h"
#include <map>
#include <vector>

using std::vector;
using namespace xfac;

/// Parameters of the tt_ci algorithm.
struct tt_ci_param {
    int bondDim=30;                         ///< the max bond dimension of the tensor train
    double reltol=1e-12;                    ///< CI will stop when pivotError < reltol*max(pivotError)
    vector<int> pivot1;                     ///< the first pivot (optional)
    bool use_cache=false;                   ///< enable memoization of function evaluations
};

/// This class is responsible for building an interpolative decomposition (ID) of an input tensor function.
/// The main output is the tensor train tt (an effective separation of variables),
/// which allows many cheap computations, i.e. the integral.
struct tt_ci
{
    using T=double;
    using tensor_fun=std::function<double(vector<int>)>;

    tensor_fun f;                     ///< the tensor function f:(a1,a2,...,an)->T
    tt_ci_param param;                                      ///< parameters of the algorithm
    vector<double> pivotError;                              ///< max pivot error for each rank
    vector< IndexSet<MultiIndex> > Iset, localSet, Jset;    ///< collection of MultiIndex for each site: left, site, and right set of multiindex
    TensorTrain tt;                                         ///< main output of the Tensor CI
    int cIter=0;                                            ///< counter of iterations. Used for sweeping only.
    int center=0;                                           ///< current position of the CI center

    tt_ci();

    /// constructs a rank-1 tt-ci from a function f:(a1,a2,...,an)->eT  where the index ai is in [0,localDim[i]).
    tt_ci(tensor_fun const& f_, vector<int> localDim, tt_ci_param param_={})
        : f {f_}
        , param(param_)
        , pivotError(1)
        , Iset {localDim.size()}
        , localSet {localDim.size()}
        , Jset {localDim.size()}
        , tt(localDim.size())
    {
        if (param.pivot1.empty())
            param.pivot1.resize(localDim.size(), 0);
        pivotError[0]=std::abs(f(param.pivot1));
        if (pivotError[0]==0.0)
            throw std::invalid_argument("Not expecting f(pivot1)=0. Provide a better first pivot in the param");

        // fill localSet, Iset Jset
        for(auto p=0u; p<localDim.size(); p++)
            for(auto i=0; i<localDim[p]; i++)
                localSet[p].push_back({char32_t(i)});

        //add param.pivot1;
        for(auto b=0u; b<localDim.size(); b++) {
            Iset[b].push_back({param.pivot1.begin(), param.pivot1.begin()+b});
            Jset[b].push_back({param.pivot1.begin()+b+1, param.pivot1.end()});
        }
        iterate(1,1); // just to define tt
    }

    /// add global pivots.
    void addPivotsAllBonds(vector<vector<int>> const& pivots)
    {
        for(auto i:{1,2}) {
            if (cIter%2==0)
                for(auto b=0u; b<len()-1; b++) addPivotsAt(pivots,b);
            else
                for(int b=len()-2; b>=0; b--) addPivotsAt(pivots,b);
            iterate(1,2);
        }
    }

    /// add these pivots at a given bond b. The tt is not touched.
    void addPivotsAt(vector<vector<int>> const& pivots, int b)
    {
        for (const auto& pg : pivots) {
            if (cIter%2==0)
                Jset[b].push_back({pg.begin()+b+1, pg.end()});
            else
                Iset[b+1].push_back({pg.begin(), pg.begin()+b+1});
        }
    }

    /// returns the length of the tensor
    size_t len() const { return localSet.size(); }

    /// makes nIter half sweeps. The dmrg_type can be 0,1,2
    void iterate(int nIter=1, int dmrg_type=2)
    {
        for(auto i=0; i<nIter; i++) {
            if (cIter%2==0)
                for(auto b=0u; b<len()-1; b++) { center=b+1; updatePivotAt(b, dmrg_type); }
            else
                for(int b=len()-2; b>=0; b--) { center=b; updatePivotAt(b, dmrg_type); }
            cIter++;
        }
    }

protected:

    /// update the pivots at bond b, the dmrg can be 0,1,2.
    void updatePivotAt(int b, int dmrg=2)
    {
        switch (dmrg) {
        case 1: dmrg1_updatePivotAt(b); break;
        case 2: dmrg2_updatePivotAt(b); break;
        }
    }

    /// update the pivots at bond b using the Pi matrix.
    void dmrg2_updatePivotAt(int b)
    {
        IndexSet<MultiIndex> Ib=kron(Iset[b],localSet[b]);
        IndexSet<MultiIndex> Jb=kron(localSet[b+1],Jset[b+1]) ;
        auto M = tensor_fun_to_mat(Ib,Jb);

        if (b<center) {
            auto result = id_rows(M,param.reltol);
            Iset[b+1] = Ib.at(result.rows);
            Jset[b] = Iset[b+1];  // fake, will never be used except for dimensions
            tt.cores[b] = result.P;
            tt.cores[b+1] = M.index({torch::tensor(result.rows, torch::kLong), torch::indexing::Slice()});
            collectPivotError(b, result.sv);
        }
        else {
            auto result=id_cols(M,param.reltol);
            Jset[b]=Jb.at(result.cols);
            Iset[b+1]=Jset[b];  // fake, will never be used except for dimensions
            tt.cores[b] = M.index({torch::indexing::Slice(),torch::tensor(result.cols, torch::kLong)});
            tt.cores[b+1] = result.P;
            collectPivotError(b, result.sv);
        }

        reshape_site_tensor(b);
        reshape_site_tensor(b+1);
    }

    void dmrg1_updatePivotAt(int b)
    {
        bool isLeft=b<center;
        IndexSet<MultiIndex> Ib= isLeft ? kron(Iset[b],localSet[b]) : Iset[b+1].from_int() ;
        IndexSet<MultiIndex> Jb= isLeft ? Jset[b].from_int() : kron(localSet[b+1],Jset[b+1]);
        auto M = tensor_fun_to_mat(Ib,Jb);

        if (b<center) {
            auto result = id_rows(M,param.reltol);
            Iset[b+1] = Ib.at(result.rows);
            Jset[b] = Iset[b+1];  // sync so reshape_site_tensor(b) gets the right c
            tt.cores[b] = result.P;
            tt.cores[b+1] = M.index({torch::tensor(result.rows, torch::kLong), torch::indexing::Slice()});
            collectPivotError(b, result.sv);
        }
        else {
            auto result=id_cols(M,param.reltol);
            Jset[b]=Jb.at(result.cols);
            tt.cores[b] = M.index({torch::indexing::Slice(),torch::tensor(result.cols, torch::kLong)});
            tt.cores[b+1] = result.P;
            collectPivotError(b, result.sv);
        }

        reshape_site_tensor(b);
    }

    // helper
    torch::Tensor tensor_fun_to_mat(vector<MultiIndex> const& I, vector<MultiIndex> const& J) const
    {
        auto values = torch::empty({int(I.size()), int(J.size())}, torch::kFloat64);
        for(auto i=0u; i<I.size(); i++)
            for(auto j=0u; j<J.size(); j++) {
                MultiIndex ij=I[i]+J[j];
                vector<int> key(ij.begin(), ij.end());
                T value;
                if (param.use_cache) {
                    auto it = cache.find(key);
                    if (it != cache.end()) {
                        value = it->second;
                    } else {
                        value = f(key);
                        cache[key] = value;
                    }
                } else {
                    value = f(key);
                }
                values.index_put_({(int)i,(int)j}, value);
            }
        return values;
    }

    // helper
    void reshape_site_tensor(int i)
    {
        auto M = tt.cores[i];
        int64_t
            a = Iset[i].size(),
            b = localSet[i].size(),
            c = Jset[i].size();
        tt.cores[i] = M.reshape({a,b,c});
    }

    //helper
    void collectPivotError(int b, vector<double> const& pe)
    {
        if (pe.size()>pivotError.size()) pivotError.resize(pe.size(), 0);
        for(auto i=0u; i<pe.size(); i++)
            if (pe[i]>pivotError[i])
                pivotError[i]=pe[i];
    }

private:
    mutable std::map<vector<int>, T> cache;   ///< memoization cache for f evaluations

};

#endif // TT_CI_H
