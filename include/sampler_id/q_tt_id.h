#pragma once
#include "tt_id.h"
#include "grid.h"

namespace ttid {

struct q_tt_id_param {
    int bondDim = 30;
    double reltol = 1e-12;
    std::vector<int> pivot1;
    bool use_cache = false;

    tt_id_param to_base() const { return {bondDim, reltol, pivot1, use_cache}; }
};

template<typename T>
struct QTensorTrain {
    TensorTrain<T> tt;
    grid::Quantics grid;

    T eval(std::vector<double> const& xs) const {
        return tt.eval(grid.coord_to_id(xs));
    }
};

template<typename T = double>
struct q_tt_id : tt_id<T> {
    grid::Quantics grid;

    q_tt_id(std::function<T(double)> f_, grid::Quantics grid_, q_tt_id_param par = {})
        : tt_id<T>(wrap(f_, grid_), grid_.tensorDims(), par.to_base())
        , grid(grid_)
    {}

    QTensorTrain<T> get_qtt() const { return {this->tt, grid}; }

    void addPivotValues(std::vector<double> const& xs)
    {
        std::vector<std::vector<int>> pivots;
        pivots.reserve(xs.size());
        for (double x : xs)
            pivots.push_back(grid.coord_to_id({x}));
        this->addPivotsAllBonds(pivots);
    }

private:
    static typename tt_id<T>::tensor_fun wrap(std::function<T(double)> f_, grid::Quantics grid_)
    {
        return [f_, grid_](std::vector<int> const& id) -> T {
            return f_(grid_.id_to_coord(id)[0]);
        };
    }
};

} // namespace ttid
