#include "sampler_id/interp_decomp.h"
#include "sampler_id/lapack_wrapper.h"

namespace ttid {

static int64_t rank_from_diag(const torch::Tensor& R, double tol) {
    auto diag = R.diagonal().abs();
    const int64_t p = diag.size(0);
    double d0 = diag[0].item().toDouble();
    if (d0 == 0.0) return 1;
    for (int64_t i = 1; i < p; ++i)
        if (diag[i].item().toDouble() < tol * d0) return i;
    return p;
}

static ColID build_col_id(const torch::Tensor& A, const torch::Tensor& R,
                           const std::vector<int>& P, int64_t k)
{
    const int64_t n = A.size(1);
    auto Rk  = R.slice(0, 0, k).contiguous();   // k×n
    auto R11 = Rk.slice(1, 0, k);
    auto R12 = Rk.slice(1, k, n);
    auto T   = at::linalg_solve_triangular(R11, R12, /*upper=*/true);

    auto P_perm = torch::cat({torch::eye(k, A.options()), T}, /*dim=*/1);
    auto Pmat   = torch::empty_like(P_perm);
    for (int64_t i = 0; i < n; ++i)
        Pmat.select(1, P[i]).copy_(P_perm.select(1, i));

    std::vector<int> cols(k);
    for (int i = 0; i < k; ++i) cols[i] = P[i];

    auto diag = R11.diagonal().abs();
    std::vector<double> sv(k);
    for (int64_t i = 0; i < k; ++i) sv[i] = diag[i].item().toDouble();

    return {std::move(cols), Pmat, std::move(sv)};
}

ColID interp_decomp_cols(const torch::Tensor& A, int64_t k) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(k > 0 && k <= std::min(A.size(0), A.size(1)), "k out of range");
    auto [R, P] = rrQR(A);
    return build_col_id(A, R, P, k);
}

ColID interp_decomp_cols(const torch::Tensor& A, double tol) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(tol > 0.0 && tol < 1.0, "tol must be in (0,1)");
    auto [R, P] = rrQR(A);
    return build_col_id(A, R, P, rank_from_diag(R, tol));
}

RowID interp_decomp_rows(const torch::Tensor& A, int64_t k) {
    auto col = interp_decomp_cols(A.t().contiguous(), k);
    return {std::move(col.cols), col.P.t().contiguous(), std::move(col.sv)};
}

RowID interp_decomp_rows(const torch::Tensor& A, double tol) {
    auto col = interp_decomp_cols(A.t().contiguous(), tol);
    return {std::move(col.cols), col.P.t().contiguous(), std::move(col.sv)};
}

} // namespace ttid
