#include "id.h"
#include <vector>

extern "C" {
void sgeqp3_(int* m, int* n, float* a, int* lda, int* jpvt, float* tau,
             float* work, int* lwork, int* info);
void dgeqp3_(int* m, int* n, double* a, int* lda, int* jpvt, double* tau,
             double* work, int* lwork, int* info);
}

template<typename T>
static void geqp3(int m, int n, T* a, int* jpvt, T* tau);

template<>
void geqp3<float>(int m, int n, float* a, int* jpvt, float* tau) {
    int lwork = -1, info = 0;
    float wq;
    sgeqp3_(&m, &n, a, &m, jpvt, tau, &wq, &lwork, &info);
    lwork = static_cast<int>(wq);
    std::vector<float> work(lwork);
    sgeqp3_(&m, &n, a, &m, jpvt, tau, work.data(), &lwork, &info);
    TORCH_CHECK(info == 0, "sgeqp3 failed, info=", info);
}

template<>
void geqp3<double>(int m, int n, double* a, int* jpvt, double* tau) {
    int lwork = -1, info = 0;
    double wq;
    dgeqp3_(&m, &n, a, &m, jpvt, tau, &wq, &lwork, &info);
    lwork = static_cast<int>(wq);
    std::vector<double> work(lwork);
    dgeqp3_(&m, &n, a, &m, jpvt, tau, work.data(), &lwork, &info);
    TORCH_CHECK(info == 0, "dgeqp3 failed, info=", info);
}

// Run pivoted QR on A and return the column-major result Af (upper triangle = R)
// together with the pivot vector jpvt (1-indexed).
static std::pair<torch::Tensor, std::vector<int32_t>>
run_geqp3(const torch::Tensor& A) {
    const int64_t m = A.size(0), n = A.size(1);
    auto Af = torch::empty_strided({m, n}, {1, m}, A.options());
    Af.copy_(A);
    std::vector<int32_t> jpvt(n, 0);
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "id", [&]() {
        std::vector<scalar_t> tau(std::min(m, n));
        geqp3<scalar_t>((int)m, (int)n,
                        Af.data_ptr<scalar_t>(), jpvt.data(), tau.data());
    });
    return {Af, jpvt};
}

// Build IDResult given already-computed pivoted QR (Af, jpvt) and rank k.
static IDResult build_id(const torch::Tensor& A,
                          const torch::Tensor& Af,
                          const std::vector<int32_t>& jpvt,
                          int64_t k) {
    const int64_t n = A.size(1);
    auto R   = torch::triu(Af.slice(0, 0, k)).contiguous();  // k×n
    auto R11 = R.slice(1, 0, k);
    auto R12 = R.slice(1, k, n);
    auto T   = at::linalg_solve_triangular(R11, R12, /*upper=*/true);

    std::vector<at::Tensor> parts = {torch::eye(k, A.options()), T};
    auto P_perm = torch::cat(parts, /*dim=*/1);
    auto P = torch::empty_like(P_perm);
    for (int64_t i = 0; i < n; ++i)
        P.select(1, jpvt[i] - 1).copy_(P_perm.select(1, i));

    std::vector<int> cols(k);
    for (int i = 0; i < k; ++i)
        cols[i] = jpvt[i] - 1;

    // Extract singular values from R11 diagonal
    auto diag = R11.diagonal().abs();
    std::vector<double> sv(k);
    AT_DISPATCH_FLOATING_TYPES(diag.scalar_type(), "id_sv", [&]() {
        for (int64_t i = 0; i < k; ++i)
            sv[i] = diag[i].item<double>();
    });

    IDResult result;
    result.cols = std::move(cols);
    result.P    = P;
    result.sv   = sv;
    return result;
}

// Determine numerical rank from the R diagonal: keep indices where
// |R[i,i]| / |R[0,0]| >= tol.
static int64_t rank_from_diag(const torch::Tensor& Af, double tol) {
    auto diag = Af.diagonal().abs();
    const int64_t p = diag.size(0);
    double d0 = diag[0].item<double>();
    if (d0 == 0.0) return 1;
    for (int64_t i = 1; i < p; ++i)
        if (diag[i].item<double>() < tol * d0) return i;
    return p;
}

static void check_inputs(const torch::Tensor& A) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(A.device() == torch::kCPU, "A must be on CPU");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 || A.scalar_type() == torch::kFloat64,
                "A must be float32 or float64");
}

IDResult id_cols(const torch::Tensor& A, int64_t k) {
    check_inputs(A);
    TORCH_CHECK(k > 0 && k <= std::min(A.size(0), A.size(1)), "k out of range [1, min(m,n)]");
    auto [Af, jpvt] = run_geqp3(A);
    return build_id(A, Af, jpvt, k);
}

IDResult id_cols(const torch::Tensor& A, double tol) {
    check_inputs(A);
    TORCH_CHECK(tol > 0.0 && tol < 1.0, "tol must be in (0, 1)");
    auto [Af, jpvt] = run_geqp3(A);
    return build_id(A, Af, jpvt, rank_from_diag(Af, tol));
}

RowIDResult id_rows(const torch::Tensor& A, int64_t k) {
    auto col = id_cols(A.t().contiguous(), k);
    RowIDResult result;
    result.rows = std::move(col.cols);
    result.P    = col.P.t().contiguous();
    result.sv   = std::move(col.sv);
    return result;
}

RowIDResult id_rows(const torch::Tensor& A, double tol) {
    auto col = id_cols(A.t().contiguous(), tol);
    RowIDResult result;
    result.rows = std::move(col.cols);
    result.P    = col.P.t().contiguous();
    result.sv   = std::move(col.sv);
    return result;
}
