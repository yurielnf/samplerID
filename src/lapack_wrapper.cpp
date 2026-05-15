#include "sampler_id/lapack_wrapper.h"

extern "C" {
void sgeqp3_(int* m, int* n, float*  a, int* lda, int* jpvt, float*  tau,
             float*  work, int* lwork, int* info);
void dgeqp3_(int* m, int* n, double* a, int* lda, int* jpvt, double* tau,
             double* work, int* lwork, int* info);
void cgeqp3_(int* m, int* n, void*   a, int* lda, int* jpvt, void*   tau,
             void*   work, int* lwork, float*  rwork, int* info);
void zgeqp3_(int* m, int* n, void*   a, int* lda, int* jpvt, void*   tau,
             void*   work, int* lwork, double* rwork, int* info);
}

template<typename T> static void geqp3(int m, int n, T* a, int* jpvt, T* tau);

template<> void geqp3<float>(int m, int n, float* a, int* jpvt, float* tau) {
    int lwork = -1, info = 0; float wq;
    sgeqp3_(&m, &n, a, &m, jpvt, tau, &wq, &lwork, &info);
    lwork = (int)wq; std::vector<float> work(lwork);
    sgeqp3_(&m, &n, a, &m, jpvt, tau, work.data(), &lwork, &info);
    TORCH_CHECK(info == 0, "sgeqp3 failed, info=", info);
}

template<> void geqp3<double>(int m, int n, double* a, int* jpvt, double* tau) {
    int lwork = -1, info = 0; double wq;
    dgeqp3_(&m, &n, a, &m, jpvt, tau, &wq, &lwork, &info);
    lwork = (int)wq; std::vector<double> work(lwork);
    dgeqp3_(&m, &n, a, &m, jpvt, tau, work.data(), &lwork, &info);
    TORCH_CHECK(info == 0, "dgeqp3 failed, info=", info);
}

template<> void geqp3<c10::complex<float>>(int m, int n,
    c10::complex<float>* a, int* jpvt, c10::complex<float>* tau)
{
    int lwork = -1, info = 0;
    c10::complex<float> wq; std::vector<float> rwork(2 * n);
    cgeqp3_(&m, &n, (void*)a, &m, jpvt, (void*)tau, (void*)&wq, &lwork, rwork.data(), &info);
    lwork = (int)wq.real(); std::vector<c10::complex<float>> work(lwork);
    cgeqp3_(&m, &n, (void*)a, &m, jpvt, (void*)tau, (void*)work.data(), &lwork, rwork.data(), &info);
    TORCH_CHECK(info == 0, "cgeqp3 failed, info=", info);
}

template<> void geqp3<c10::complex<double>>(int m, int n,
    c10::complex<double>* a, int* jpvt, c10::complex<double>* tau)
{
    int lwork = -1, info = 0;
    c10::complex<double> wq; std::vector<double> rwork(2 * n);
    zgeqp3_(&m, &n, (void*)a, &m, jpvt, (void*)tau, (void*)&wq, &lwork, rwork.data(), &info);
    lwork = (int)wq.real(); std::vector<c10::complex<double>> work(lwork);
    zgeqp3_(&m, &n, (void*)a, &m, jpvt, (void*)tau, (void*)work.data(), &lwork, rwork.data(), &info);
    TORCH_CHECK(info == 0, "zgeqp3 failed, info=", info);
}

std::tuple<torch::Tensor, std::vector<int>> rrQR(const torch::Tensor& A) {
    TORCH_CHECK(A.dim() == 2 && A.device() == torch::kCPU, "rrQR: need 2D CPU tensor");
    const int m = A.size(0), n = A.size(1), k = std::min(m, n);
    auto Af = torch::empty_strided({m, n}, {1, m}, A.options());
    Af.copy_(A);
    std::vector<int32_t> jpvt(n, 0);

    switch (A.scalar_type()) {
        case torch::kFloat32: {
            std::vector<float> tau(k);
            geqp3<float>(m, n, Af.data_ptr<float>(), jpvt.data(), tau.data());
            break;
        }
        case torch::kFloat64: {
            std::vector<double> tau(k);
            geqp3<double>(m, n, Af.data_ptr<double>(), jpvt.data(), tau.data());
            break;
        }
        case torch::kComplexFloat: {
            std::vector<c10::complex<float>> tau(k);
            geqp3<c10::complex<float>>(m, n,
                Af.data_ptr<c10::complex<float>>(), jpvt.data(), tau.data());
            break;
        }
        case torch::kComplexDouble: {
            std::vector<c10::complex<double>> tau(k);
            geqp3<c10::complex<double>>(m, n,
                Af.data_ptr<c10::complex<double>>(), jpvt.data(), tau.data());
            break;
        }
        default:
            TORCH_CHECK(false, "rrQR: unsupported dtype; use float32/64 or complex64/128");
    }

    auto R = torch::triu(Af.slice(0, 0, k)).contiguous();
    std::vector<int> P(n);
    for (int i = 0; i < n; ++i) P[i] = jpvt[i] - 1;
    return {R, P};
}
