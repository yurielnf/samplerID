#include <iostream>
#include <torch/torch.h>
#include "sampler_id/interp_decomp.h"
using namespace ttid;

int main() {
    torch::manual_seed(42);

    int64_t m = 20, n = 15, r = 3;
    auto U = torch::randn({m, r}, torch::kFloat64);
    auto V = torch::randn({n, r}, torch::kFloat64);
    auto A = torch::mm(U, V.t());  // rank-r matrix

    auto [cols, P, sv] = interp_decomp_cols(A, 0.0, r);

    auto C     = A.index({torch::indexing::Slice(), torch::tensor(cols, torch::kLong)});
    auto A_approx = torch::mm(C, P);
    double err = (A - A_approx).norm().item().toDouble() / A.norm().item().toDouble();

    std::cout << "Selected columns: ";
    for (auto c : cols) std::cout << c << " ";
    std::cout << "\nSingular values: ";
    for (auto s : sv) std::cout << s << " ";
    std::cout << "\nRelative error (exact rank-3): " << err << "\n";

    auto B  = torch::randn({m, n}, torch::kFloat64);
    auto [cols2, P2, sv2] = interp_decomp_cols(B, r);
    auto C2  = B.index({torch::indexing::Slice(), torch::tensor(cols2, torch::kLong)});
    double err2 = (B - torch::mm(C2, P2)).norm().item().toDouble() / B.norm().item().toDouble();
    std::cout << "Relative error (rank-3 approx of random): " << err2 << "\n";

    return 0;
}
