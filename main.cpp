#include <iostream>
#include <torch/torch.h>
#include "id.h"

int main() {
    torch::manual_seed(42);

    // Build an exact rank-3 matrix: A = U V^T
    int64_t m = 20, n = 15, r = 3;
    auto U = torch::randn({m, r}, torch::kFloat64);
    auto V = torch::randn({n, r}, torch::kFloat64);
    auto A = torch::mm(U, V.t());  // m×n, exact rank r

    auto [cols, P, sv] = id_cols(A, r);

    // Reconstruct: A_approx = A[:, cols] * P
    auto C = A.index({torch::indexing::Slice(),
                      torch::tensor(cols, torch::kLong)});  // m×r
    auto A_approx = torch::mm(C, P);

    double err = (A - A_approx).norm().item().toDouble() / A.norm().item().toDouble();

    std::cout << "Selected columns: ";
    for (auto c : cols) std::cout << c << " ";
    std::cout << "\nSingular values: ";
    for (auto s : sv) std::cout << s << " ";
    std::cout << "\nRelative reconstruction error (exact rank-3): " << err << "\n";

    // Approximate ID on a full-rank matrix
    auto B = torch::randn({m, n}, torch::kFloat64);
    auto [cols2, P2, sv2] = id_cols(B, r);
    auto C2 = B.index({torch::indexing::Slice(),
                       torch::tensor(cols2, torch::kLong)});
    double err2 = (B - torch::mm(C2, P2)).norm().item().toDouble() / B.norm().item().toDouble();
    std::cout << "Relative reconstruction error (rank-3 approx of random): " << err2 << "\n";

    return 0;
}
