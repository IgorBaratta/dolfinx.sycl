#include <cstdint>
#include <vector>

namespace dolfinx_sycl
{
void solve(double* A, double* b, double* x, std::vector<std::int32_t> rows,
           std::vector<std::int32_t> cols, int ndofs);
}
