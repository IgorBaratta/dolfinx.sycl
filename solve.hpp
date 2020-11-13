#include <cstdint>
#include <vector>

namespace dolfinx_sycl
{
void solve(double* A, double* b, double* x, std::int32_t* coo_rows,
           std::int32_t* coo_cols, std::int32_t stored_nz, int ndofs);
}
