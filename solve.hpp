// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <cstdint>
#include <string>

namespace dolfinx::experimental::sycl::solve
{

double ginkgo(double* A, std::int32_t* indptr, std::int32_t* indices,
              std::int32_t nrows, std::int32_t nnz, double* b, double* x,
              std::string executor);

}
