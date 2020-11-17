// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>
#include <dolfinx.h>

#include <cstdint>

#include "timing.hpp"

using namespace dolfinx;

namespace dolfinx::experimental::sycl::memory
{

// TODO: create class ?
struct form_data_t
{
  double* x;
  std::int32_t* xdofs;
  double* coeffs_L;
  double* coeffs_a;
  std::int32_t* dofs;

  std::int32_t ndofs;
  std::int32_t ncells;
  int ndofs_cell;
};

form_data_t send_form_data(MPI_Comm comm, cl::sycl::queue& queue,
                           const fem::Form<double>& L,
                           const fem::Form<double>& a, int verbose_mode = 1);

} // namespace dolfinx::experimental::sycl::memory