// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>
#include <dolfinx.h>
#include <mpi.h>

#include <cstdint>

#include "memory.hpp"

using namespace dolfinx;

namespace dolfinx::experimental::sycl::la
{

struct CsrMatrix
{
  double* data;
  std::int32_t* indptr;
  std::int32_t* indices;

  std::int32_t nrows;
  std::int32_t ncols;
};

struct matrix_acc_map_t
{
  std::int32_t* forward;
  std::int32_t* reverse;
  std::int32_t size;
};

struct AdjacencyList
{
  std::int32_t* indptr;
  std::int32_t* indices;
  std::int32_t num_nodes;
  std::int32_t num_links;
};

std::pair<CsrMatrix, AdjacencyList>
create_sparsity_pattern(MPI_Comm comm, cl::sycl::queue& queue,
                        const experimental::sycl::memory::form_data_t& data,
                        int verbose_mode = 1);

AdjacencyList
compute_vector_acc_map(MPI_Comm comm, cl::sycl::queue& queue,
                       const experimental::sycl::memory::form_data_t& data);

} // namespace dolfinx::experimental::sycl::la