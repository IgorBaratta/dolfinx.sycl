// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>
#include <dolfinx.h>

#include "assemble_impl.hpp"
#include "la.hpp"
#include "memory.hpp"
#include "poisson.h"

using namespace dolfinx::experimental::sycl;

namespace dolfinx::experimental::sycl::assemble
{
//--------------------------------------------------------------------------
// Submit vector assembly kernels to queue
double* assemble_vector(MPI_Comm comm, cl::sycl::queue& queue,
                        const memory::form_data_t& data, int verbose_mode = 1)
{

  std::string step{"Assemble vector on device"};
  std::map<std::string, std::chrono::duration<double>> timings;
  std::cout << "Assemble vector on device, starting ....\n";

  auto timer_start = std::chrono::system_clock::now();
  experimental::sycl::la::AdjacencyList acc
      = experimental::sycl::la::compute_vector_acc_map(comm, queue, data);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Create  accumulator from dofmap"] = (timer_end - timer_start);
  std::cout << "Create accumulator ....\n";

  auto start = std::chrono::system_clock::now();

  timer_start = std::chrono::system_clock::now();
  // Allocated unassembled vector on device
  std::int32_t ndofs_ext = data.ndofs_cell * data.ncells;
  auto b_ext = cl::sycl::malloc_device<double>(ndofs_ext, queue);
  queue.fill<double>(b_ext, 0., ndofs_ext).wait();
  // Assemble local contributions
  assemble_vector_impl(queue, b_ext, data.x, data.xdofs, data.coeffs_L,
                       data.ncells, data.ndofs, data.ndofs_cell);
  timer_end = std::chrono::system_clock::now();
  timings["1 - Compute cell contributions"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  double* b = cl::sycl::malloc_device<double>(data.ndofs, queue);
  accumulate_impl(queue, b, b_ext, acc.indptr, acc.indices, acc.num_nodes);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Accumulate cells contributions"] = (timer_end - timer_start);

  // Free temporary device data
  cl::sycl::free(b_ext, queue);
  cl::sycl::free(acc.indices, queue);
  cl::sycl::free(acc.indptr, queue);

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);
  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);

  return b;
}

//--------------------------------------------------------------------------
// Submit vector assembly kernels to queue
experimental::sycl::la::CsrMatrix
assemble_matrix(MPI_Comm comm, cl::sycl::queue& queue,
                const memory::form_data_t& data, int verbose_mode = 1)
{

  // Compute Sparsity pattern
  auto [mat, acc_map] = experimental::sycl::la::create_sparsity_pattern(
      comm, queue, data, verbose_mode);

  std::string step{"Assemble matrix on device"};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto start = std::chrono::system_clock::now();

  // Number of stored nonzeros on the extended COO format
  std::int32_t stored_nz = data.ncells * data.ndofs_cell * data.ndofs_cell;

  auto timer_start = std::chrono::system_clock::now();
  auto A_ext = cl::sycl::malloc_device<double>(stored_nz, queue);
  queue.fill<double>(A_ext, 0., stored_nz).wait_and_throw();
  assemble_matrix_impl(queue, A_ext, data.x, data.xdofs, data.coeffs_a,
                       data.ncells, data.ndofs, data.ndofs_cell);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Compute cell contributions"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  accumulate_impl(queue, mat.data, A_ext, acc_map.indptr, acc_map.indices,
                  acc_map.num_nodes);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Accumulate contributions"] = (timer_end - timer_start);

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);

  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);
  return mat;
}

} // namespace dolfinx::experimental::sycl::assemble