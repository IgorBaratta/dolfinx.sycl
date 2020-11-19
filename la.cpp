// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "la.hpp"
#include "algorithms.hpp"

using namespace dolfinx;

namespace
{
//--------------------------------------------------------------------------
// Data structure for C++ 17 binding
struct coo_pattern_t
{
  std::int32_t* rows;
  std::int32_t* cols;
  std::int32_t store_nz;
};
//--------------------------------------------------------------------------
inline void swap(std::int32_t* a, std::int32_t* b)
{
  std::int32_t t = *a;
  *a = *b;
  *b = t;
}
//--------------------------------------------------------------------------
coo_pattern_t
create_coo_pattern(cl::sycl::queue& queue,
                   const experimental::sycl::memory::form_data_t& data)
{
  const int ncells = data.ncells;
  const int ndofs_cell = data.ndofs_cell;
  std::int32_t stored_nz = ncells * ndofs_cell * ndofs_cell;

  std::int32_t* dofs = data.dofs;

  auto rows = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);
  auto cols = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);

  int local_size = ndofs_cell * ndofs_cell;
  auto kernel = [=](cl::sycl::id<1> ID) {
    const int i = ID.get(0);

    std::int32_t pos = local_size * i;
    for (int j = 0; j < ndofs_cell; j++)
      for (int k = 0; k < ndofs_cell; k++)
      {
        rows[pos + j * ndofs_cell + k] = dofs[i * ndofs_cell + j];
        cols[pos + j * ndofs_cell + k] = dofs[i * ndofs_cell + k];
      }
  };

  cl::sycl::range<1> range(ncells);
  queue.parallel_for<class CooPattern>(range, kernel);

  queue.wait();

  return {rows, cols, stored_nz};
}
//--------------------------------------------------------------------------
std::pair<experimental::sycl::la::CsrMatrix,
          experimental::sycl::la::matrix_acc_map_t>
coo_to_csr(cl::sycl::queue& queue, coo_pattern_t coo_pattern,
           const experimental::sycl::memory::form_data_t& data)
{
  std::int32_t* rows = coo_pattern.rows;
  std::int32_t* cols = coo_pattern.cols;

  std::int32_t ndofs = data.ndofs;
  std::int32_t ncells = data.ncells;
  std::int32_t ndofs_cell = data.ndofs_cell;
  std::int32_t stored_nz = coo_pattern.store_nz;

  auto counter = cl::sycl::malloc_device<std::int32_t>(ndofs, queue);
  queue.fill(counter, 0, ndofs).wait();

  // Allocate device memory
  auto row_ptr = cl::sycl::malloc_device<std::int32_t>(ndofs + 1, queue);
  auto indices = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);

  auto forward = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);
  auto reverse = cl::sycl::malloc_device<std::int32_t>(stored_nz, queue);

  // Count the number of stored nonzeros per row
  auto count_nonzeros = [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    std::int32_t dofs_pos = i * ndofs_cell * ndofs_cell;
    for (int j = 0; j < ndofs_cell * ndofs_cell; j++)
    {
      std::int32_t row = rows[dofs_pos + j];
      auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
      cl::sycl::atomic<std::int32_t> counter{global_ptr};
      counter.fetch_add(1);
    }
  };

  // Submit kernel to the queue
  cl::sycl::range<1> cell_range(ncells);
  queue.parallel_for<class CountRowNz>(cell_range, count_nonzeros).wait();

  // TODO: Improve exclusive scan implementation for GPUs
  experimental::sycl::algorithms::exclusive_scan(queue, counter, row_ptr,
                                                 ndofs);

  // Populate column indices, might have repeated values
  auto populate_indices = [=](cl::sycl::id<1> id) {
    int i = id.get(0);

    std::int32_t offset = i * ndofs_cell * ndofs_cell;
    for (int j = 0; j < ndofs_cell * ndofs_cell; j++)
    {
      std::int32_t row = rows[offset + j];
      auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[row]);
      cl::sycl::atomic<std::int32_t> state{global_ptr};

      std::int32_t current_count = state.fetch_add(1);
      std::int32_t pos = current_count + row_ptr[row];
      indices[pos] = cols[offset + j];
      // Map old to new data position
      forward[offset + j] = pos;

      // Map new to old data position
      reverse[pos] = offset + j;
    }
  };

  queue.fill(counter, 0, ndofs).wait();
  queue.parallel_for<class ColIndices>(cell_range, populate_indices).wait();

  experimental::sycl::la::CsrMatrix matrix{};
  matrix.indices = indices;
  matrix.indptr = row_ptr;
  matrix.ncols = ndofs;
  matrix.nrows = ndofs;

  experimental::sycl::la::matrix_acc_map_t map{forward, reverse, stored_nz};

  cl::sycl::free(counter, queue);

  return {matrix, map};
}
//--------------------------------------------------------------------------
experimental::sycl::la::CsrMatrix
csr_remove_duplicate(cl::sycl::queue& queue,
                     experimental::sycl::la::CsrMatrix mat,
                     experimental::sycl::la::matrix_acc_map_t map)
{

  std::int32_t nrows = mat.nrows;
  std::int32_t* row_ptr = mat.indptr;
  std::int32_t* indices = mat.indices;

  auto counter = cl::sycl::malloc_device<std::int32_t>(nrows, queue);
  queue.fill(counter, 0, nrows).wait();

  queue.parallel_for<class SortIndices>(
      cl::sycl::range<1>(nrows), [=](cl::sycl::id<1> it) {
        int i = it.get(0);

        std::int32_t begin = row_ptr[i];
        std::int32_t end = row_ptr[i + 1];
        std::int32_t size = end - begin;

        // TODO: Improve performance of sorting algorithm
        for (std::int32_t j = 0; j < size - 1; j++)
          for (std::int32_t k = 0; k < size - j - 1; k++)
            if (indices[begin + k] > indices[begin + k + 1])
            {
              swap(&indices[begin + k], &indices[begin + k + 1]);

              map.forward[map.reverse[begin + k]]++;
              map.forward[map.reverse[begin + k + 1]]--;

              swap(&map.reverse[begin + k], &map.reverse[begin + k + 1]);
            }

        // Count number of unique column indices per row
        std::int32_t temp = -1;
        for (std::int32_t j = 0; j < size; j++)
        {
          if (temp != indices[begin + j])
          {
            counter[i]++;
            temp = indices[begin + j];
          }
        }
      });
  queue.wait();

  // create new csr matrix and remove repeated indices
  experimental::sycl::la::CsrMatrix out;
  out.indptr = cl::sycl::malloc_device<std::int32_t>(nrows + 1, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, out.indptr,
                                                 nrows);
  out.ncols = mat.ncols;
  out.nrows = mat.nrows;

  // number of nonzeros cannot be acessed directly on the host, instead use
  // memcpy
  std::int32_t nnz;
  queue.memcpy(&nnz, &out.indptr[nrows], sizeof(std::int32_t)).wait();
  out.indices = cl::sycl::malloc_device<std::int32_t>(nnz, queue);
  out.data = cl::sycl::malloc_device<double>(nnz, queue);

  queue.parallel_for<class UniqueIndices>(
      cl::sycl::range<1>(nrows), [=](cl::sycl::id<1> it) {
        int i = it.get(0);

        // old rowsize
        std::int32_t row_size = row_ptr[i + 1] - row_ptr[i];

        std::int32_t temp = -1;
        std::int32_t counter = 0;
        for (std::int32_t j = 0; j < row_size; j++)
        {
          if (temp != indices[row_ptr[i] + j])
          {
            temp = indices[row_ptr[i] + j];
            out.indices[out.indptr[i] + counter] = indices[row_ptr[i] + j];
            counter++;
          }
          map.forward[map.reverse[row_ptr[i] + j]]
              = out.indptr[i] + (counter - 1);
        }
      });
  queue.wait();

  return out;
}
//--------------------------------------------------------------------------
experimental::sycl::la::AdjacencyList
transpose_map(cl::sycl::queue& queue,
              experimental::sycl::la::matrix_acc_map_t map, std::int32_t nnz)
{
  // Transpose original to csr position
  auto counter = cl::sycl::malloc_device<std::int32_t>(nnz, queue);
  queue.fill<std::int32_t>(counter, 0, nnz).wait_and_throw();

  // Count the number times the entry appears
  cl::sycl::range<1> range(map.size);
  queue.parallel_for<class CountSharedEntries>(range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);

    std::int32_t entry = map.forward[i];
    auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[entry]);
    cl::sycl::atomic<std::int32_t> state{global_ptr};
    state.fetch_add(1);
  });
  queue.wait_and_throw();

  // Create accumulator adjacency list
  experimental::sycl::la::AdjacencyList tmap;

  tmap.num_nodes = nnz;
  tmap.num_links = map.size;
  tmap.indptr
      = cl::sycl::malloc_device<std::int32_t>(tmap.num_nodes + 1, queue);
  tmap.indices = cl::sycl::malloc_device<std::int32_t>(tmap.num_links, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, tmap.indptr,
                                                 tmap.num_nodes);

  queue.fill<std::int32_t>(counter, 0, nnz).wait_and_throw();

  // Position to accumulate
  queue.parallel_for<class GatherEntries>(range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    std::int32_t entry = map.forward[i];
    auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[entry]);
    cl::sycl::atomic<std::int32_t> state{global_ptr};
    std::int32_t current_count = state.fetch_add(1);
    std::int32_t pos = tmap.indptr[entry] + current_count;
    tmap.indices[pos] = i;
  });

  return tmap;
}
//--------------------------------------------------------------------------
void free_coo(cl::sycl::queue& queue, coo_pattern_t& coo_pattern)
{
  queue.wait();
  cl::sycl::free(coo_pattern.cols, queue);
  cl::sycl::free(coo_pattern.rows, queue);
}
//--------------------------------------------------------------------------
void free_csr(cl::sycl::queue& queue, experimental::sycl::la::CsrMatrix& mat)
{
  queue.wait();
  cl::sycl::free(mat.indices, queue);
  cl::sycl::free(mat.indptr, queue);
}

} // namespace

//--------------------------------------------------------------------------
//--------------------------------------------------------------------------
std::pair<experimental::sycl::la::CsrMatrix,
          experimental::sycl::la::AdjacencyList>
experimental::sycl::la::create_sparsity_pattern(
    MPI_Comm comm, cl::sycl::queue& queue,
    const experimental::sycl::memory::form_data_t& data, int verbose_mode)
{
  std::string step{"Create Local CSR Sparsity Pattern on Device"};
  std::map<std::string, std::chrono::duration<double>> timings;
  auto start = std::chrono::system_clock::now();

  auto timer_start = std::chrono::system_clock::now();
  coo_pattern_t coo_matrix = create_coo_pattern(queue, data);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Create Extended COO pattern"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  auto [csr_mat, acc_map] = coo_to_csr(queue, coo_matrix, data);
  free_coo(queue, coo_matrix);
  timer_end = std::chrono::system_clock::now();
  timings["1 - Convert COO pattern to CSR"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  auto new_mat = csr_remove_duplicate(queue, csr_mat, acc_map);
  free_csr(queue, csr_mat);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Remove duplicated entries"] = (timer_end - timer_start);

  std::int32_t nnz;
  queue.memcpy(&nnz, &new_mat.indptr[new_mat.nrows], sizeof(std::int32_t)).wait();
  auto map = transpose_map(queue, acc_map, nnz);

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);
  experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                verbose_mode);

  return {new_mat, map};
}
//--------------------------------------------------------------------------
experimental::sycl::la::AdjacencyList
experimental::sycl::la::compute_vector_acc_map(
    MPI_Comm comm, cl::sycl::queue& queue,
    const experimental::sycl::memory::form_data_t& data)
{

  auto counter = cl::sycl::malloc_device<std::int32_t>(data.ndofs, queue);
  queue.fill<std::int32_t>(counter, 0, data.ndofs).wait();
  
  std::cout << "break point 1";
  // Count the number times the dof is shared
  cl::sycl::range<1> cell_range(data.ncells);
  queue.parallel_for<class CountSharedDofs>(
      cell_range, [=](cl::sycl::id<1> Id) {
        int i = Id.get(0);
        std::int32_t offset = i * data.ndofs_cell;
        for (int j = 0; j < data.ndofs_cell; j++)
        {
          std::int32_t dof = data.dofs[offset + j];
          auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[dof]);
          cl::sycl::atomic<std::int32_t> state{global_ptr};
          state.fetch_add(1);
        }
      });
  queue.wait();
  
  std::cout << "break point 2";

  // Create accumulator adjacency list
  experimental::sycl::la::AdjacencyList acc;
  acc.num_nodes = data.ndofs;
  acc.num_links = data.ndofs_cell * data.ncells;
  acc.indptr = cl::sycl::malloc_device<std::int32_t>(acc.num_nodes + 1, queue);
  experimental::sycl::algorithms::exclusive_scan(queue, counter, acc.indptr,
                                                 acc.num_nodes);
  std::cout << "break point 3";

  acc.indices = cl::sycl::malloc_device<std::int32_t>(acc.num_links, queue);
  queue.fill<std::int32_t>(counter, 0, data.ndofs).wait();

  // Position to accumulate
  queue.parallel_for<class GatherDofs>(cell_range, [=](cl::sycl::id<1> Id) {
    int i = Id.get(0);
    std::int32_t offset = i * data.ndofs_cell;
    for (int j = 0; j < data.ndofs_cell; j++)
    {
      std::int32_t dof = data.dofs[offset + j];
      auto global_ptr = cl::sycl::global_ptr<std::int32_t>(&counter[dof]);
      cl::sycl::atomic<std::int32_t> state{global_ptr};
      std::int32_t current_count = state.fetch_add(1);

      std::int32_t pos = acc.indptr[dof] + current_count;
      acc.indices[pos] = offset + j;
    }
  });
  std::cout << "break point 4";
  queue.wait();

  return acc;
}