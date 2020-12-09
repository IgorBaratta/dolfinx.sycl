// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#include <numeric>
#include <vector>

namespace dolfinx::experimental::sycl::algorithms
{
//--------------------------------------------------------------------------
// A utility function to swap two elements
template <typename T>
void swap(T* a, T* b)
{
  T t = *a;
  *a = *b;
  *b = t;
}
//--------------------------------------------------------------------------
void exclusive_scan(cl::sycl::queue& queue, std::int32_t* input,
                    std::int32_t* output, std::int32_t size)
{
  //FIXME: do not copy data back to host!!!!!!!!!!!!!!
  std::vector<std::int32_t> in(size, 0);
  std::vector<std::int32_t> out(size + 1, 0);

  queue.memcpy(in.data(), input, size * sizeof(std::int32_t)).wait();

  std::partial_sum(in.begin(), in.end(), out.begin() + 1);

  queue.memcpy(output, out.data(), (size + 1) * sizeof(std::int32_t)).wait();
}

} // namespace dolfinx::experimental::sycl::algorithms