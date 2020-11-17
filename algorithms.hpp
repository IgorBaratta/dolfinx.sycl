// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

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
  std::int32_t N = size;
  if ((size && !(size & (size - 1))) == 0)
  {
    N--;
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    N++;
  }

  if (N < 64)
    N = 64;

  constexpr std::int32_t L = 64;
  const std::size_t G = N / L;

  // Allocate memory on device for working arrays
  auto input_copy = cl::sycl::malloc_device<std::int32_t>(N, queue);
  auto temp = cl::sycl::malloc_device<std::int32_t>(N, queue);
  auto global_work = cl::sycl::malloc_device<std::int32_t>(G, queue);

  // Initialize data
  queue.fill(input_copy, 0, N).wait();
  queue.fill(temp, 0, N).wait();
  queue.memcpy(input_copy, input, sizeof(std::int32_t) * size).wait();

  queue.parallel_for<class LocalScan>(cl::sycl::range<1>(G),
                                      [=](cl::sycl::id<1> it) {
                                        int i = it.get(0);
                                        int offset = i * L;
                                        std::int32_t local_sum = 0;
                                        for (std::size_t j = 0; j < L; j++)
                                          local_sum += input_copy[offset + j];
                                        global_work[i] = local_sum;
                                      });

  queue.wait();

  queue.parallel_for<class LocalUpdate>(
      cl::sycl::range<1>(G), [=](cl::sycl::id<1> Id) {
        int i = Id.get(0);
        std::int32_t local_offset = 0;
        for (int j = 0; j < i; j++)
          local_offset += global_work[j];

        int offset = i * L;
        temp[offset] = local_offset;
        for (std::size_t j = 0; j < L - 1; j++)
          temp[offset + j + 1] = temp[offset + j] + input_copy[offset + j];
      });

  queue.wait();
  queue.memcpy(output, temp, size * sizeof(std::int32_t)).wait();
  output[size] = output[size - 1] + input[size - 1];

  cl::sycl::free(temp, queue);
  cl::sycl::free(global_work, queue);
  cl::sycl::free(input_copy, queue);
}

} // namespace dolfinx::experimental::sycl::algorithms