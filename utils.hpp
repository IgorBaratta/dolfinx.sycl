// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <CL/sycl.hpp>

#undef SYCL_DEVICE_ONLY
#include <dolfinx.h>
#define SYCL_DEVICE_ONLY

#include <mpi.h>

using namespace dolfinx;

namespace dolfinx::experimental::sycl::utils
{
//--------------------------------------------------------------------------
void exception_handler(cl::sycl::exception_list exceptions)
{
  for (std::exception_ptr const& e : exceptions)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (cl::sycl::exception const& e)
    {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

//--------------------------------------------------------------------------
/// heuristic to select device
cl::sycl::queue select_queue(MPI_Comm comm)
{
  int mpi_size, mpi_rank;
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);

  std::vector<cl::sycl::device> gpus;
  auto platforms = cl::sycl::platform::get_platforms();
  for (auto& p : platforms)
  {
    auto devices = p.get_devices();
    for (auto& device : devices)
      if (device.is_gpu())
        gpus.push_back(device);
  }

  int num_devices = gpus.size();
  if (num_devices >= mpi_size)
    return cl::sycl::queue(gpus[mpi_rank], exception_handler, {});
  else
    return cl::sycl::queue(cl::sycl::cpu_selector(), exception_handler, {});
}

//--------------------------------------------------------------------------
template <typename T>
void print_device_info(T device)
{
  std::cout
      << "Running on "
      << device.template get_info<cl::sycl::info::device::name>() << "\n"
      << "Number of compute units: "
      << device.template get_info<cl::sycl::info::device::max_compute_units>()
      << "\n"
      << "Driver version: "
      << device.template get_info<cl::sycl::info::device::driver_version>()
      << std::endl;
}

//--------------------------------------------------------------------------
void print_function_space_info(
    const std::shared_ptr<dolfinx::fem::FunctionSpace>& V)
{
  const auto& mesh = V->mesh();
  int tdim = mesh->topology().dim();
  std::cout << "\nNumber of cells: "
            << mesh->topology().index_map(tdim)->size_global() << std::endl;
  std::cout << "Number of dofs: " << V->dofmap()->index_map->size_global()
            << std::endl;
}

} // namespace dolfinx::experimental::sycl::utils