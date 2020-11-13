// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <dolfinx.h>
#include <mpi.h>

using namespace dolfinx;

namespace dolfinx_sycl::utils
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
    return cl::sycl::queue(cl::sycl::host_selector(), exception_handler, {});
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
    const std::shared_ptr<dolfinx::function::FunctionSpace>& V)
{
  const auto& mesh = V->mesh();
  int tdim = mesh->topology().dim();
  std::cout << "\nNumber of cells: "
            << mesh->topology().index_map(tdim)->size_global() << std::endl;
  std::cout << "Number of dofs: " << V->dofmap()->index_map->size_global()
            << std::endl;
}

//--------------------------------------------------------------------------
void print_timing_info(
    MPI_Comm mpi_comm,
    const std::map<std::string, std::chrono::duration<double>>& timings)
{
  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  if (mpi_rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0)
    {
      std::string pad(32 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
  }
}

} // namespace dolfinx_sycl::utils