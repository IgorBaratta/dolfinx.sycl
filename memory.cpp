// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <cassert>

#include "memory.hpp"

using namespace dolfinx::experimental::sycl;

memory::form_data_t dolfinx::experimental::sycl::memory::send_form_data(
    MPI_Comm comm, cl::sycl::queue& queue, const fem::Form<double>& L,
    const fem::Form<double>& a, int verbose_mode)
{

  assert(L.rank() == 1);
  assert(a.rank() == 2);

  std::string step{"Send Form Data to Device"};
  std::map<std::string, std::chrono::duration<double>> timings;

  auto start = std::chrono::system_clock::now();

  auto mesh = L.mesh();
  auto dofmap = L.function_spaces()[0]->dofmap();
  int tdim = mesh->topology().dim();
  std::int32_t ndofs
      = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local()
                        + mesh->topology().index_map(tdim)->num_ghosts();
  int ndofs_cell = dofmap->list().num_links(0);

  // Send coordinates to device
  auto timer_start = std::chrono::system_clock::now();
  const auto& geometry = mesh->geometry().x();
  auto x_d = cl::sycl::malloc_device<double>(geometry.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_d, geometry.data(), sizeof(double) * geometry.size());
  });
  if (verbose_mode > 1)
    queue.wait();
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Send Coordinates to Device"] = (timer_end - timer_start);

  // Send geometry dofmap to device
  timer_start = std::chrono::system_clock::now();
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  auto x_dofs_d = cl::sycl::malloc_device<std::int32_t>(x_dofmap.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_dofs_d, x_dofmap.data(), sizeof(std::int32_t) * x_dofmap.size());
  });
  if (verbose_mode > 1)
    queue.wait();
  timer_end = std::chrono::system_clock::now();
  timings["1 - Send Coordinate Map to Device"] = (timer_end - timer_start);

  // Send RHS coefficients to device
  timer_start = std::chrono::system_clock::now();
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(L);
  auto coeff_d = cl::sycl::malloc_device<double>(coeffs.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_d, coeffs.data(), sizeof(double) * coeffs.size());
  });
  if (verbose_mode > 1)
    queue.wait();
  timer_end = std::chrono::system_clock::now();
  timings["2 - Send RHS Coefficies to Device"] = (timer_end - timer_start);

  // Send LHS coefficients to device
  timer_start = std::chrono::system_clock::now();
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs_a = dolfinx::fem::pack_coefficients(L);
  auto coeff_a_d = cl::sycl::malloc_device<double>(coeffs_a.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_a_d, coeffs_a.data(), sizeof(double) * coeffs_a.size());
  });
  if (verbose_mode > 1)
    queue.wait();
  timer_end = std::chrono::system_clock::now();

  timings["3 - Send LHS Coefficies to Device"] = (timer_end - timer_start);

  // Send dofmap to device
  timer_start = std::chrono::system_clock::now();
  auto& dofs = dofmap->list().array();
  auto dofs_d = cl::sycl::malloc_device<std::int32_t>(dofs.size(), queue);
  auto e = queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(dofs_d, dofs.data(), sizeof(std::int32_t) * dofs.size());
  });
  timer_end = std::chrono::system_clock::now();
  if (verbose_mode > 1)
    queue.wait();
  timings["4 - Send Dof Array to Device"] = (timer_end - timer_start);

  queue.wait();

  auto end = std::chrono::system_clock::now();
  timings["Total"] = (end - start);

  dolfinx::experimental::sycl::timing::print_timing_info(comm, timings, step,
                                                         verbose_mode);

  return {x_d, x_dofs_d, coeff_d, coeff_a_d, dofs_d, ndofs, ncells, ndofs_cell};
}