
#include <CL/sycl.hpp>
#include <dolfinx.h>

#include "assemble_impl.hpp"
#include "poisson.h"

namespace dolfinx_sycl::assemble
{
struct device_data_t
{
  double* x;
  std::int32_t* xdofs;
  double* coeffs_L;
  double* coeffs_a;

  std::int32_t ndofs;
  std::int32_t ncells;
  int ndofs_cell;
};

device_data_t send_data_to_device(cl::sycl::queue& queue,
                                  const dolfinx::fem::Form<double>& L,
                                  const dolfinx::fem::Form<double>& a)
{
  auto mesh = L.mesh();
  auto dofmap = L.function_spaces()[0]->dofmap();
  int tdim = mesh->topology().dim();
  std::int32_t ndofs
      = dofmap->index_map->size_local() + dofmap->index_map->num_ghosts();
  std::int32_t ncells = mesh->topology().index_map(tdim)->size_local()
                        + mesh->topology().index_map(tdim)->num_ghosts();
  int ndofs_cell = dofmap->list().num_links(0);

  // Send geometry data to device
  const auto& geometry = mesh->geometry().x();
  auto x_d = cl::sycl::malloc_device<double>(geometry.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_d, geometry.data(), sizeof(double) * geometry.size());
  });

  // Send geometry dofmap to device
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  auto x_dofs_d = cl::sycl::malloc_device<std::int32_t>(x_dofmap.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_dofs_d, x_dofmap.data(), sizeof(std::int32_t) * x_dofmap.size());
  });

  // Send RHS coefficients to device
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(L);
  auto coeff_d = cl::sycl::malloc_device<double>(coeffs.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_d, coeffs.data(), sizeof(double) * coeffs.size());
  });

  // Send RHS coefficients to device
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs_a = dolfinx::fem::pack_coefficients(L);
  auto coeff_a_d = cl::sycl::malloc_device<double>(coeffs_a.size(), queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_a_d, coeffs_a.data(), sizeof(double) * coeffs_a.size());
  });

  queue.wait();

  return {x_d, x_dofs_d, coeff_d, coeff_a_d, ndofs, ncells, ndofs_cell};
}

// Submit vector assembly kernels to queue
double* assemble_vector(cl::sycl::queue& queue, device_data_t& data)
{
  std::int32_t ndofs_ext = data.ndofs_cell * data.ncells;
  auto b_ext = cl::sycl::malloc_device<double>(ndofs_ext, queue);

  const double fill_value = 0.;
  queue.submit([&](cl::sycl::handler& h) {
    h.fill<double>(b_ext, fill_value, ndofs_ext);
  });

  assemble_vector_impl(queue, b_ext, data.x, data.xdofs, data.coeffs_L,
                       data.ncells, data.ndofs, data.ndofs_cell);

  return b_ext;
}

// Submit vector assembly kernels to queue
double*
accumulate_vector(cl::sycl::queue& queue, double* b_ext, device_data_t& data,
                  const dolfinx::graph::AdjacencyList<std::int32_t>& perm)
{
  std::int32_t ndofs = data.ndofs;

  auto b = cl::sycl::malloc_device<double>(ndofs, queue);
  std::size_t offset_size = perm.offsets().size();
  std::size_t indices_size = perm.array().size();
  auto offsets = cl::sycl::malloc_device<std::int32_t>(offset_size, queue);
  auto indices = cl::sycl::malloc_device<std::int32_t>(indices_size, queue);

  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(offsets, perm.offsets().data(),
             sizeof(std::int32_t) * perm.offsets().size());
  });

  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(indices, perm.array().data(),
             sizeof(std::int32_t) * perm.array().size());
  });

  queue.wait_and_throw();

  accumulate_vector_impl(queue, b, b_ext, offsets, indices, data.ndofs);

  return b;
}

// Submit vector assembly kernels to queue
double* assemble_matrix(cl::sycl::queue& queue, device_data_t& data)
{
  std::int32_t ext_size = data.ncells * data.ndofs_cell * data.ndofs_cell;

  auto A_ext = cl::sycl::malloc_device<double>(ext_size, queue);

  const double fill_value = 0.;
  queue.submit([&](cl::sycl::handler& h) {
    h.fill<double>(A_ext, fill_value, ext_size);
  });

  queue.wait_and_throw();

  assemble_matrix_impl(queue, A_ext, data.x, data.xdofs, data.coeffs_a,
                       data.ncells, data.ndofs, data.ndofs_cell);

  return A_ext;
}

} // namespace dolfinx_sycl::assemble