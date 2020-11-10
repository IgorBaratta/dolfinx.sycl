
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
  double* coeffs;

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
  auto x_d = (double*)cl::sycl::malloc_device(sizeof(double) * geometry.size(),
                                              queue);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_d, geometry.data(), sizeof(double) * geometry.size());
  });

  // Send geometry dofmap to device
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  auto x_dofs_d = static_cast<std::int32_t*>(
      cl::sycl::malloc_device(sizeof(std::int32_t) * x_dofmap.size(), queue));
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(x_dofs_d, x_dofmap.data(), sizeof(std::int32_t) * x_dofmap.size());
  });

  // Send RHS coefficients to device
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(L);
  auto coeff_d = static_cast<double*>(
      cl::sycl::malloc_device(sizeof(double) * coeffs.size(), queue));
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(coeff_d, coeffs.data(), sizeof(std::int32_t) * coeffs.size());
  });

  queue.wait();

  return {x_d, x_dofs_d, coeff_d, ndofs, ncells, ndofs_cell};
}

// Submit vector assembly kernels to queue
double* assemble_vector(cl::sycl::queue& queue, device_data_t& data)
{
  std::int32_t ndofs_ext = data.ndofs_cell * data.ncells;
  auto b = (double*)cl::sycl::malloc_device(sizeof(double) * ndofs_ext, queue);

  assemble_vector_ext(queue, b, data.x, data.xdofs, data.coeffs, data.ncells,
                      data.ndofs, data.ndofs_cell);

  return b;
}
} // namespace dolfinx_sycl::assemble