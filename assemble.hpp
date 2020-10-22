
#include <CL/sycl.hpp>
#include <dolfinx.h>

#include <Eigen/Sparse>

#include "assemble_impl.hpp"
#include "poisson.h"

/// Assemble linear form into an SYCL buffer
Eigen::VectorXd
assemble_vector(cl::sycl::queue& queue, const dolfinx::fem::Form<double>& L,
                const dolfinx::graph::AdjacencyList<std::int32_t>& dm_index_b)
{
  auto mesh = L.mesh();
  auto dofmap = L.function_spaces()[0]->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();

  std::int32_t ndofs = dofmap->index_map->size_local();
  std::int32_t nghost_dofs = dofmap->index_map->num_ghosts();
  ndofs = ndofs + nghost_dofs;
  int nelem_dofs = dofs.num_links(0);

  Eigen::VectorXd global_vector(ndofs);

  // Device memory to accumulate assembly entries before summing
  cl::sycl::buffer<double, 1> b_buf(
      cl::sycl::range<1>{(std::size_t)dofs.array().size()});

  // Get geometry buffer
  const auto& geometry = mesh->geometry().x();
  cl::sycl::buffer<double, 2> geom_buf(geometry.data(),
                                       {(std::size_t)geometry.rows(), 3});

  // Get geometry dofmap buffer
  const auto& x_dofmap = mesh->geometry().dofmap();
  cl::sycl::buffer<int, 2> coord_dm_buf(
      x_dofmap.array().data(),
      {(std::size_t)x_dofmap.num_nodes(), (std::size_t)x_dofmap.num_links(0)});

  // Get coefficient buffer
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeff = dolfinx::fem::pack_coefficients(L);
  cl::sycl::buffer<double, 2> coeff_buf(
      coeff.data(), {(std::size_t)coeff.rows(), (std::size_t)coeff.cols()});

  assemble_rhs(queue, b_buf, geom_buf, coord_dm_buf, coeff_buf, nelem_dofs);

  cl::sycl::buffer<int, 1> off_buf(dm_index_b.offsets().data(),
                                   dm_index_b.offsets().size());

  cl::sycl::buffer<int, 1> index_buf(dm_index_b.array().data(),
                                     dm_index_b.array().size());

  cl::sycl::buffer<double, 1> gv_buf(global_vector.data(),
                                     global_vector.size());

  accumulate_rhs(queue, b_buf, gv_buf, index_buf, off_buf);

  return global_vector;
}

/// Assemble bilinear form into an SYCL buffer
Eigen::VectorXd
assemble_matrix(cl::sycl::queue& queue, const dolfinx::fem::Form<double>& a,
                const dolfinx::graph::AdjacencyList<std::int32_t>& dm_index_A)
{
  std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
  std::shared_ptr<const dolfinx::fem::DofMap> dofmap
      = a.function_spaces()[0]->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();

  int tdim = mesh->topology().dim();
  [[maybe_unused]] std::int32_t ndofs = dofmap->index_map->size_local();
  std::int32_t nelem = mesh->topology().index_map(tdim)->size_local();
  int nelem_dofs = dofs.num_links(0);

  // Get geometry buffer
  const auto& geometry = mesh->geometry().x();
  cl::sycl::buffer<double, 2> geom_buf(geometry.data(),
                                       {(std::size_t)geometry.rows(), 3});

  // Get geometry dofmap buffer
  const auto& x_dofmap = mesh->geometry().dofmap();
  cl::sycl::buffer<int, 2> coord_dm_buf(
      x_dofmap.array().data(),
      {(std::size_t)x_dofmap.num_nodes(), (std::size_t)x_dofmap.num_links(0)});

  // Prepare coefficients
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeffs
      = dolfinx::fem::pack_coefficients(a);

  // FIXME: allow non-empty coefficients and functions
  cl::sycl::buffer<double, 2> coeff_buf(
      {(std::size_t)nelem, (std::size_t)nelem_dofs});

  cl::sycl::buffer<double, 1> A_buf(
      cl::sycl::range<1>{(std::size_t)nelem * nelem_dofs * nelem_dofs});

  assemble_lhs(queue, A_buf, geom_buf, coord_dm_buf, coeff_buf, nelem_dofs);

  Eigen::VectorXd global_matrix(dm_index_A.offsets().size() - 1);
  cl::sycl::buffer<double, 1> gm_buf(global_matrix.data(),
                                     global_matrix.size());

  // Second kernel to accumulate RHS for each dof
  cl::sycl::buffer<int, 1> off_buf(dm_index_A.offsets().data(),
                                   dm_index_A.offsets().size());

  cl::sycl::buffer<int, 1> index_buf(dm_index_A.array().data(),
                                     dm_index_A.array().size());

  accumulate_lhs(queue, A_buf, gm_buf, index_buf, off_buf);

  return global_matrix;
}

/// Assemble linear form into an SYCL buffer
void assemble_vector_usm(
    cl::sycl::queue& queue, const dolfinx::fem::Form<double>& L,
    const dolfinx::graph::AdjacencyList<std::int32_t>& dm_index_b)
{
  auto mesh = L.mesh();
  auto dofmap = L.function_spaces().at(0)->dofmap();
  const dolfinx::graph::AdjacencyList<std::int32_t>& dofs = dofmap->list();

  std::int32_t ncells = dofs.num_nodes();
  std::int32_t ndofs = dofmap->index_map->size_local();
  std::int32_t nghost_dofs = dofmap->index_map->num_ghosts();
  ndofs = ndofs + nghost_dofs;
  int nelem_dofs = dofs.num_links(0);

  std::int32_t ndofs_ext = dofs.array().size();

  auto b = static_cast<double*>(
      cl::sycl::malloc_shared(sizeof(double) * ndofs_ext, queue));

  // Get geometry data
  const auto& geometry = mesh->geometry().x();
  auto x = static_cast<double*>(
      cl::sycl::malloc_shared(sizeof(double) * geometry.size(), queue));
  std::memcpy(x, geometry.data(), sizeof(double) * geometry.size());

  // Get geometry dofmap buffer
  const auto& x_dofmap = mesh->geometry().dofmap().array();
  auto coord_dm = static_cast<std::int32_t*>(
      cl::sycl::malloc_shared(sizeof(std::int32_t) * x_dofmap.size(), queue));
  std::memcpy(coord_dm, x_dofmap.data(),
              sizeof(std::int32_t) * x_dofmap.size());

  // Get coefficient buffer
  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(L);
  auto coeff = static_cast<double*>(
      cl::sycl::malloc_shared(sizeof(double) * coeffs.size(), queue));
  std::memcpy(coeff, coeffs.data(), sizeof(double) * coeffs.size());

  assemble_rhs_usm(queue, b, x, coord_dm, coeff, ncells, ndofs, nelem_dofs);

  double cc = 0;
  for (int i = 0; i < ndofs_ext; i++)
  {
    cc += b[i];
  }
  std::cout << cc;
}