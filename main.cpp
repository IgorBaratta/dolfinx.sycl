// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <dolfinx.h>
#include <iostream>
#include <math.h>
#include <numeric>

#include "dolfinx_sycl.hpp"
#include "poisson.h"
#include "solve.hpp"

using namespace dolfinx;
using namespace dolfinx::experimental::sycl;

int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  MPI_Comm mpi_comm{MPI_COMM_WORLD};

  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  std::size_t nx = 20;
  if (argc == 2)
    nx = std::stoi(argv[1]);

  auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
  std::array<Eigen::Vector3d, 2> pts{Eigen::Vector3d(-1, -1, -1),
                                     Eigen::Vector3d(1.0, 1.0, 1.0)};

  auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
      mpi_comm, pts, {{nx, nx, nx}}, cmap, mesh::GhostMode::none));

  mesh->topology_mutable().create_entity_permutations();
  auto V = fem::create_functionspace(create_functionspace_form_poisson_a, "u",
                                     mesh);

  auto f = std::make_shared<function::Function<double>>(V);
  f->interpolate([](auto& x) {
    return (12 * M_PI * M_PI + 1) * Eigen::cos(2 * M_PI * x.row(0))
           * Eigen::cos(2 * M_PI * x.row(1)) * Eigen::cos(2 * M_PI * x.row(2));
  });

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_L, {V},
                                                  {{"f", f}, {}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_a, {V, V},
                                                  {}, {}, {});

  auto queue = utils::select_queue(mpi_comm);

  int verb_mode = 2;
  if (verb_mode)
  {
    utils::print_device_info(queue.get_device());
    utils::print_function_space_info(V);
  }

  // Send form data to device (Geometry, Dofmap, Coefficients)
  auto form_data = memory::send_form_data(mpi_comm, queue, *L, *a, verb_mode);

  // Assemble vector on device
  double* b = assemble::assemble_vector(mpi_comm, queue, form_data, verb_mode);

  // Assemble matrix on device
  auto mat = assemble::assemble_matrix(mpi_comm, queue, form_data, verb_mode);

  double* x = cl::sycl::malloc_device<double>(form_data.ndofs, queue);
  queue.fill<double>(x, 0., form_data.ndofs);

  std::int32_t nnz; //Todo: Store nnz 
  queue.memcpy(&nnz, &mat.indptr[mat.nrows], sizeof(std::int32_t)).wait();
  double norm = solve::ginkgo(mat.data, mat.indptr, mat.indices, mat.nrows, nnz,
                              b, x, "omp");

  auto vec = f->vector();
  double ex_norm = 0;
  VecNorm(vec, NORM_2, &ex_norm);

  std::cout << "\nComputed norm " << norm << "\n";
  std::cout << "Reference norm " << ex_norm / (12 * M_PI * M_PI + 1) << "\n";

  return 0;
}
