#include <CL/sycl.hpp>

#include <Eigen/Dense>
#include <dolfinx.h>
#include <dolfinx/fem/petsc.h>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "assemble.hpp"
#include "poisson.h"
#include "utils.hpp"

using namespace dolfinx;

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_petsc(argc, argv);

  int rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  std::size_t nx = 32;
  if (argc == 2)
    nx = std::stoi(argv[1]);

  auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 1.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::BoxMesh::create(
      MPI_COMM_WORLD, pt, {{nx, nx, nx}}, cmap, mesh::GhostMode::none));

  mesh->topology_mutable().create_entity_permutations();

  auto V = fem::create_functionspace(create_functionspace_form_poisson_a, "u",
                                     mesh);

  // Create dofmap permutation for insertion into array
  const graph::AdjacencyList<std::int32_t>& v_dofmap = V->dofmap()->list();
  const graph::AdjacencyList<std::int32_t> dm_index_b
      = create_index_vec(v_dofmap);
  const graph::AdjacencyList<std::int32_t> dm_index_A
      = create_index_mat(v_dofmap, v_dofmap);

  auto f = std::make_shared<function::Function<double>>(V);
  f->interpolate([](auto& x) {
    auto dx = Eigen::square(x - 0.5);
    return 10.0 * Eigen::exp(-(dx.row(0) + dx.row(1)) / 0.02);
  });
  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_L, {V},
                                                  {{"f", f}, {}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_a, {V, V},
                                                  {}, {}, {});

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeff = fem::pack_coefficients(*L);

  // Select device to offload computation, default is implementation dependent
  cl::sycl::default_selector device_selector;
  cl::sycl::queue queue(device_selector, exception_handler, {});

  // Print some information
  if (rank == 0)
  {
    print_device_info(queue.get_device());
    int tdim = mesh->topology().dim();
    std::cout << "\nNumber of cells: "
              << mesh->topology().index_map(tdim)->size_global() << std::endl;
    std::cout << "Number of dofs: " << V->dofmap()->index_map->size_global()
              << std::endl;
  }

  // SYCL Code
  //--------------------------
  {
    auto timer_start = std::chrono::system_clock::now();
    Eigen::VectorXd vec = assemble_vector(queue, *L, dm_index_b);
    auto timer_end = std::chrono::system_clock::now();
    std::chrono::duration<double> dt_L = (timer_end - timer_start);

    timer_start = std::chrono::system_clock::now();
    Eigen::VectorXd data = assemble_matrix(queue, *a, dm_index_A);
    timer_end = std::chrono::system_clock::now();
    std::chrono::duration<double> dt_a = (timer_end - timer_start);

    if (rank == 0)
    {
      std::cout << "\nSYCL" << std::endl;

      std::cout << "Assemble Vector(s) " << dt_L.count() << "\n";
      std::cout << "Assemble Matrix (s) " << dt_a.count() << "\n";
      std::cout << "Vector norm " << vec.norm() << "\n";
    }
  }

  // Comparison CPU code below
  //--------------------------
  auto timer_start = std::chrono::system_clock::now();

  double norm;
  la::PETScVector u(*L->function_space(0)->dofmap()->index_map);
  VecSet(u.vec(), 0);
  dolfinx::fem::assemble_vector_petsc(u.vec(), *L);
  VecGhostUpdateBegin(u.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(u.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecNorm(u.vec(), NORM_2, &norm);
  auto timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt_L = (timer_end - timer_start);

  la::PETScMatrix A = fem::create_matrix(*a);
  MatZeroEntries(A.mat());
  timer_start = std::chrono::system_clock::now();
  fem::assemble_matrix(la::PETScMatrix::add_fn(A.mat()), *a, {});
  timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt_a = (timer_end - timer_start);

  if (rank == 0)
  {
    std::cout << "\nPETSc" << std::endl;
    std::cout << "Assemble Vector(s) " << dt_L.count() << "\n";
    std::cout << "Assemble Matrix(s) " << dt_a.count() << "\n";
    std::cout << "Vector norm " << norm << "\n";
  }

  return 0;
}
