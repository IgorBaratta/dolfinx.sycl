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
int main(int argc, char *argv[])
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
  const graph::AdjacencyList<std::int32_t> &v_dofmap = V->dofmap()->list();
  const graph::AdjacencyList<std::int32_t> dm_index_b = create_index_vec(v_dofmap);
  const graph::AdjacencyList<std::int32_t> dm_index_A = create_index_mat(v_dofmap, v_dofmap);

  auto f = std::make_shared<function::Function<double>>(V);
  f->interpolate([](auto &x) {
    auto dx = Eigen::square(x - 0.5);
    return 10.0 * Eigen::exp(-(dx.row(0) + dx.row(1)) / 0.02);
  });
  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_L, {V},
                                                  {{"f", f}, {}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_poisson_a, {V, V},
                                                  {}, {}, {});

  std::vector<cl::sycl::device> gpus;
  auto platforms = cl::sycl::platform::get_platforms();
  for (auto &p : platforms)
  {
    auto devices = p.get_devices();
    for (auto &device : devices)
      if (device.is_gpu())
        gpus.push_back(device);
  }

  cl::sycl::queue queue;

  int num_devices = gpus.size();
  if (num_devices >= mpi_size)
    queue = cl::sycl::queue(gpus[rank], exception_handler, {});
  else
    queue = cl::sycl::queue(cl::sycl::cpu_selector(), exception_handler, {});

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

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  // =========================== //
  // Send data to device

  // Get geometry data
  auto timer_start = std::chrono::system_clock::now();
  const auto &geometry = mesh->geometry().x();
  auto x_d = static_cast<double *>(
      cl::sycl::malloc_device(sizeof(double) * geometry.size(), queue));
  queue.submit([&](cl::sycl::handler &h) {
    h.memcpy(x_d, geometry.data(), sizeof(double) * geometry.size());
  });
  queue.wait();
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Geometry"] = (timer_end - timer_start);

  timer_start = std::chrono::system_clock::now();
  const dolfinx::graph::AdjacencyList<std::int32_t> &dofs = L->function_spaces()[0]->dofmap()->list();
  auto b_d = static_cast<double *>(
      cl::sycl::malloc_device(sizeof(double) * dofs.array().size(), queue));
  timer_end = std::chrono::system_clock::now();
  timings["1 - Vector"] = (timer_end - timer_start);

  const auto &x_dofmap = mesh->geometry().dofmap().array();
  timer_start = std::chrono::system_clock::now();
  auto x_dofs_d = static_cast<std::int32_t *>(
      cl::sycl::malloc_device(sizeof(std::int32_t) * x_dofmap.size(), queue));
  queue.submit([&](cl::sycl::handler &h) {
    h.memcpy(x_dofs_d, x_dofmap.data(), sizeof(std::int32_t) * x_dofmap.size());
  });
  queue.wait();
  timer_end = std::chrono::system_clock::now();
  timings["2 - Geom Dofmap"] = (timer_end - timer_start);

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      coeffs = dolfinx::fem::pack_coefficients(*L);

  timer_start = std::chrono::system_clock::now();
  auto coeff_d = static_cast<double *>(
      cl::sycl::malloc_device(sizeof(double) * coeffs.size(), queue));
  queue.submit([&](cl::sycl::handler &h) {
    h.memcpy(coeff_d, coeffs.data(), sizeof(std::int32_t) * coeffs.size());
  });
  queue.wait();
  timer_end = std::chrono::system_clock::now();
  timings["3 - Coefficients"] = (timer_end - timer_start);

  std::int32_t ncells = dofs.num_nodes();
  std::int32_t ndofs = V->dofmap()->index_map->size_local();
  std::int32_t nghost_dofs = V->dofmap()->index_map->num_ghosts();
  ndofs = ndofs + nghost_dofs;
  int nelem_dofs = dofs.num_links(0);

  timer_start = std::chrono::system_clock::now();
  assemble_rhs_usm(queue, b_d, x_d, x_dofs_d, coeff_d, ncells, ndofs, nelem_dofs);
  timer_end = std::chrono::system_clock::now();
  timings["4 - Assemble RHS"] = (timer_end - timer_start);

  if (rank == 0)
    std::cout << "\nTimings (" << mpi_size
              << ")\n----------------------------\n";
  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
      std::string pad(32 - q.first.size(), ' ');
      std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                << "\n";
    }
  }

  return 0;
}
