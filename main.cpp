#include <CL/sycl.hpp>

#include <Eigen/Dense>
#include <dolfinx.h>
#include <ginkgo/ginkgo.hpp>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <numeric>

#include "dolfinx_sycl.hpp"
#include "poisson.h"

using namespace dolfinx;

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
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

  auto queue = dolfinx_sycl::utils::select_queue(mpi_comm);

  dolfinx_sycl::utils::print_device_info(queue.get_device());
  dolfinx_sycl::utils::print_function_space_info(V);

  // Keep list of timings
  std::map<std::string, std::chrono::duration<double>> timings;

  // Create dofmap permutation for insertion into array
  // TODO: Create SYCL kernel.
  auto timer_start = std::chrono::system_clock::now();
  const graph::AdjacencyList<std::int32_t>& v_dofmap = V->dofmap()->list();
  const graph::AdjacencyList<std::int32_t> dm_index_vec
      = dolfinx_sycl::la::create_index_vec(v_dofmap);
  // const graph::AdjacencyList<std::int32_t> dm_index_A
  //     = dolfinx_sycl::la::create_index_mat(v_dofmap, v_dofmap);
  auto coo_pattern = dolfinx_sycl::la::coo_pattern(v_dofmap, v_dofmap);
  auto timer_end = std::chrono::system_clock::now();
  timings["0 - Create Permutation"] = (timer_end - timer_start);

  // Send data to device
  timer_start = std::chrono::system_clock::now();
  dolfinx_sycl::assemble::device_data_t data
      = dolfinx_sycl::assemble::send_data_to_device(queue, *L, *a);
  timer_end = std::chrono::system_clock::now();
  timings["1 - Transfer Data"] = (timer_end - timer_start);

  // Assemble Vector
  // Cells-wise contribution
  timer_start = std::chrono::system_clock::now();
  double* b_ext = dolfinx_sycl::assemble::assemble_vector(queue, data);
  timer_end = std::chrono::system_clock::now();
  timings["2 - Assemble Vector"] = (timer_end - timer_start);
  // Accumulate
  timer_start = std::chrono::system_clock::now();
  double* b = dolfinx_sycl::assemble::accumulate_vector(queue, b_ext, data,
                                                        dm_index_vec);

  cl::sycl::free(b_ext, queue);

  timer_end = std::chrono::system_clock::now();
  timings["3 - Vector Accumulate"] = (timer_end - timer_start);

  // Assemble Matrix
  // Cells-wise contributions
  timer_start = std::chrono::system_clock::now();
  double* A = dolfinx_sycl::assemble::assemble_matrix(queue, data);
  timer_end = std::chrono::system_clock::now();
  timings["4 - Assemble Matrix"] = (timer_end - timer_start);

  Eigen::VectorXd b_host(data.ndofs);
  queue.submit([&](cl::sycl::handler& h) {
    h.memcpy(b_host.data(), b, sizeof(double) * b_host.size());
  });
  queue.wait_and_throw();

  // auto exec = gko::DpcppExecutor::create(0,
  // gko::ReferenceExecutor::create());
  auto exec = gko::OmpExecutor::create();

  // Create Vector
  auto b_view = gko::Array<double>::view(exec, data.ndofs, b);
  auto vec = gko::matrix::Dense<double>::create(
      exec, gko::dim<2>(data.ndofs, 1), b_view, 1);

  using mtx = gko::matrix::Coo<double, std::int32_t>;

  std::int32_t values_size = data.ndofs_cell * data.ndofs_cell * data.ncells;
  auto values = gko::Array<double>::view(exec, values_size, A);
  auto rows = gko::Array<std::int32_t>::view(exec, values_size,
                                             coo_pattern.rows.data());
  auto cols = gko::Array<std::int32_t>::view(exec, values_size,
                                             coo_pattern.cols.data());
  auto matrix = mtx::create(exec, gko::dim<2>(data.ndofs), values, rows, cols);

  auto x = cl::sycl::malloc_device<double>(data.ndofs, queue);
  auto x_view = gko::Array<double>::view(exec, data.ndofs, x);
  auto x_vec = gko::matrix::Dense<double>::create(
      exec, gko::dim<2>(data.ndofs, 1), x_view, 1);

  const gko::remove_complex<double> reduction_factor = 1e-5;
  using cg = gko::solver::Cg<double>;
  // using bj = gko::preconditioner::Jacobi<double, std::int32_t>;

  auto solver_gen
      = cg::build()
            .with_criteria(gko::stop::Iteration::build()
                               .with_max_iters(data.ndofs)
                               .on(exec),
                           gko::stop::ResidualNormReduction<double>::build()
                               .with_reduction_factor(reduction_factor)
                               .on(exec))
            .on(exec);

  auto solver = solver_gen->generate(gko::give(matrix));

  timer_start = std::chrono::system_clock::now();
  solver->apply(gko::lend(vec), gko::lend(x_vec));
  timer_end = std::chrono::system_clock::now();
  timings["5 - Solve System"] = (timer_end - timer_start);

  auto norm = gko::initialize<gko::matrix::Dense<double>>({0.0}, exec);
  x_vec->compute_norm2(gko::lend(norm));
  std::cout << "\nVector norm " << b_host.norm() << " " << norm->get_values()[0]
            << "\n";

  dolfinx_sycl::utils::print_timing_info(mpi_comm, timings);

  // std::cout << solver->get_size();

  cl::sycl::free(A, queue);
  cl::sycl::free(b, queue);
  return 0;
}
