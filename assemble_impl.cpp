
#include "assemble_impl.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

// Need to include C file in same translation unit as lambda
#include "poisson.c"

void assemble_vector_impl(cl::sycl::queue& queue, double* b, double* x,
                          int* x_coor, double* coeff, int ncells, int ndofs,
                          int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = L_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];
      double c[ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int pos = i * nelem_dofs;
      tabulate_cell_L(&b[pos], &coeff[pos], c, cell_geom, nullptr, nullptr, 0);
    };

    cgh.parallel_for<class AssemblyKernelUSM_b>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}

// Second kernel to accumulate RHS for each dof
void accumulate_vector_impl(cl::sycl::queue& queue, double* b, double* b_ext,
                            int* offsets, int* indices, int ndofs)
{
  cl::sycl::range<1> range{(std::size_t)ndofs};
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);

      double val = 0.0;
      for (int j = offsets[i]; j < offsets[i + 1]; ++j)
        val += b_ext[indices[j]];

      b[i] = val;
    };

    cgh.parallel_for<class AccumulationKernel_b>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}

void assemble_matrix_impl(cl::sycl::queue& queue, double* A, double* x,
                          int* x_coor, double* coeff, int ncells, int ndofs,
                          int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    constexpr int ndofs_cell = a_num_dofs;

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];

      double c[ndofs_cell] = {0};
      double Ae[ndofs_cell * ndofs_cell] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      const int pos_c = i * nelem_dofs;
      tabulate_cell_a(Ae, &coeff[pos_c], c, cell_geom, nullptr, nullptr, 0);

      const int pos_A = i * nelem_dofs * nelem_dofs;
      for (int j = 0; j < nelem_dofs * nelem_dofs; j++)
        A[pos_A + j] = Ae[j];
    };

    cgh.parallel_for<class AssemblyKernelUSM_A>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}