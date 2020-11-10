
#include "assemble_impl.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

// Need to include C file in same translation unit as lambda
#include "poisson.c"

void assemble_vector_ext(cl::sycl::queue &queue, double *b, double *x,
                         int *x_coor, double *coeff, int ncells, int ndofs,
                         int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler &cgh) {
    int gdim = 3;
    cl::sycl::range<1> range{std::size_t(ncells)};

    auto kernel = [=](cl::sycl::id<1> ID) {
      const int i = ID.get(0);
      double cell_geom[12];
      double c[32] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = x_coor[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      tabulate_cell_L(b + i * nelem_dofs, coeff + i * nelem_dofs, c, cell_geom,
                      nullptr, nullptr, 0);
    };

    cgh.parallel_for<class AssemblyKernelUSM_b>(range, kernel);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (cl::sycl::exception const &e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}