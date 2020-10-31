
#include "assemble_impl.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>

// Need to include C file in same translation unit as lambda
#include "poisson.c"

class AssemblyKernel_b;

void assemble_rhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& b_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf, int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_cdm
        = coord_dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_coeff = coeff_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_b = b_buf.get_access<cl::sycl::access::mode::write>(cgh);
    cl::sycl::range<2> coord_dims = coord_dm_buf.get_range();
    cl::sycl::range<1> nelem_sycl{coord_dims[0]};
    int ncoeff = coeff_buf.get_range()[1];
    int gdim = 3;

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];

      double cell_geom[12];
      double w[32] = {0};
      double b[32] = {0};
      double c[32] = {0};
      for (int j = 0; j < ncoeff; ++j)
        w[j] = access_coeff[i][j];

      // Pull out points for this cell
      for (std::size_t j = 0; j < coord_dims[1]; ++j)
      {
        const std::size_t dmi = access_cdm[i][j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = access_geom[dmi][k];
      }

      // Get local values
      tabulate_cell_L(b, w, c, cell_geom, nullptr, nullptr, 0);

      // Insert result into array range corresponding to each dof
      for (int j = 0; j < nelem_dofs; ++j)
        access_b[i * nelem_dofs + j] = b[j];
    };
    cgh.parallel_for<AssemblyKernel_b>(nelem_sycl, kern);
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

class AccumulationKernel_b;

void accumulate_rhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& b_buf,
                    cl::sycl::buffer<double, 1>& global_vec_buf,
                    cl::sycl::buffer<int, 1>& index_buf,
                    cl::sycl::buffer<int, 1>& offset_buf)
{
  // Second kernel to accumulate RHS for each dof

  cl::sycl::range<1> ndofs_sycl = offset_buf.get_range() - 1;

  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto access_gv
        = global_vec_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto access_b = b_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_off = offset_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_idx = index_buf.get_access<cl::sycl::access::mode::read>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];
      access_gv[i] = 0.0;
      double val = 0.0;
      for (int j = access_off[i]; j < access_off[i + 1]; ++j)
        val += access_b[access_idx[j]];
      access_gv[i] = val;
    };
    cgh.parallel_for<AccumulationKernel_b>(ndofs_sycl, kern);
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

class AssemblyKernel_A;

void assemble_lhs(cl::sycl::queue& queue,
                  cl::sycl::buffer<double, 1>& accum_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf, int nelem_dofs)
{

  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_cdm
        = coord_dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_coeff = coeff_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_ac = accum_buf.get_access<cl::sycl::access::mode::write>(cgh);
    cl::sycl::range<2> coord_dims = coord_dm_buf.get_range();
    cl::sycl::range<1> nelem_sycl{coord_dims[0]};
    int ncoeff = coeff_buf.get_range()[1];
    int gdim = 3;

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];

      // FIXME: Dynamic allocation (Use )
      double cell_geom[12];
      double w[20] = {0};
      double A[400] = {0};
      double c[20] = {0};
      for (int j = 0; j < ncoeff; ++j)
        w[j] = access_coeff[i][j];

      // Pull out points for this cell
      for (std::size_t j = 0; j < coord_dims[1]; ++j)
      {
        const std::size_t dmi = access_cdm[i][j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = access_geom[dmi][k];
      }

      // Get local values
      tabulate_cell_a(A, w, c, cell_geom, nullptr, nullptr, 0);

      // Insert result into array range corresponding to each dof
      for (int j = 0; j < nelem_dofs * nelem_dofs; ++j)
        access_ac[i * nelem_dofs * nelem_dofs + j] = A[j];
    };
    cgh.parallel_for<AssemblyKernel_A>(nelem_sycl, kern);
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

class AccumulationKernel_A;

void accumulate_lhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& A_buf,
                    cl::sycl::buffer<double, 1>& global_mat_buf,
                    cl::sycl::buffer<int, 1>& index_buf,
                    cl::sycl::buffer<int, 1>& offset_buf)
{
  // Second kernel to accumulate RHS for each dof

  cl::sycl::range<1> ndofs_sycl = offset_buf.get_range() - 1;

  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    auto access_gmat
        = global_mat_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto access_A = A_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_off = offset_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_idx = index_buf.get_access<cl::sycl::access::mode::read>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];
      double val = 0.0;
      for (int j = access_off[i]; j < access_off[i + 1]; ++j)
        val += access_A[access_idx[j]];
      access_gmat[i] = val;
    };
    cgh.parallel_for<AccumulationKernel_A>(ndofs_sycl, kern);
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

class AssemblyKernelUSM_b;

void assemble_rhs_usm(cl::sycl::queue& queue, double* b, double* x,
                      int* cood_dm, double* coeff, int ncells, int ndofs,
                      int nelem_dofs)
{
  cl::sycl::event event = queue.submit([&](cl::sycl::handler& cgh) {
    int gdim = 3;
    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];

      double cell_geom[12];
      double c[32] = {0};

      // Pull out points for this cell
      for (std::size_t j = 0; j < 4; ++j)
      {
        const std::size_t dmi = cood_dm[i * 4 + j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = x[dmi * gdim + k];
      }

      // Get local values
      tabulate_cell_L(b + i * nelem_dofs, coeff + i * nelem_dofs, c, cell_geom,
                      nullptr, nullptr, 0);
    };
    cgh.parallel_for<AssemblyKernelUSM_b>(cl::sycl::range<1>{std::size_t(ncells)},
                                       kern);
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