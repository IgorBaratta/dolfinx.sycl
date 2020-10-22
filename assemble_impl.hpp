
#include <CL/sycl.hpp>

#pragma once

// Submit assembly kernels to queue
void assemble_rhs(cl::sycl::queue& queue,
                  cl::sycl::buffer<double, 1>& accum_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf, int nelem_dofs);

// Submit accumulation kernels to queue
void accumulate_rhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& ac_buf,
                    cl::sycl::buffer<double, 1>& global_vec_buf,
                    cl::sycl::buffer<int, 1>& offset_buf,
                    cl::sycl::buffer<int, 1>& index_buf);

// Submit assembly kernels to queue
void assemble_lhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& A_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf, int nelem_dofs);

// Submit accumulation kernels to queue
void accumulate_lhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& A_buf,
                    cl::sycl::buffer<double, 1>& global_mat_buf,
                    cl::sycl::buffer<int, 1>& index_buf,
                    cl::sycl::buffer<int, 1>& offset_buf);

// Submit assembly kernels to queue
void assemble_rhs_usm(cl::sycl::queue& queue, double* b, double* c,
                      int* cood_dm, double* coeff, int ncells, int ndofs,
                      int nelem_dofs);
