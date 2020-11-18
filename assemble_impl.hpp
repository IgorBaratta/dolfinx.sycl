// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#pragma once

// Submit assembly kernels to queue
void assemble_vector_impl(cl::sycl::queue& queue, double* b, double* x,
                          int* x_dof, double* coeff, int ncells, int ndofs,
                          int nelem_dofs);

// Submit assembly kernels to queue
void assemble_matrix_impl(cl::sycl::queue& queue, double* A, double* x,
                          int* x_dof, double* coeff, int ncells, int ndofs,
                          int nelem_dofs);


// Submit accumulation kernels to queue
void accumulate_impl(cl::sycl::queue& queue, double* b, double* b_ext,
                            int* offset, int* indices, int ndofs);