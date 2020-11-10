
#include <CL/sycl.hpp>

#pragma once

// Submit assembly kernels to queue
void assemble_vector_ext(cl::sycl::queue &queue, double *b, double *x,
                         int *x_dof, double *coeff, int ncells, int ndofs,
                         int nelem_dofs);
