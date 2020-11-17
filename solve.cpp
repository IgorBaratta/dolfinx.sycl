// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include "solve.hpp"
#include <ginkgo/ginkgo.hpp>
#include <map>

double dolfinx::experimental::sycl::solve::ginkgo(
    double* A, std::int32_t* indptr, std::int32_t* indices, std::int32_t nrows,
    std::int32_t nnz, double* b, double* x, std::string executor)
{
  using mtx = gko::matrix::Csr<double, std::int32_t>;
  using cg = gko::solver::Cg<double>;

  std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
      exec_map{{"omp", [] { return gko::OmpExecutor::create(); }},
               {"cuda",
                [] {
                  return gko::CudaExecutor::create(
                      0, gko::OmpExecutor::create(), true);
                }},
               {"hip",
                [] {
                  return gko::HipExecutor::create(0, gko::OmpExecutor::create(),
                                                  true);
                }},
               {"dpcpp",
                [] {
                  return gko::DpcppExecutor::create(0,
                                                    gko::OmpExecutor::create());
                }},
               {"reference", [] { return gko::ReferenceExecutor::create(); }}};

  auto exec = exec_map.at(executor)(); // throws if not valid

  // Create Input Vector
  auto b_view = gko::Array<double>::view(exec, nrows, b);
  auto in = gko::matrix::Dense<double>::create(exec, gko::dim<2>(nrows, 1),
                                               b_view, 1);

  // Create Output Vector
  auto x_view = gko::Array<double>::view(exec, nrows, x);
  auto out = gko::matrix::Dense<double>::create(exec, gko::dim<2>(nrows, 1),
                                                x_view, 1);

  // Create Matrix
  auto data_v = gko::Array<double>::view(exec, nnz, A);
  auto indptr_v = gko::Array<std::int32_t>::view(exec, nrows + 1, indptr);
  auto indices_v = gko::Array<std::int32_t>::view(exec, nnz, indices);
  auto dim = gko::dim<2>(nrows);
  auto matrix = mtx::create(exec, dim, data_v, indices_v, indptr_v);

  const double reduction_factor = 1e-5;

  // Generate solver
  auto solver_gen
      = cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(100u).on(exec),
                gko::stop::ResidualNormReduction<double>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .on(exec);

  auto solver = solver_gen->generate(gko::give(matrix));
  solver->apply(gko::lend(in), gko::lend(out));

  auto res = gko::initialize<gko::matrix::Dense<double>>({0.0}, exec);
  out->compute_norm2(gko::lend(res));

  return res->get_values()[0];
}