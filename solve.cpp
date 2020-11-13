#include <ginkgo/ginkgo.hpp>

#include "solve.hpp"

void dolfinx_sycl::solve(double* A, double* b, double* x,
                         std::vector<std::int32_t> coo_rows,
                         std::vector<std::int32_t> coo_cols, int ndofs)
{
  // auto exec = gko::DpcppExecutor::create(0,
  // gko::ReferenceExecutor::create());
  auto exec = gko::OmpExecutor::create();

  // Create Vector
  auto b_view = gko::Array<double>::view(exec, ndofs, b);
  auto vec = gko::matrix::Dense<double>::create(exec, gko::dim<2>(ndofs, 1),
                                                b_view, 1);

  using mtx = gko::matrix::Coo<double, std::int32_t>;

  std::int32_t values_size = coo_rows.size();
  auto values = gko::Array<double>::view(exec, values_size, A);
  auto rows
      = gko::Array<std::int32_t>::view(exec, values_size, coo_rows.data());
  auto cols
      = gko::Array<std::int32_t>::view(exec, values_size, coo_cols.data());
  auto matrix = mtx::create(exec, gko::dim<2>(ndofs), values, rows, cols);

  auto x_view = gko::Array<double>::view(exec, ndofs, x);
  auto x_vec = gko::matrix::Dense<double>::create(exec, gko::dim<2>(ndofs, 1),
                                                  x_view, 1);

  const gko::remove_complex<double> reduction_factor = 1e-5;
  using cg = gko::solver::Cg<double>;

  auto solver_gen
      = cg::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(ndofs).on(exec),
                gko::stop::ResidualNormReduction<double>::build()
                    .with_reduction_factor(reduction_factor)
                    .on(exec))
            .on(exec);

  auto solver = solver_gen->generate(gko::give(matrix));

  solver->apply(gko::lend(vec), gko::lend(x_vec));
}