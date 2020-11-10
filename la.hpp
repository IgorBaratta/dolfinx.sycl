// Copyright (C) 2020 Igor A. Baratta and Chris Richardson
// SPDX-License-Identifier:    MIT

#include <CL/sycl.hpp>

#include <dolfinx.h>
#include <mpi.h>

using namespace dolfinx;

namespace dolfinx_sycl::la
{

struct csr_data_t
{
  double* data;
  std::int32_t* indptr;
  std::int32_t* indices;

  std::int32_t nrows;
  std::int32_t ncols;
};

//--------------------------------------------------------------------------
template <typename T>
graph::AdjacencyList<std::int32_t> sort_and_offset(std::vector<T> indices)
{
  // Get the permutation which takes the entries into order
  std::vector<int> perm(indices.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
    return (indices[a] < indices[b]);
  });

  // Compute offsets for each entry
  std::vector<int> offset = {0};
  T last = indices[perm[0]];
  for (std::size_t i = 0; i < perm.size(); ++i)
  {
    int idx = perm[i];
    const T& current = indices[idx];
    if (current != last)
      offset.push_back(i);
    last = current;
  }
  offset.push_back(perm.size());

  return graph::AdjacencyList<std::int32_t>(perm, offset);
}
//--------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
create_index_mat(const graph::AdjacencyList<std::int32_t>& dofmap0,
                 const graph::AdjacencyList<std::int32_t>& dofmap1)
{
  // CSR format for Matrix
  // Creates a layout and offsets to move data from the assembly order into the
  // dof order

  assert(dofmap0.num_nodes() == dofmap1.num_nodes());

  const int ncells = dofmap0.num_nodes();
  std::vector<std::array<int, 2>> indices;
  const int nelem_dofs0 = dofmap0.num_links(0);
  const int nelem_dofs1 = dofmap1.num_links(0);
  indices.reserve(ncells * nelem_dofs0 * nelem_dofs1);

  // Iterate through all indices in assembly order
  for (int i = 0; i < ncells; ++i)
  {
    auto dofs0 = dofmap0.links(i);
    auto dofs1 = dofmap1.links(i);
    for (int j = 0; j < nelem_dofs0; ++j)
    {
      for (int k = 0; k < nelem_dofs1; ++k)
        indices.push_back({dofs0[j], dofs1[k]});
    }
  }

  return sort_and_offset(indices);
}
//--------------------------------------------------------------------------
graph::AdjacencyList<std::int32_t>
create_index_vec(const graph::AdjacencyList<std::int32_t>& dofmap)
{
  int ncells = dofmap.num_nodes();
  int nelem_dofs = dofmap.links(0).size();
  std::vector<int> indices;
  indices.reserve(ncells * nelem_dofs);

  // Stack up indices in assembly order
  for (int i = 0; i < ncells; ++i)
  {
    const auto dofs = dofmap.links(i);
    for (int j = 0; j < nelem_dofs; ++j)
      indices.push_back(dofs[j]);
  }

  // Get the permutation that sorts them into dof order
  return sort_and_offset(indices);
}
} // namespace dolfinx_sycl::la