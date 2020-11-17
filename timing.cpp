// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#include <iostream>

#include "timing.hpp"

void dolfinx::experimental::sycl::timing::print_timing_info(
    MPI_Comm mpi_comm,
    const std::map<std::string, std::chrono::duration<double>>& timings,
    std::string title, int verbose_mode)
{
  if (verbose_mode == 0)
    return;

  int mpi_size, mpi_rank;
  MPI_Comm_size(mpi_comm, &mpi_size);
  MPI_Comm_rank(mpi_comm, &mpi_rank);

  if (mpi_rank == 0)
    std::cout << "\n----------------------------\n"
              << title << " (" << mpi_size
              << ")\n----------------------------\n";
  for (auto q : timings)
  {
    double q_local = q.second.count(), q_max, q_min;
    MPI_Reduce(&q_local, &q_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&q_local, &q_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0)
    {
      std::string pad(48 - q.first.size(), ' ');
      if (verbose_mode == 1)
      {
        if (q.first == "Total")
          std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                    << "\n";
      }
      else
        std::cout << "[" << q.first << "]" << pad << q_min << '\t' << q_max
                  << "\n";
    }
  }
}