// Copyright (C) 2020 Igor A. Baratta
// SPDX-License-Identifier:    MIT

#pragma once

#include <chrono>
#include <map>
#include <mpi.h>

namespace dolfinx::experimental::sycl::timing
{

void print_timing_info(
    MPI_Comm mpi_comm,
    const std::map<std::string, std::chrono::duration<double>>& timings,
    std::string title = {}, int verbose_mode = 1);

} // namespace dolfinx::experimental::sycl::timing