cmake_minimum_required(VERSION 3.4.3)

project(sycl_poisson)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(supported_implementations
  ComputeCpp
  hipSYCL
  LLVM
  LLVM-CUDA
)

list(FIND supported_implementations ${SYCL_IMPL} impl_idx)
if(NOT SYCL_IMPL OR impl_idx EQUAL -1)
  message(FATAL_ERROR "Please specify SYCL_IMPL (one of: ${supported_implementations})")
endif()

if(SYCL_IMPL STREQUAL "ComputeCpp")
  find_package(ComputeCpp REQUIRED)
elseif(SYCL_IMPL STREQUAL "hipSYCL")
  find_package(hipSYCL CONFIG REQUIRED)
elseif(SYCL_IMPL STREQUAL "LLVM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
  set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice --cuda-path=${CUDA_PATH}")
endif()

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG ${CMAKE_CXX_FLAGS} -Wall")
find_package(DOLFINX REQUIRED)
find_package(Ginkgo REQUIRED)

add_executable(sycl_poisson main.cpp poisson.c assemble_impl.cpp)
target_link_libraries(sycl_poisson PRIVATE Ginkgo::ginkgo)


if(SYCL_IMPL STREQUAL "LLVM")
    target_link_libraries(sycl_poisson PRIVATE dolfinx)
    target_compile_definitions(sycl_poisson PRIVATE __LLVM_SYCL__)
elseif(SYCL_IMPL STREQUAL "LLVM-CUDA")
    target_link_libraries(sycl_poisson PRIVATE dolfinx)
    target_compile_definitions(sycl_poisson PRIVATE __LLVM_SYCL__)
elseif(SYCL_IMPL STREQUAL "hipSYCL")
    target_link_libraries(sycl_poisson PRIVATE dolfinx)
    add_sycl_to_target(TARGET sycl_poisson SOURCES main.cpp assemble_impl.cpp)
else()
    target_link_libraries(sycl_poisson dolfinx)
    add_sycl_to_target(TARGET sycl_poisson SOURCES assemble_impl.cpp )
endif()
