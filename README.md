# Dolfinx.SYCL
Simple code to assemble the Poisson equation on manycore architectures using Dolfinx and SYCL.

## Requirements:
  - DOLFINX
  - A SYCL implementation

### Supported SYCL Implementations

Supported (tested) SYCL implementation:
- hipSYCL
- LLVM 
- LLVM-CUDA
- ComputeCpp


## Building

### Using the hipSYCL implementation
Building for CPUs:
```bash
ffcx --sycl_defines=True poisson.ufl

mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DCMAKE_PREFIX_PATH=${Ginkgo_DIR} ..
make -j8
```

Building with CUDA and Nvidia Tesla P100 GPU accelerator:
```bash
ffcx --sycl_defines=True poisson.ufl

export HIPSYCL_PLATFORM=cuda
export HIPSYCL_GPU_ARCH=sm_60

mkdir build
cd build
cmake -DSYCL_IMPL=hipSYCL -DCMAKE_PREFIX_PATH=${Ginkgo_DIR}..
make -j8
```
** Runnning Dolfinx.sycl with hipsycl + CUDA requires eigen@master.

### Using the ComputeCPP implementation
```bash
ffcx --sycl_defines=True poisson.ufl

export ComputeCpp_DIR=/path/to/computecpp
export OpenCL_INCLUDE_DIR=path/to/opencl/include
export OpenCL_LIBRARY=/path/to/libOpenCL.so

mkdir build
cd build
cmake -DSYCL_IMPL=ComputeCpp -DComputeCpp_DIR=$ComputeCpp_DIR -DOpenCL_INCLUDE_DIR=$OpenCL_INCLUDE_DIR -DOpenCL_LIBRARY=$OpenCL_LIBRARY ..
make -j8
```
If cmake does not identify the correct version of gcc libraries (eg: libstd++), add
`-DCOMPUTECPP_USER_FLAGS="--gcc-toolchain=gcc/toolchain/dir"` when invoking cmake.


## Using Intel SYCL
```bash
export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
export CXX=clang++
export CC=gcc

mkdir build
cd build
cmake -DSYCL_IMPL=LLVM -DCMAKE_PREFIX_PATH=${Ginkgo_DIR} ..
make -j8
```

Using Intel SYCL with CUDA:
```bash

export PATH=$DPCPP_HOME/llvm/build/bin:$PATH
export LD_LIBRARY_PATH=$DPCPP_HOME/llvm/build/lib:$LD_LIBRARY_PATH
export CXX=clang++
export CC=gcc

mkdir build
cd build
cmake -DSYCL_IMPL=LLVM-CUDA -DCUDA_PATH=${CUDA_PATH} -DCMAKE_PREFIX_PATH=${Ginkgo_DIR} ..
make -j8
```


## Runinng
```bash
./sycl_poisson {Ncells}
```

{Ncells} - Number of cells in each direction ($`N_x`$, $`N_y`$, $`N_z`$), default is 50. 
Totalizing 750000 cells ($`6 \times N_x \times N_y \times N_z`$).

## Limitations
Too many to mention ...
- Assemble on cells only
