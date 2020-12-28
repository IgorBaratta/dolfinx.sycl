FROM dolfinx/dolfinx:latest
LABEL description="DOLFIN-X in real mode with SYCL"
ARG PETSC_ARCH=linux-gnu-real-32

WORKDIR /tmp

RUN apt update && apt-get -y install software-properties-common &&\
    apt-get install -y libboost-fiber-dev

RUN apt install -y gcc-7 g++-7 &&\
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100 &&\
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
   
RUN apt update && export DEBIAN_FRONTEND=noninteractive &&\
    add-apt-repository ppa:graphics-drivers && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    bash -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list' && \
    apt update && apt install -yq cuda-10-0 cuda-cublas-10-0

ARG PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
ARG LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

RUN apt update && apt-get install -y libc++-dev && \
    apt-get install -y clang-format clang-tidy clang-tools clang clangd libc++-dev && \
    apt-get install -y libc++1 libc++abi-dev libc++abi1 libclang-dev libclang1 liblldb-dev libllvm-ocaml-dev &&\
    apt-get install -y libomp-dev libomp5 lld lldb llvm-dev llvm-runtime llvm python-clang

RUN git clone --recurse-submodules https://github.com/illuhad/hipSYCL && \
    cd hipSYCL && mkdir build &&\ 
    cd build && cmake .. &&\
    make -j4 && make install

RUN git clone https://github.com/ginkgo-project/ginkgo.git &&\
    cd ginkgo && mkdir build && cd build &&\
    export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}} &&\
    export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} &&\
    export CUDACXX=/usr/local/cuda-10.0/bin/nvcc &&\ 
    export CXX=g++-7 && export CC=gcc-7 && \    
    cmake -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_TESTS=OFF -DGINKGO_BUILD_BENCHMARKS=OFF \
    -DGINKGO_BUILD_REFERENCE=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local/ginkgo .. &&\
    make -j3 && make install

ARG GINKGO_DIR=/usr/local/ginkgo

RUN git clone https://github.com/NVIDIA/AMGX.git &&\
    cd AMGX && mkdir build && cd build &&\
    cmake ../ && make -j3 all

