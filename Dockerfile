# Build stage with Spack pre-installed and ready to be used
FROM spack/ubuntu-noble:0.23.1

RUN apt update && apt upgrade -y && apt install llvm-18-dev clang-18 libc++-18-dev libclang-18-dev libomp-18-dev -y
RUN apt install -y git nano python3 cmake mpich

RUN spack compiler find
RUN spack compiler remove clang
RUN spack compiler find /usr/lib/llvm-18

RUN spack external find --path /usr/lib/llvm-18 llvm
RUN spack external find python
RUN spack external find cmake
RUN spack external find mpich
