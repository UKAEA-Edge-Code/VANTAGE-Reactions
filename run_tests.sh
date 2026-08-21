#!/bin/bash

: "${MAX_JOBS:=$(nproc)}"

source /opt/spack/share/spack/setup-env.sh
cd /root/Reactions
spack env activate -p -d environments/spack_default
spack install -j"$MAX_JOBS"
spack load vantagereactions
OMP_NUM_THREADS=1 mpirun -n 1 --allow-run-as-root unit_tests

# External-consumer smoke test: in compiled (default) mode VANTAGE-Reactions
# installs as libVANTAGE-Reactions.so; verify a standalone project can
# find_package it, link the .so and exercise a library-shipped instantiation
# without any source-tree access. Skipped in header-only mode (no .so to link).
VANTAGE_PREFIX=$(spack location -i vantagereactions)
if [ -f "${VANTAGE_PREFIX}/lib/libVANTAGE-Reactions.so" ]; then
  rm -rf build-consumer
  spack build-env --dirty vantagereactions cmake -S test/external_consumer -B build-consumer \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  cmake --build build-consumer >/dev/null
  ./build-consumer/consumer_smoke
fi

spack unload vantagereactions

