#!/bin/bash

source /opt/spack/share/spack/setup-env.sh
cd /root/Reactions
spack env activate -p -d environments/spack_omp_accelerated
spack install
spack load reactions
OMP_NUM_THREADS=1 mpirun -n 1 --allow-run-as-root unit_tests
