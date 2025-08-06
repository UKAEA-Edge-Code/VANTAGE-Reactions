#!/bin/bash

source /opt/spack/share/spack/setup-env.sh
spack env activate -p -d .
spack install
OMP_NUM_THREADS=1 find build-linux-ubuntu24.04*/*/test/unit/ -name "unit_tests" -exec mpirun -n 1 {} \;