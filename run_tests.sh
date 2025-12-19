#!/bin/bash

source /opt/spack/share/spack/setup-env.sh
spack env activate -p -d .
spack install
spack load reactions
OMP_NUM_THREADS=1 mpirun -n 1 unit_tests
mkdir coverage_report
pipx run gcovr build-linux-ubuntu24.04*/*/test/unit/CMakeFiles/unit_tests.dir -r ./src --jacoco-pretty --jacoco coverage_report/coverage.xml
