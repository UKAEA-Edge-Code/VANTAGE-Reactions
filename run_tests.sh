#!/bin/bash

source /opt/spack/share/spack/setup-env.sh
spack env activate -p -d .
spack install
spack load reactions
OMP_NUM_THREADS=1 mpirun -n 1 unit_tests
mkdir -p coverage
find ./src -name '*.hpp' -exec cp -t ./coverage {} +
find ./build-linux-ubuntu24.04*/*/test/unit/CMakeFiles/unit_tests.dir -name '*.gcno' -exec cp -t ./coverage {} +
find ./build-linux-ubuntu24.04*/*/test/unit/CMakeFiles/unit_tests.dir -name '*.gcda' -exec cp -t ./coverage {} +
find ./coverage -name '*.hpp' -exec gcov -pb {} \;
lcov --ignore-errors mismatch -c -d ./coverage --output-file ./coverage/coverage.txt
