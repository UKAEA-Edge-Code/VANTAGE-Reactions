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
cd ./coverage
lcov --ignore-errors mismatch,mismatch -c -d . --output-file lcov.txt
lcov --extract lcov.txt 'src/*' --output-file lcov_src.txt
