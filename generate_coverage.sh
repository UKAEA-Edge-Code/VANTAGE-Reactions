#!/bin/bash
# Generate lcov HTML coverage for VANTAGE-Reactions.
# Pre-requisite: spack dependencies installed once via
#   spack -e environments/spack_default install --only dependencies
# Usage: ./generate_coverage.sh
# Note: run within the same container that run_tests.sh has already been run in.

: "${MAX_JOBS:=$(nproc)}"
: "${HTML:=OFF}"
BUILD=build-coverage

source /opt/spack/share/spack/setup-env.sh
cd /root/Reactions
spack env activate -p -d environments/spack_default
spack uninstall -y vantagereactions;
spack clean vantagereactions;

spack build-env vantagereactions -- cmake -S . -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="--coverage" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DCMAKE_SHARED_LINKER_FLAGS="--coverage" \
  -DREACTIONS_ENABLE_TESTS=ON
spack build-env vantagereactions -- cmake --build "$BUILD" -j"$MAX_JOBS"

# --ignore-errors inconsistent: For lcov >= 2.0, rejects bogus end lines that gcc's gcov emits for some templates
lcov --capture --initial -d "$BUILD" -o "$BUILD/base.info" --ignore-errors inconsistent,inconsistent,mismatch,mismatch -j "$MAX_JOBS"
OMP_NUM_THREADS=1 spack build-env vantagereactions -- mpirun --allow-run-as-root -n 1 "$BUILD/test/unit/unit_tests"
TEST_NESOASSERT=ON OMP_NUM_THREADS=1 spack build-env vantagereactions -- mpirun --allow-run-as-root -n 1 "$BUILD/test/unit/unit_tests"
lcov --capture -d "$BUILD" -o "$BUILD/tests.info" --ignore-errors inconsistent,inconsistent,mismatch,mismatch --rc geninfo_unexecuted_blocks=1 -j "$MAX_JOBS"
lcov -a "$BUILD/base.info" -a "$BUILD/tests.info" -o "$BUILD/coverage.info" --ignore-errors inconsistent,inconsistent,mismatch,mismatch -j "$MAX_JOBS"
lcov --remove \
  "$BUILD/coverage.info" \
  '*/neso-particles/*' \
  '*/test/*.cpp' \
  "*/$BUILD/*" \
  '/usr/*' \
  '/opt/spack/*' \
  -o "$BUILD/coverage_lib.info" --ignore-errors inconsistent,inconsistent,mismatch,mismatch,unused,unused -j "$MAX_JOBS"


if [ "$HTML" != "OFF" ]; then
  genhtml "$BUILD/coverage_lib.info" -o "$BUILD/coverage_html" --ignore-errors inconsistent,inconsistent,mismatch,mismatch -j "$MAX_JOBS"
  echo "Report: $BUILD/coverage_html/index.html"
fi
