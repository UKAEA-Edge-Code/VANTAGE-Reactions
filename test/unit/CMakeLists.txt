set(EXECUTABLE unit_tests)

# ---------------------------------------------------------------------------
# Per-file test executables
# ---------------------------------------------------------------------------
# One executable is built per test_*.cpp, each linking a shared OBJECT
# library of the non-test TUs (unit_test_common). REACTIONS_TEST_FILTER
# selects a subset by source stem (empty = all test_*.cpp, which also builds
# the unit_tests monolith; a single stem or ;-list = just those per-file
# executables, no monolith; an unknown stem is a hard configure error).
# The monolith is the only target installed / used by run_tests.sh and CI.
#
# Each test_*.cpp is compiled exactly once into a per-file OBJECT library;
# the monolith and the matching per-file executable link the object library
# by name (CMake >= 3.12: pulls in its objects, usage requirements and link
# dependencies), so selecting all stems does not duplicate any TU compilation.
set(REACTIONS_TEST_FILTER "" CACHE STRING
    "Build only these test_*.cpp stems (empty = all + the unit_tests monolith)")


set(common_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/instantiations/instantiations.cpp)
add_library(unit_test_common OBJECT ${common_sources})
target_link_libraries(unit_test_common PRIVATE
    VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(unit_test_common PRIVATE GPU_SELECTOR=0)
target_compile_options(unit_test_common PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
add_sycl_to_target(TARGET unit_test_common SOURCES ${common_sources})

file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

############################################################
# Apply REACTIONS_TEST_FILTER (empty = all; stem / ;-list = subset).
############################################################
if(REACTIONS_TEST_FILTER)
  set(wanted ${REACTIONS_TEST_FILTER})
  set(filtered)
  set(matched_stems)
  foreach(src ${test_sources})
    get_filename_component(stem ${src} NAME_WE)
    if(stem IN_LIST wanted)
      list(APPEND filtered ${src})
      list(APPEND matched_stems ${stem})
    endif()
  endforeach()
  # Typo-proofing: every requested stem must match a test_*.cpp.
  foreach(w ${wanted})
    if(NOT w IN_LIST matched_stems)
      message(FATAL_ERROR
        "REACTIONS_TEST_FILTER='${w}' matches no test_*.cpp stem in "
        "${CMAKE_CURRENT_SOURCE_DIR}.")
    endif()
  endforeach()
  set(test_sources ${filtered})
endif()

############################################################
# One OBJECT library + one executable per test_*.cpp
############################################################
set(test_object_libraries)
foreach(src ${test_sources})
  get_filename_component(name ${src} NAME_WE)        # e.g. test_reaction_base

  # Compile the TU exactly once; both the per-file exe and the monolith
  # link these objects.
  set(obj ${name}_objects)
  add_library(${obj} OBJECT ${src})
  target_link_libraries(${obj} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest unit_test_common)
  target_compile_definitions(${obj} PRIVATE GPU_SELECTOR=0)
  target_compile_options(${obj} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
  add_sycl_to_target(TARGET ${obj} SOURCES ${src})
  list(APPEND test_object_libraries ${obj})

  # Per-file executables are a dev-loop convenience: not installed. Only the
  # monolith is installed (preserves the Spack package + external-consumer
  # smoke test in run_tests.sh). The link deps of ${obj} (VANTAGE::Reactions,
  # NESO-Particles, GTest, acpp-rt) propagate transitively.
  add_executable(${name})
  target_link_libraries(${name} PRIVATE ${obj} VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest unit_test_common)
  add_sycl_to_target(TARGET ${name})
endforeach()

############################################################
# The unit_tests monolith (all test objects in one exe)
############################################################
if (REACTIONS_TEST_FILTER STREQUAL "")
add_executable(${EXECUTABLE})
target_link_libraries(${EXECUTABLE} PRIVATE ${test_object_libraries} VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest unit_test_common)
add_sycl_to_target(TARGET ${EXECUTABLE})

install(TARGETS ${EXECUTABLE} DESTINATION bin)
endif()
