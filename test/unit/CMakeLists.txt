set(EXECUTABLE unit_tests)

# ---------------------------------------------------------------------------
# Split-test mode
# ---------------------------------------------------------------------------
# REACTIONS_TESTS_SPLIT=ON builds one executable per test_*.cpp (linked against
# a shared OBJECT library of the non-test TUs) instead of the single fat
# unit_tests monolith. REACTIONS_TEST_FILTER selects a subset by source stem
# (empty = all test_*.cpp; a single stem or ;-list = just those; an unknown
# stem is a hard configure error). The monolith remains the default and is the
# only target installed / used by run_tests.sh and CI.
option(REACTIONS_TESTS_SPLIT "Build one executable per test_*.cpp instead of the unit_tests monolith" OFF)
set(REACTIONS_TEST_FILTER "" CACHE STRING
    "When REACTIONS_TESTS_SPLIT=ON, build only these test_*.cpp stems (empty = all)")

if(NOT REACTIONS_TESTS_SPLIT)
  ############################################################
  # Default: the unit_tests monolith (unchanged behaviour)
  ############################################################
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
  
  add_executable(${EXECUTABLE} ${test_sources})

  target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest unit_test_common)
  target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
  target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${test_sources})

  install(TARGETS ${EXECUTABLE} DESTINATION bin)
else()
  ############################################################
  # Split mode: one executable per test_*.cpp
  ############################################################
  # Non-test TUs shared by every per-file exe: just the gtest/MPI main.
  # Compiled once into an OBJECT library and linked into each exe.
  set(common_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/instantiations/instantiations.cpp)
  add_library(unit_test_common OBJECT ${common_sources})
  target_link_libraries(unit_test_common PRIVATE
      VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
  target_compile_definitions(unit_test_common PRIVATE GPU_SELECTOR=0)
  target_compile_options(unit_test_common PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
  add_sycl_to_target(TARGET unit_test_common SOURCES ${common_sources})

  # The per-file executables are the test_*.cpp sources. Main.cpp does not
  # match test_*.cpp so it is excluded from this glob by construction.
  file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

  # Apply REACTIONS_TEST_FILTER (empty = all; stem / ;-list = subset).
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

  foreach(src ${test_sources})
    get_filename_component(name ${src} NAME_WE)        # e.g. test_reaction_base
    add_executable(${name} ${src})
    target_link_libraries(${name} PRIVATE
        VANTAGE::Reactions NESO-Particles::NESO-Particles
        GTest::gtest unit_test_common)
    target_compile_definitions(${name} PRIVATE GPU_SELECTOR=0)
    target_compile_options(${name} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
    add_sycl_to_target(TARGET ${name} SOURCES ${src})
    # Per-file executables are a dev-loop convenience: not installed. Only the
    # monolith is installed (preserves the Spack package + external-consumer
    # smoke test in run_tests.sh).
  endforeach()
endif()
