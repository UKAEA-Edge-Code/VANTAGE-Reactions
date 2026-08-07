set(EXECUTABLE unit_tests)

file(GLOB instantiation_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/instantiations/*.cpp")
set(common_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
    ${instantiation_sources})

add_library(unit_test_common OBJECT ${common_sources})
target_link_libraries(unit_test_common PRIVATE
    VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(unit_test_common PRIVATE GPU_SELECTOR=0)
target_compile_options(unit_test_common PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
set_target_properties(unit_test_common PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_sycl_to_target(TARGET unit_test_common SOURCES ${common_sources})

file(GLOB_RECURSE test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp")

############################################################
# One OBJECT library + one executable per test_*.cpp
############################################################
set(test_object_libraries)
# Install individual tests with: make {test_file_name_without_extension}
add_custom_target(tests EXCLUDE_FROM_ALL TRUE)
foreach(src ${test_sources})
  get_filename_component(name ${src} NAME_WE)        # e.g. test_reaction_base

  # Per-file executables are a dev-loop convenience: not installed. Only the
  # monolith is installed (preserves the Spack package + external-consumer
  # smoke test in run_tests.sh).
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest unit_test_common)
  target_compile_definitions(${name} PRIVATE GPU_SELECTOR=0)
  target_compile_options(${name} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})

  set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_dependencies(tests ${name})
  add_sycl_to_target(TARGET ${name})
endforeach()

############################################################
# The unit_tests monolith (all test objects in one exe)
############################################################
add_executable(${EXECUTABLE} ${test_sources} ${common_sources})
target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
add_sycl_to_target(TARGET ${EXECUTABLE})

install(TARGETS ${EXECUTABLE} DESTINATION bin)
