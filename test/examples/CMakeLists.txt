file(GLOB_RECURSE src_examples "*.cpp")

set(EXECUTABLE examples)
include(GoogleTest)

add_executable(${EXECUTABLE} ${src_examples})

if (REACTIONS_ENABLE_COVERAGE)
  set(COVERAGE_LINK gcov)
endif()

target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest ${COVERAGE_LINK})
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${src_examples})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${src_examples})

install(TARGETS ${EXECUTABLE} DESTINATION bin)
