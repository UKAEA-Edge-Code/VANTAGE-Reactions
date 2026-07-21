file(GLOB_RECURSE src_unit "*.cpp")

set(EXECUTABLE unit_tests)
include(GoogleTest)

add_executable(${EXECUTABLE} ${src_unit})

target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
# When the library is compiled it ships the example instantiations itself
# (src/reactions_lib/instantiations/instantiations.cpp); in that case the test
# TU must NOT re-instantiate them or the linker sees duplicate symbols. Mirror
# the library's build mode into the test compile definitions so the seed
# explicit-instantiation TU can gate itself.
if(VANTAGE_REACTIONS_HEADER_ONLY)
  target_compile_definitions(${EXECUTABLE} PRIVATE VANTAGE_REACTIONS_HEADER_ONLY=1)
endif()
target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${src_unit})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${src_unit})

install(TARGETS ${EXECUTABLE} DESTINATION bin)
