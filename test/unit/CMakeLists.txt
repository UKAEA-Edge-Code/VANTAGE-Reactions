file(GLOB_RECURSE src_unit "*.cpp")

set(EXECUTABLE unit_tests)
include(GoogleTest)

add_executable(${EXECUTABLE} ${src_unit})
target_precompile_headers(
        ${EXECUTABLE} PRIVATE
        ${CMAKE_SOURCE_DIR}/src/reactions/reactions.hpp
        ${CMAKE_SOURCE_DIR}/test/unit/include/mock_particle_group.hpp
        ${CMAKE_SOURCE_DIR}/test/unit/include/mock_particle_properties.hpp
        ${CMAKE_SOURCE_DIR}/test/unit/include/mock_reactions.hpp
    )

target_link_libraries(${EXECUTABLE} PRIVATE VANTAGE::Reactions NESO-Particles::NESO-Particles GTest::gtest)
target_compile_definitions(${EXECUTABLE} PRIVATE GPU_SELECTOR=0)
target_compile_options(${EXECUTABLE} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
add_sycl_to_target(TARGET ${EXECUTABLE} SOURCES ${src_unit})
gtest_add_tests(TARGET ${EXECUTABLE} SOURCES ${src_unit})

install(TARGETS ${EXECUTABLE} DESTINATION bin)
