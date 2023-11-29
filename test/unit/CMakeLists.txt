file(GLOB_RECURSE src_unit "*.cpp")

add_executable(unit_tests ${src_unit})
add_compile_options(unit_tests ${BUILD_TYPE_COMPILE_FLAGS})

target_compile_features(unit_tests PRIVATE cxx_std_11)
target_include_directories(unit_tests PRIVATE ${GTEST_INCLUDE_DIRS})
target_include_directories(unit_tests PRIVATE ${GMOCK_INCLUDE_DIRS})
target_include_directories(unit_tests PUBLIC ${PROJECT_SOURCE_DIR}/src/${PROJECT_NAME}_lib)
target_include_directories(unit_tests PUBLIC ${NESO_PARTICLES_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_sycl_to_target(TARGET unit_tests SOURCES src_unit)
target_link_libraries(unit_tests PUBLIC ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})
target_link_libraries(unit_tests PUBLIC ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY} pthread)
target_link_libraries(unit_tests PUBLIC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/lib${PROJECT_NAME}.so)

include(GoogleTest)
gtest_discover_tests(unit_tests)