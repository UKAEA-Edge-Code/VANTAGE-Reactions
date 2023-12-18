file(GLOB_RECURSE src_files "*.h" "*.hpp" "*.cpp")

message(STATUS "src files:")
message(STATUS ${src_files})

add_executable(${PROJECT_NAME} ${src_files})

add_compile_options(${PROJECT_NAME} ${BUILD_TYPE_COMPILE_FLAGS})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../reactions_lib)
target_include_directories(${PROJECT_NAME} PUBLIC ${NESO_PARTICLES_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})

add_sycl_to_target(TARGET ${PROJECT_NAME} SOURCES ${src_files})
target_link_libraries(${PROJECT_NAME} PUBLIC ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES} ${NESO_PARTICLES_LIBRARIES} ${NESO_PARTICLES_LINK_FLAGS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}-lib)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
