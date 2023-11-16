file(GLOB_RECURSE src_files "*.h" "*.hpp" "*.cpp")

message(STATUS "src files:")
message(STATUS ${src_files})

add_executable(${PROJECT_NAME} ${src_files})

add_compile_options(${PROJECT_NAME} ${BUILD_TYPE_COMPILE_FLAGS})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_11)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../reactions_lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}-lib)



