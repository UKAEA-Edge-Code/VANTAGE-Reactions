file(GLOB_RECURSE src_files "*.h" "*.hpp" "*.cpp")

message(STATUS "src files:")
message(STATUS ${src_files})

# make a shared object library to link against for testing
add_library(${PROJECT_LIB_NAME} SHARED ${src_files})
target_compile_options(${PROJECT_LIB_NAME} PRIVATE ${BUILD_TYPE_COMPILE_FLAGS})
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_LIB_NAME} PROPERTIES OUTPUT_NAME ${PROJECT_NAME})
