cmake_minimum_required(VERSION 3.14)

# no space in these names! used as shared object name and python module name
set(PROJECT_NAME reactions) 
set(PROJECT_LIB_NAME ${PROJECT_NAME}-lib)

set(CMAKE_CXX_STANDARD 17)
project(${PROJECT_NAME} VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

#include(${CMAKE_SOURCE_DIR}/cmake/SetBuildType.cmake)

enable_testing()

## put all targets in bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
## put all libraries in lib
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# Produce compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(MPI REQUIRED)
find_package(NESO-PARTICLES REQUIRED)
find_package(hipSYCL REQUIRED)

############################################################
# Set the build type environment variable
############################################################
if(BUILD_TYPE STREQUAL "TEST")
  set(BUILD_TYPE_COMPILE_FLAGS "-g;-O0;--coverage")
  link_libraries(gcov)
  set(TEST_LIBRARIES "gcov")
elseif(BUILD_TYPE STREQUAL "DEBUG")
  set(BUILD_TYPE_COMPILE_FLAGS "-g;-O0")
  set(TEST_LIBRARIES "")
elseif(BUILD_TYPE STREQUAL "RELEASE")
  set(BUILD_TYPE_COMPILE_FLAGS "-O2")
  set(TEST_LIBRARIES "")
endif()

############################################################
# Add directories with source files
############################################################
add_subdirectory(src)
add_subdirectory(test)

############################################################
# all configurable options
############################################################
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ${DOXYGEN_FOUND})

############################################################
# Create doxygen documentation `make doc`
############################################################
if(BUILD_DOCUMENTATION)
    if(NOT DOXYGEN_FOUND)
        message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()

    set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${doxyfile_in} ${doxyfile} @ONLY)

    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
endif()
