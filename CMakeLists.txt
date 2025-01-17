cmake_minimum_required(VERSION 3.24)

# no space in these names! used as shared object name and python module name
set(PROJECT_NAME Reactions)
set(PROJECT_LIB_NAME ${PROJECT_NAME}-lib)

project(${PROJECT_NAME} VERSION 0.0.0 LANGUAGES CXX)

# Produce compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Create interface/Header only library
add_library(Reactions INTERFACE)
add_library(Reactions::Reactions ALIAS Reactions)

set_property(TARGET Reactions PROPERTY CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(NESO-Particles REQUIRED)
find_package(hipSYCL REQUIRED)

############################################################
# Add directories with source files
############################################################
add_subdirectory(src)

############################################################
# Set the build type environment variable
############################################################
option(REACTIONS_ENABLE_TESTS "Build tests." ON)
if (REACTIONS_ENABLE_TESTS)
  if (REACTIONS_DEVICE_TYPE STREQUAL GPU)
    set(BUILD_TYPE_COMPILE_FLAGS "-g;-O2")
  else()
    set(BUILD_TYPE_COMPILE_FLAGS "-g;-O2;--coverage")
    link_libraries(gcov)
    set(TEST_LIBRARIES "gcov")
  endif()
  enable_testing()
  add_subdirectory(test)
endif()

############################################################
# all configurable options
############################################################
find_package(Doxygen)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" OFF)

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
