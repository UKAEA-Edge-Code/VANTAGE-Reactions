cmake_minimum_required(VERSION 3.24)

# no space in these names! used as shared object name and python module name
set(PROJECT_NAME VANTAGE-Reactions)

project(${PROJECT_NAME} VERSION 0.0.0 LANGUAGES CXX)

include(GNUInstallDirs)

# RPATH so installed binaries/libraries find their deps without
# LD_LIBRARY_PATH juggling (matches neso-particles).
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Produce compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Designate as shared library and alias definition
add_library(VANTAGE-Reactions SHARED)
add_library(VANTAGE::Reactions ALIAS VANTAGE-Reactions)


if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
message(STATUS CMAKE_CXX_STANDARD " is " ${CMAKE_CXX_STANDARD})
set_property(TARGET VANTAGE-Reactions PROPERTY CXX_STANDARD
    ${CMAKE_CXX_STANDARD})

find_package(MPI REQUIRED)
find_package(NESO-Particles REQUIRED)
find_sycl_if_required()

############################################################
# Include headers
############################################################
file(GLOB_RECURSE include_headers "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

message(STATUS "header files:")

string(REPLACE ";" "\n" delim_include_headers "${include_headers}")
message(STATUS ${delim_include_headers})

set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HEADER_FILES ${include_headers})

# Installed headers (FILE_SET), in both header-only and compiled modes.
target_sources(
    VANTAGE-Reactions
    PUBLIC FILE_SET
        headers
        TYPE
        HEADERS
        BASE_DIRS
        ${INCLUDE_DIR}
        FILES
        ${HEADER_FILES}
)

############################################################
# Install compiled runtime library
############################################################
# Compiled library sources: explicit template instantiations (and any future
# non-template definitions). Compiled with SYCL enabled on this target so the
# device code lives in the library, not in every consumer.
file(GLOB_RECURSE src_compiled "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
message(STATUS "src files:")

string(REPLACE ";" "\n" delim_src_compiled "${src_compiled}")

message(STATUS ${delim_src_compiled})

target_sources(VANTAGE-Reactions PRIVATE ${src_compiled})
add_sycl_to_target(TARGET VANTAGE-Reactions SOURCES ${src_compiled})

# IntelSYCL's add_sycl_to_target applies -fsycl and the SYCL include path
# via per-source COMPILE_OPTIONS/INCLUDE_DIRECTORIES. CMake silently drops
# per-source properties for sources added to a target with target_sources()
# from a subdirectory when the target was created in the parent scope (this
# target is created in the root CMakeLists.txt), so the compiled
# instantiations TU never receives them and fails with 'CL/sycl.hpp' not
# found. Re-apply them at the target level. AdaptiveCpp drives SYCL through
# a per-target compiler launcher instead of per-source flags, so it is
# unaffected; the guard keeps this fix specific to the IntelSYCL backend.
if(TARGET IntelSYCL::SYCL_CXX)
  target_compile_options(
    VANTAGE-Reactions
    PRIVATE $<TARGET_PROPERTY:IntelSYCL::SYCL_CXX,INTERFACE_COMPILE_OPTIONS>)
  target_include_directories(
    VANTAGE-Reactions
    PRIVATE $<TARGET_PROPERTY:IntelSYCL::SYCL_CXX,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

target_link_libraries(VANTAGE-Reactions
    PUBLIC MPI::MPI_CXX NESO-Particles::NESO-Particles)
message(STATUS "Runtime library linked.")

############################################################
# Set the build type environment variable
############################################################
option(REACTIONS_ENABLE_TESTS "Build tests." ON)
# The external-consumer smoke test (test/external_consumer/) is a standalone
# CMake project that links ONLY the installed compiled library via
# find_package(VANTAGE-Reactions). It cannot be built as part of the main
# build because it requires a prior `cmake --install`. CI runs it as a
# separate post-install step; see run_tests.sh and .github/workflows/tests-ci.yml.

if (REACTIONS_ENABLE_TESTS)
  add_subdirectory(test)
endif()

install(
    TARGETS VANTAGE-Reactions
    EXPORT ${PROJECT_NAME}_Targets
    FILE_SET headers
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    EXPORT ${PROJECT_NAME}_Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})
write_basic_package_version_file(
  ${PROJECT_NAME}ConfigVersion.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
export(
    EXPORT ${PROJECT_NAME}_Targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
    NAMESPACE ${PROJECT_NAME}::
)
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
