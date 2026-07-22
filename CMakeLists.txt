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

# ---------------------------------------------------------------------------
# Library type
# ---------------------------------------------------------------------------
# By default VANTAGE-Reactions is built as a compiled library (SHARED, or
# STATIC when BUILD_SHARED_LIBS is OFF) that ships pre-compiled template
# instantiations and SYCL device code. The original header-only INTERFACE
# target is still available with -DVANTAGE_REACTIONS_HEADER_ONLY=ON.
option(VANTAGE_REACTIONS_HEADER_ONLY "Build header-only (no compiled library)" OFF)

if(VANTAGE_REACTIONS_HEADER_ONLY)
  add_library(VANTAGE-Reactions INTERFACE)
  add_library(VANTAGE::Reactions ALIAS VANTAGE-Reactions)
else()
  if(DEFINED BUILD_SHARED_LIBS AND NOT BUILD_SHARED_LIBS)
    set(VANTAGE_REACTIONS_DEFAULT_LIB_TYPE STATIC)
  else()
    set(VANTAGE_REACTIONS_DEFAULT_LIB_TYPE SHARED)
  endif()
  set(VANTAGE_REACTIONS_LIB_TYPE "${VANTAGE_REACTIONS_DEFAULT_LIB_TYPE}"
      CACHE STRING "Library type (STATIC or SHARED)")
  set_property(CACHE VANTAGE_REACTIONS_LIB_TYPE PROPERTY STRINGS STATIC SHARED)
  add_library(VANTAGE-Reactions ${VANTAGE_REACTIONS_LIB_TYPE})
  add_library(VANTAGE::Reactions ALIAS VANTAGE-Reactions)
endif()

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
# Add directories with source files
############################################################
add_subdirectory(src)

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
    FILE_SET public_headers
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
