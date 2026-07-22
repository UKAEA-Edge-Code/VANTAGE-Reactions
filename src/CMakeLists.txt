file(GLOB_RECURSE src_headers "*.hpp")

message(STATUS "src headers:")
message(STATUS ${src_headers})

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_REACTIONS_DIR ${INCLUDE_DIR}/reactions_lib)
set(HEADER_FILES ${src_headers})

# Installed public headers (FILE_SET), in both header-only and compiled modes.
target_sources(
    VANTAGE-Reactions
    PUBLIC FILE_SET
        public_headers
        TYPE
        HEADERS
        BASE_DIRS
        ${INCLUDE_DIR}
        FILES
        ${HEADER_FILES}
)

if(VANTAGE_REACTIONS_HEADER_ONLY)
  target_link_libraries(VANTAGE-Reactions
      INTERFACE MPI::MPI_CXX NESO-Particles::NESO-Particles)
else()
  # Compiled library sources: explicit template instantiations (and any future
  # non-template definitions). Compiled with SYCL enabled on this target so the
  # device code lives in the library, not in every consumer.
  file(GLOB_RECURSE src_compiled "*.cpp")
  if(src_compiled)
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
  endif()
  target_link_libraries(VANTAGE-Reactions
      PUBLIC MPI::MPI_CXX NESO-Particles::NESO-Particles)
endif()
