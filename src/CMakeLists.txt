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
  endif()
  target_link_libraries(VANTAGE-Reactions
      PUBLIC MPI::MPI_CXX NESO-Particles::NESO-Particles)
endif()
