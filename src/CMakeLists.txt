file(GLOB_RECURSE src_files "*.hpp")

message(STATUS "src files:")
message(STATUS ${src_files})

set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_REACTIONS_DIR ${INCLUDE_DIR}/reactions_lib)
set(HEADER_FILES ${src_files})

target_sources(
    Reactions
    PUBLIC FILE_SET
        public_headers
        TYPE
        HEADERS
        BASE_DIRS
        ${INCLUDE_DIR}
        FILES
        ${HEADER_FILES}
)

target_link_libraries(Reactions INTERFACE MPI::MPI_CXX NESO-Particles::NESO-Particles Reactions)
