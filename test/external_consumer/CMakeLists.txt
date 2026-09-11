cmake_minimum_required(VERSION 3.24)
project(vantage_consumer_smoke LANGUAGES CXX)

# Consumer links ONLY the installed VANTAGE-Reactions library + its transitive
# deps (MPI, NESO-Particles, SYCL). No access to the source tree.
find_package(VANTAGE-Reactions REQUIRED)

add_executable(consumer_smoke consumer_smoke.cpp)
# The installed EXPORT namespace is VANTAGE-Reactions:: (the in-tree alias
# VANTAGE::Reactions is only available to the source build). Both refer to
# the same target; consumers of the installed package use the EXPORT name.
target_link_libraries(consumer_smoke PRIVATE VANTAGE-Reactions::VANTAGE-Reactions)
# The consumer's own TU must also be compiled with the SYCL launcher for any
# device code it triggers through the library templates.
add_sycl_to_target(TARGET consumer_smoke SOURCES consumer_smoke.cpp)
