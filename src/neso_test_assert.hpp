#pragma once
#ifdef TEST_ASSERT
#include <stdexcept>
#define NESOASSERT_FUNCTION neso_particles_test_assert

template <typename T>
inline void neso_particles_test_assert(const char *expr_str, bool expr, const char *file, int line, T && msg) {
  if (!expr) {
    throw std::logic_error("");
  }
}
#include <neso_particles.hpp>
#endif