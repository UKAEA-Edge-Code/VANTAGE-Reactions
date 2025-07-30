#ifndef NESO_TEST_ASSERT_H
#define NESO_TEST_ASSERT_H
#include <cstdlib>
#include <neso_particles/typedefs.hpp>
#include <stdexcept>

#undef NESOASSERT_FUNCTION
#define NESOASSERT_FUNCTION neso_particles_test_assert

template <typename T>
inline void neso_particles_test_assert(const char *expr_str, bool expr, const char *file, int line, T && msg) {
  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    if (!expr) {
      throw std::logic_error("");
    }
  }
  else {
    NESO::Particles::neso_particles_assert(expr_str, expr, file, line, msg);
  }
}
#include <neso_particles.hpp>
#endif