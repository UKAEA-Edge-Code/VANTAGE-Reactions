#ifndef REACTIONS_NESO_TEST_ASSERT_H
#define REACTIONS_NESO_TEST_ASSERT_H
#include "neso_particles/typedefs.hpp"
#include <cstdlib>
#include <stdexcept>

#undef NESOASSERT_FUNCTION
#define NESOASSERT_FUNCTION neso_particles_test_assert

/**
 * @brief Helper function that disables NESOASSERT when TEST_NESOASSERT is set
 * and instead replaces it with a throw of std::logic_error if the expr boolean
 * is false. This is useful for unit tests where EXPECT_THROW is used to check
 * expected failures (it just checks that std::logic_error is thrown).
 *
 * @param expr_str A string identifying the conditional to check. (passed to
 * neso_particles_assert)
 * @param expr Bool resulting from the evaluation of the expression.
 * @param file Filename containing the call to neso_particles_assert. (passed to
 * neso_particles_assert)
 * @param line Line number for the call to neso_particles assert. (passed to
 * neso_particles_assert)
 * @param msg Message to print to stderr on evaluation of conditional to false.
 * (passed to neso_particles_assert)
 */

template <typename T>
inline void neso_particles_test_assert(const char *expr_str, bool expr,
                                       const char *file, int line, T &&msg) {
  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    if (!expr) {
      throw std::logic_error("");
    }
  } else {
    NESO::Particles::neso_particles_assert(expr_str, expr, file, line, msg);
  }
}
#include <neso_particles.hpp>
#endif