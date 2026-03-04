#ifndef REACTIONS_TEST_UTILS_H
#define REACTIONS_TEST_UTILS_H
#include <reactions/reactions.hpp>

namespace VANTAGE::Reactions {
inline REAL relative_error(const REAL correct, const REAL to_test) {
  const REAL abs_error = Kernel::abs(correct - to_test);
  const REAL abs_correct = Kernel::abs(correct);
  return abs_correct > 0.0 ? abs_error / abs_correct : abs_error;
}
} // namespace VANTAGE::Reactions
#endif