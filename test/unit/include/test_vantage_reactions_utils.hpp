#ifndef REACTIONS_TEST_UTILS_H
#define REACTIONS_TEST_UTILS_H

#include "test_common.hpp"
#include <random>

namespace VANTAGE::Reactions {
inline REAL relative_error(const REAL correct, const REAL to_test) {
  const REAL abs_error = NP::Kernel::abs(correct - to_test);
  const REAL abs_correct = NP::Kernel::abs(correct);
  return abs_correct > 0.0 ? abs_error / abs_correct : abs_error;
}

inline auto rng_lambda_wrapper_int =
    [](std::uniform_int_distribution<INT> &dist, std::mt19937 &rng) {
      auto rng_lambda = [&]() -> INT {
        INT rng_sample = 0.0;
        do {
          rng_sample = dist(rng);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };

inline auto rng_lambda_wrapper_real =
    [](std::uniform_real_distribution<REAL> &dist, std::mt19937 &rng) {
      auto rng_lambda = [&]() -> REAL {
        REAL rng_sample = 0.0;
        do {
          rng_sample = dist(rng);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };
} // namespace VANTAGE::Reactions
#endif