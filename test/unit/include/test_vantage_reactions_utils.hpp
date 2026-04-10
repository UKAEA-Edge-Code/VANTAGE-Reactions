#ifndef REACTIONS_TEST_UTILS_H
#define REACTIONS_TEST_UTILS_H
#include <reactions/reactions.hpp>

namespace VANTAGE::Reactions {
inline REAL relative_error(const REAL correct, const REAL to_test) {
  const REAL abs_error = Kernel::abs(correct - to_test);
  const REAL abs_correct = Kernel::abs(correct);
  return abs_correct > 0.0 ? abs_error / abs_correct : abs_error;
}

static inline auto rng(const int &rank) {
  return std::mt19937(52234126 + rank);
};

inline auto rng_lambda_wrapper_int =
    [](std::uniform_int_distribution<INT> &dist, const int &rank) {
      auto rng_lambda = [&]() -> INT {
        INT rng_sample = 0.0;
        auto rng_ = rng(rank);
        do {
          rng_sample = dist(rng_);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };

inline auto rng_lambda_wrapper_real =
    [](std::uniform_real_distribution<REAL> &dist, const int &rank) {
      auto rng_lambda = [&]() -> REAL {
        REAL rng_sample = 0.0;
        auto rng_ = rng(rank);
        do {
          rng_sample = dist(rng_);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };
} // namespace VANTAGE::Reactions
#endif