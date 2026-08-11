#ifndef REACTIONS_TEST_UTILS_H
#define REACTIONS_TEST_UTILS_H

#include <random>
#include <reactions/reactions.hpp>

namespace VANTAGE::Reactions {
inline NP::REAL relative_error(const NP::REAL correct, const NP::REAL to_test) {
  const NP::REAL abs_error = NP::Kernel::abs(correct - to_test);
  const NP::REAL abs_correct = NP::Kernel::abs(correct);
  return abs_correct > 0.0 ? abs_error / abs_correct : abs_error;
}

inline auto rng_lambda_wrapper_int =
    [](std::uniform_int_distribution<NP::INT> &dist, std::mt19937 &rng) {
      auto rng_lambda = [&]() -> NP::INT {
        NP::INT rng_sample = 0.0;
        do {
          rng_sample = dist(rng);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };

inline auto rng_lambda_wrapper_real =
    [](std::uniform_real_distribution<NP::REAL> &dist, std::mt19937 &rng) {
      auto rng_lambda = [&]() -> NP::REAL {
        NP::REAL rng_sample = 0.0;
        do {
          rng_sample = dist(rng);
        } while (rng_sample == 0.0);
        return rng_sample;
      };
      return rng_lambda;
    };
} // namespace VANTAGE::Reactions
#endif