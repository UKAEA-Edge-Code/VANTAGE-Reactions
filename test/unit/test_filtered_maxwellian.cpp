#include "include/mock_particle_group.hpp"
#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(FilteredMaxwellianSampler, INVALID_NUM_COMPONENTS) {
  static constexpr size_t num_req_samples = 2;
  static constexpr size_t required_components = num_req_samples + 1;

  std::mt19937 rng = std::mt19937(52234126);
  std::uniform_real_distribution<REAL> uniform_dist(0.0, 1.0);

  auto rng_lambda = [&]() -> REAL {
    REAL rng_sample;
    do {
      rng_sample = uniform_dist(rng);
    } while (rng_sample == 0.0);
    return rng_sample;
  };

  auto rng_kernel = NP::host_atomic_block_kernel_rng<REAL>(
      rng_lambda, required_components - 1);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(FilteredMaxwellianSampler<2>(1.0, rng_kernel),
                 std::logic_error);
  }
}