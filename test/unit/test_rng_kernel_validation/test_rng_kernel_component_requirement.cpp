#include "include/test_common.hpp"
#include "reactions_lib/downsampling_kernels/simple_thinning_kernels.hpp"
#include "reactions_lib/reaction_data/filtered_maxwellian_sampler.hpp"
#include "reactions_lib/reaction_data/one_way_maxwellian_flux_sampler.hpp"
#include "reactions_lib/reaction_data/sampler_data.hpp"
#include "reactions_lib/rng_kernel_component_requirement.hpp"

using namespace VANTAGE::Reactions;

TEST(RNGKernelComponentRequirement, ReportsRequiredComponentAvailability) {
  using RNG = NP::HostAtomicBlockKernelRNG<REAL>;
  using Requirement = RNGKernelComponentRequirement<RNG, 3>;

  auto short_rng =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 2);
  auto sufficient_rng =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 3);
  Requirement requirement;

  EXPECT_EQ(Requirement::required_kernel_components, 3);
  EXPECT_FALSE(requirement.has_required_kernel_components(nullptr));
  EXPECT_FALSE(requirement.has_required_kernel_components(short_rng));
  EXPECT_TRUE(requirement.has_required_kernel_components(sufficient_rng));

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(requirement.validate_required_kernel_components(short_rng),
                 std::logic_error);
  }
}

TEST(RNGKernelComponentRequirement, FilteredMaxwellianRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 2);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(FilteredMaxwellianSampler<2>(1.0, rng_kernel),
                 std::logic_error);
  }
}

TEST(RNGKernelComponentRequirement, OneWayMaxwellianRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 3);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(OneWayMaxwellianFluxSampler(1.0, rng_kernel),
                 std::logic_error);
  }
}

TEST(RNGKernelComponentRequirement, SamplerDataRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 0);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(SamplerData(rng_kernel), std::logic_error);
  }
}

TEST(RNGKernelComponentRequirement, SimpleThinningRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_per_particle_block_rng<REAL>([]() -> REAL { return 0.5; }, 0);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(SimpleThinningKernels(0.5, rng_kernel), std::logic_error);
  }
}