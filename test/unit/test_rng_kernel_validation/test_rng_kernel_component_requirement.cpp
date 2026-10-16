#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(ValidateRequiredKernelComponents, SufficientKernelPasses) {
  using RNG = NP::HostAtomicBlockKernelRNG<REAL>;
  auto sufficient_rng =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 3);

  // A kernel with enough components must not throw (regardless of whether
  // TEST_NESOASSERT is set).
  utils::validate_required_kernel_components<3>(sufficient_rng);
}

TEST(ValidateRequiredKernelComponents, InsufficientKernelFails) {
  using RNG = NP::HostAtomicBlockKernelRNG<REAL>;
  auto short_rng =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 2);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(utils::validate_required_kernel_components<3>(short_rng),
                 std::logic_error);
    EXPECT_THROW(utils::validate_required_kernel_components<3>(
                     std::shared_ptr<RNG>(nullptr)),
                 std::logic_error);
  }
}

TEST(ValidateRequiredKernelComponents, SamplerRequiredComponentCounts) {
  EXPECT_EQ(OneWayMaxwellianFluxSampler::required_kernel_components, 4);
  EXPECT_EQ(FilteredMaxwellianSampler<2>::required_kernel_components, 3);
  EXPECT_EQ(FilteredMaxwellianSampler<3>::required_kernel_components, 5);
  EXPECT_EQ(SamplerData<
                NP::HostAtomicBlockKernelRNG<REAL>>::required_kernel_components,
            1);
  EXPECT_EQ(SimpleThinningKernels::required_kernel_components, 1);
}

TEST(ValidateRequiredKernelComponents,
     FilteredMaxwellianRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 2);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(FilteredMaxwellianSampler<2>(1.0, rng_kernel),
                 std::logic_error);
  }
}

TEST(ValidateRequiredKernelComponents, OneWayMaxwellianRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 3);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(OneWayMaxwellianFluxSampler(1.0, rng_kernel),
                 std::logic_error);
  }
}

TEST(ValidateRequiredKernelComponents, SamplerDataRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_atomic_block_kernel_rng<REAL>([]() -> REAL { return 0.5; }, 0);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW((void)SamplerData(rng_kernel), std::logic_error);
  }
}

TEST(ValidateRequiredKernelComponents, SimpleThinningRejectsInsufficientRNG) {
  auto rng_kernel =
      NP::host_per_particle_block_rng<REAL>([]() -> REAL { return 0.5; }, 0);

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW(SimpleThinningKernels(0.5, rng_kernel), std::logic_error);
  }
}