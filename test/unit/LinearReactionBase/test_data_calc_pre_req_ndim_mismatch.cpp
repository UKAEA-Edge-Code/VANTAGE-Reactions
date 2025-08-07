#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(LinearReactionBase, data_calc_pre_req_ndim_mismatch) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");

  auto particle_group = create_test_particle_group(100);

  struct TestDataCalcNdimReaction
      : public LinearReactionBase<0, FixedRateData, IoniseReactionKernels<>> {
    TestDataCalcNdimReaction(ParticleGroupSharedPtr particle_group)
        : LinearReactionBase<0, FixedRateData, IoniseReactionKernels<>>(
              particle_group->sycl_target, 0, std::array<int, 0>{},
              FixedRateData(1),
              IoniseReactionKernels<>(Species("ION", 1.0, 1.0, 0),
                                      Species("ELECTRON"), Species("ELECTRON"))) {}
  };

  if (std::getenv("TEST_NESOASSERT") != nullptr) {
    EXPECT_THROW((TestDataCalcNdimReaction(particle_group)), std::logic_error);
  }

  particle_group->domain->mesh->free();
}