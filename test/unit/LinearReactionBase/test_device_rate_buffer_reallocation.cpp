#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(LinearReactionBase, device_rate_buffer_reallocation) {
  auto particle_group = create_test_particle_group(1600);

  struct TestDeviceRateBufferReaction
      : public LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                                  DataCalculator<FixedRateData>> {

    TestDeviceRateBufferReaction(ParticleGroupSharedPtr particle_group)
        : LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                             DataCalculator<FixedRateData>>(
              particle_group->sycl_target, 0, std::array<int, 0>{},
              FixedRateData(1),
              IoniseReactionKernels<2>(Species("ION", 1.0, 1.0, 0),
                                       Species("ELECTRON"),
                                       Species("ELECTRON")),
              DataCalculator<FixedRateData>(FixedRateData(1))) {}

    const LocalArraySharedPtr<REAL> &get_device_rate_buffer_derived() {
      return this->get_device_rate_buffer();
    }
  };

  auto test_reaction = TestDeviceRateBufferReaction(particle_group);

  // Starting particle number in cell #0: 100
  test_reaction.blockwise_flush_buffer(
      std::make_shared<ParticleSubGroup>(particle_group), 0, 1);
  EXPECT_EQ(test_reaction.get_device_rate_buffer_derived()->size, 200);

  // Subtract 70 particles
  std::vector<INT> cells;
  std::vector<INT> layers;
  cells.reserve(70);
  layers.reserve(70);

  for (int i = 0; i < 70; i++) {
    cells.push_back(0);
    layers.push_back(i);
  }

  particle_group->remove_particles(cells.size(), cells, layers);

  // Check resize of device_rate_buffer to n_part_cell*2 = 60
  test_reaction.blockwise_flush_buffer(
      std::make_shared<ParticleSubGroup>(particle_group), 0, 1);
  EXPECT_EQ(test_reaction.get_device_rate_buffer_derived()->size, 60);

  particle_group->domain->mesh->free();
}