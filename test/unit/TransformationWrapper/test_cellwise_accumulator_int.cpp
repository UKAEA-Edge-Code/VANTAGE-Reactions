#include <reactions.hpp>
#include <gtest/gtest.h>
#include "transformation_wrapper_utils.hpp"

using namespace NESO::Particles;
using namespace Reactions;

TEST(TransformationWrapper, CellwiseAccumulatorINT) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform = std::make_shared<CellwiseAccumulator<INT>>(
      particle_group, std::vector<std::string>{"MOCK_INT"});

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_1d = accumulator_transform->get_cell_data("MOCK_INT");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_1d[cellx]->at(0, 0), num_parts); //, 1e-10);
  };

  particle_group->domain->mesh->free();
}