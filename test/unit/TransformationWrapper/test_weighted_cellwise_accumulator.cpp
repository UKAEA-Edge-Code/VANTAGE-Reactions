#include <reactions.hpp>
#include <gtest/gtest.h>
#include "transformation_wrapper_utils.hpp"

using namespace NESO::Particles;
using namespace Reactions;

TEST(TransformationWrapper, WeightedCellwiseAccumulator) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group_marking(N_total);

  auto accumulator_transform =
      std::make_shared<WeightedCellwiseAccumulator<REAL>>(
          particle_group, std::vector<std::string>{"MOCK_SOURCE2D"},
          "MOCK_SOURCE1D");

  auto test_wrapper = TransformationWrapper(
      std::dynamic_pointer_cast<TransformationStrategy>(accumulator_transform));
  test_wrapper.transform(particle_group);

  auto num_cells = particle_group->domain->mesh->get_cell_count();
  auto accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  auto accumulated_weight = accumulator_transform->get_weight_cell_data();
  // Here we use the 1D mock source as a weight for this test (because there are
  // different weights in the default particle group used in this set of tests)
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_weight[cellx]->at(0, 0), 0.5 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.5 * 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.5 * 0.2 * num_parts, 1e-10);
  };
  // Testing out zeroing features and repeated accumulation

  accumulator_transform->zero_buffer("MOCK_SOURCE2D");
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_weight[cellx]->at(0, 0), 0.5 * num_parts);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  test_wrapper.transform(particle_group);
  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_weight = accumulator_transform->get_weight_cell_data();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    // Kept as EXPECT_NEAR for consistency with accumulated_2d checks
    EXPECT_NEAR(accumulated_weight[cellx]->at(0, 0), 1.0 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(0, 0), 0.5 * 0.1 * num_parts, 1e-10);
    // Result can be out by as much as ULP=8 so EXPECT_DOUBLE_EQ is not
    // appropriate.
    EXPECT_NEAR(accumulated_2d[cellx]->at(1, 0), 0.5 * 0.2 * num_parts, 1e-10);
  };

  accumulator_transform->zero_all_buffers();

  accumulated_2d = accumulator_transform->get_cell_data("MOCK_SOURCE2D");
  accumulated_weight = accumulator_transform->get_weight_cell_data();
  for (int cellx = 0; cellx < num_cells; cellx++) {
    auto num_parts = particle_group->get_npart_cell(cellx);

    EXPECT_DOUBLE_EQ(accumulated_weight[cellx]->at(0, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(1, 0), 0);
    EXPECT_DOUBLE_EQ(accumulated_2d[cellx]->at(0, 0), 0);
  };

  particle_group->domain->mesh->free();
}