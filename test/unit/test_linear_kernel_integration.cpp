#include "include/mock_particle_group.hpp"
#include "include/test_common.hpp"

using namespace VANTAGE::Reactions;

TEST(LinearKernelIntegration, RemappedInputAndDescendantIndices) {
  // Reverse velocity/weight ordering relative to the default names.
  auto map = PropertiesMap();
  map.at(default_properties.weight) = "A_WEIGHT";
  map.at(default_properties.velocity) = "Z_VELOCITY";
  auto group = create_test_particle_group(256);
  group->add_particle_dat(NP::Sym<REAL>("A_WEIGHT"), 1);
  group->add_particle_dat(NP::Sym<REAL>("Z_VELOCITY"), 2);
  auto subgroup = NP::particle_sub_group(group);
  particle_loop(
      "initialize_remapped_kernel", subgroup,
      [=](auto weight, auto velocity) {
        weight.at(0) = 2.0;
        velocity.at(0) = 3.0;
        velocity.at(1) = -4.0;
      },
      NP::Access::write(NP::Sym<REAL>("A_WEIGHT")),
      NP::Access::write(NP::Sym<REAL>("Z_VELOCITY")))
      ->execute();

  const Species target("ION", 2.0, 0.0, 5), projectile("ION2", 3.0, 0.0, 0);
  auto calculator = DataCalculator(FixedRateData(-1.0), FixedRateData(2.0));
  LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                     decltype(calculator)>
      reaction(group->sycl_target, 0, {5}, FixedRateData(1.0),
               CXReactionKernels<2>(target, projectile, map), calculator, map);
  auto children = std::make_shared<NP::ParticleGroup>(
      group->domain, group->get_particle_spec(), group->sycl_target);
  const int cells = group->domain->mesh->get_cell_count();
  for (int cell = 0; cell < cells; ++cell) {
    reaction.calculate_rates(subgroup, cell, cell + 1);
    reaction.apply(subgroup, cell, cell + 1, 0.25, children);
    auto weights = group->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto child_weights = children->get_cell(NP::Sym<REAL>("A_WEIGHT"), cell);
    auto velocities = children->get_cell(NP::Sym<REAL>("Z_VELOCITY"), cell);
    auto target_density =
        group->get_cell(NP::Sym<REAL>("ION_SOURCE_DENSITY"), cell);
    auto projectile_density =
        group->get_cell(NP::Sym<REAL>("ION2_SOURCE_DENSITY"), cell);
    auto projectile_momentum =
        group->get_cell(NP::Sym<REAL>("ION2_SOURCE_MOMENTUM"), cell);
    ASSERT_EQ(child_weights->nrow, weights->nrow);
    for (int row = 0; row < weights->nrow; ++row) {
      EXPECT_DOUBLE_EQ(weights->at(row, 0), 1.75);
      EXPECT_DOUBLE_EQ(child_weights->at(row, 0), 0.25);
      EXPECT_DOUBLE_EQ(velocities->at(row, 0), -1.0);
      EXPECT_DOUBLE_EQ(velocities->at(row, 1), 2.0);
      EXPECT_DOUBLE_EQ(target_density->at(row, 0), -0.25);
      EXPECT_DOUBLE_EQ(projectile_density->at(row, 0), 0.25);
      EXPECT_DOUBLE_EQ(projectile_momentum->at(row, 0), 2.25);
      EXPECT_DOUBLE_EQ(projectile_momentum->at(row, 1), -3.0);
    }
  }
  group->sycl_target->free();
  group->domain->mesh->free();
}
