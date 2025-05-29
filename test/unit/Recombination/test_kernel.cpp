#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(Recombination, kernel_test) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  particle_loop(
      particle_group, [=](auto internal_state) { internal_state.at(0) = -1; },
      Access::write(Sym<INT>("INTERNAL_STATE")))
      ->execute();

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();
  auto marker_species = Species("ION", 1.0, 0.0, -1);
  auto neutral_species = Species("ION", 1.0, 0.0, 0);

  auto test_data = FixedRateData(1.0);
  auto test_data_2 = FixedRateData(2.0);

  auto test_data_calc = DataCalculator<decltype(test_data), decltype(test_data),
                                       decltype(test_data_2)>(
      test_data, test_data, test_data_2);

  auto test_normalised_potential_energy = 13.6;

  auto test_reaction =
      Recombination<decltype(test_data), decltype(test_data_calc)>(
          particle_group->sycl_target, test_data, test_data_calc,
          marker_species, Species("ELECTRON"), neutral_species,
          test_normalised_potential_energy);

  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_spec, particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.descendant_product_loop(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);

    auto weight_child = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);

    auto id_child =
        descendant_particles->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    auto target_source_density =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_DENSITY"), i);
    auto projectile_source_density =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_DENSITY"), i);

    auto target_source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto target_source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    auto projectile_source_energy =
        particle_group->get_cell(Sym<REAL>("ELECTRON_SOURCE_ENERGY"), i);

    const int nrow = weight_child->nrow;
    const int parent_nrow = vel_parent->nrow;

    EXPECT_EQ(nrow, parent_nrow);

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(weight_child->at(rowx, 0), 0.1);
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 0), 1.0);
      EXPECT_DOUBLE_EQ(vel_child->at(rowx, 1), 2.0);
      EXPECT_EQ(id_child->at(rowx, 0), 0);
      EXPECT_DOUBLE_EQ(target_source_density->at(rowx, 0), -0.1);
      EXPECT_DOUBLE_EQ(projectile_source_density->at(rowx, 0), -0.1);
      EXPECT_DOUBLE_EQ(target_source_momentum->at(rowx, 0), -0.1);
      EXPECT_DOUBLE_EQ(target_source_momentum->at(rowx, 1), -0.2);
      EXPECT_DOUBLE_EQ(target_source_energy->at(rowx, 0), -0.25);
      EXPECT_DOUBLE_EQ(projectile_source_energy->at(rowx, 0),
                       -(0.1 - (test_normalised_potential_energy * 0.1)));
    }
  }

  particle_group->domain->mesh->free();
}
