#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ChargeExchange, sampled_beam_exchange_2D) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto target_species = Species("ION2", 2.0, 0.0, 1);

  auto rng_lambda = [&]() -> REAL { return 0.25; };
  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1000);

  auto test_reaction =
      LinearReactionBase<1, FixedRateData, CXReactionKernels<>,
                         DataCalculator<FilteredMaxwellianSampler<2>>>(
          particle_group->sycl_target, 0, std::array<int, 1>{1},
          FixedRateData(1.0),
          CXReactionKernels<>(target_species, projectile_species),
          DataCalculator<FilteredMaxwellianSampler<2>>(
              FilteredMaxwellianSampler<2>(2.0, rng_kernel)));

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    test_reaction.descendant_product_loop(particle_sub_group, i, i + 1, 0.1,
                                          descendant_particles);

    auto weight = descendant_particles->get_cell(Sym<REAL>("WEIGHT"), i);
    auto vel_parent = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    auto vel_child = descendant_particles->get_cell(Sym<REAL>("VELOCITY"), i);
    auto id_child =
        descendant_particles->get_cell(Sym<INT>("INTERNAL_STATE"), i);

    auto target_source =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_DENSITY"), i);
    auto projectile_source =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_DENSITY"), i);

    auto target_source_momentum =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_MOMENTUM"), i);
    auto projectile_source_momentum =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_MOMENTUM"), i);

    auto target_source_energy =
        particle_group->get_cell(Sym<REAL>("ION2_SOURCE_ENERGY"), i);
    auto projectile_source_energy =
        particle_group->get_cell(Sym<REAL>("ION_SOURCE_ENERGY"), i);
    const int nrow = weight->nrow;
    const int parent_nrow = vel_parent->nrow;

    EXPECT_EQ(nrow, parent_nrow);

    REAL expected_vel_x = 1.0; // u1 = 0.25 => cos(2*pi*u1) = 0, v_x = 1
    REAL expected_vel_y = 4.0 * std::sqrt(2 * std::log(4)) +
                          3.0; // norm_ratio=2, T =2, v_y = 3, u2 = 0.25

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_NEAR(weight->at(rowx, 0), 0.1, 1e-14);
      EXPECT_NEAR(vel_child->at(rowx, 0), expected_vel_x, 1e-14);
      EXPECT_NEAR(vel_child->at(rowx, 1), expected_vel_y, 1e-14);
      EXPECT_EQ(id_child->at(rowx, 0), 1);
      EXPECT_NEAR(target_source->at(rowx, 0), -0.1, 1e-14);
      EXPECT_NEAR(projectile_source->at(rowx, 0), 0.1, 1e-14);
      EXPECT_NEAR(target_source_momentum->at(rowx, 0),
                  -0.1 * 2 * expected_vel_x, 1e-14);
      EXPECT_NEAR(target_source_momentum->at(rowx, 1),
                  -0.1 * 2 * expected_vel_y, 1e-14);
      EXPECT_NEAR(projectile_source_momentum->at(rowx, 0),
                  0.1 * 1.2 * vel_parent->at(rowx, 0), 1e-14);
      EXPECT_NEAR(projectile_source_momentum->at(rowx, 1),
                  0.1 * 1.2 * vel_parent->at(rowx, 1), 1e-14);
      EXPECT_NEAR(
          target_source_energy->at(rowx, 0),
          -0.1 * (std::pow(expected_vel_x, 2) + std::pow(expected_vel_y, 2)),
          1e-14); // w*m*vi^2 / 2
      EXPECT_NEAR(projectile_source_energy->at(rowx, 0),
                  0.1 * 0.6 *
                      (std::pow(vel_parent->at(rowx, 0), 2) +
                       std::pow(vel_parent->at(rowx, 1), 2)),
                  1e-14); // w*m*v^2 / 2
    }
  }

  particle_group->domain->mesh->free();
}