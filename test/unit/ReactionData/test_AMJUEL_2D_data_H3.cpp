#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <cmath>
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

TEST(ReactionData, AMJUEL2DDataH3) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  REAL mass_amu = 1.0;
  REAL vel_norm = std::sqrt(
      2 * 1.60217663e-19 /
      (mass_amu * 1.66053904e-27)); // Makes the normalisation constant for the
                                    // energy equal to 1

  // Normalisation chosen to set the multiplicative constant in front of the
  // exp(sum...) to 1.0, assuming n = 3e18
  auto amjuel_data = AMJUEL2DDataH3<2, 2, 2>(
      3e12, 1.0, 1.0, 1.0, vel_norm, mass_amu,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}});

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DDataH3<2, 2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>());

  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  REAL logT = std::log(2);
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    auto vel = particle_group->get_cell(Sym<REAL>("VELOCITY"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      REAL logE = std::log(std::pow(vel->at(rowx, 0) - 1.0, 2) +
                           std::pow(vel->at(rowx, 1) - 3.0,
                                    2)); // Assuming vx = 1.0 and vy = 3.0
      REAL expected_rate = 1.0 + 0.02 * logE + 0.01 * logT + 0.02 * logE * logT;
      expected_rate = std::exp(expected_rate);
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate);
    }
  }

  particle_group->domain->mesh->free();
}