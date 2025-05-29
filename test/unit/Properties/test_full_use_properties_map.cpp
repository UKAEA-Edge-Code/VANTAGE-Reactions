#include "../include/mock_particle_group.hpp"
#include "../include/mock_reactions.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(Properties, full_use_properties_map) {
  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);
  int cell_count = particle_group->domain->mesh->get_cell_count();

  auto test_prop1 = ParticleProp(Sym<REAL>("TEST_PROP1"), 1);
  auto test_prop2 = ParticleProp(Sym<REAL>("TEST_PROP2"), 1);

  auto test_prop1_dat = ParticleDat<REAL>(
      particle_group->sycl_target, Sym<REAL>("TEST_PROP1"), 1, cell_count);
  auto test_prop2_dat = ParticleDat<REAL>(
      particle_group->sycl_target, Sym<REAL>("TEST_PROP2"), 1, cell_count);

  particle_group->add_particle_dat(test_prop1_dat);
  particle_group->add_particle_dat(test_prop2_dat);

  particle_loop(
      particle_group,
      [=](auto test_prop1_write, auto test_prop2_write) {
        test_prop1_write[0] = 3.0e18;
        test_prop2_write[0] = 2.0;
      },
      Access::write(Sym<REAL>("TEST_PROP1")),
      Access::write(Sym<REAL>("TEST_PROP2")))
      ->execute();

  particle_group->remove_particle_dat(Sym<REAL>("FLUID_DENSITY"));
  particle_group->remove_particle_dat(Sym<REAL>("FLUID_TEMPERATURE"));

  auto test_prop_map = get_default_map();
  test_prop_map[default_properties.fluid_density] = "TEST_PROP1";
  test_prop_map[default_properties.fluid_temperature] = "TEST_PROP2";

  auto particle_sub_group = std::make_shared<ParticleSubGroup>(particle_group);

  auto particle_spec = particle_group->get_particle_spec();

  auto amjuel_data = AMJUEL2DData<2, 2>(
      3e12, 1.0, 1.0, 1.0,
      std::array<std::array<REAL, 2>, 2>{std::array<REAL, 2>{1.0, 0.02},
                                         std::array<REAL, 2>{0.01, 0.02}},
      test_prop_map);

  auto test_reaction =
      LinearReactionBase<0, AMJUEL2DData<2, 2>, TestReactionKernels<0>>(
          particle_group->sycl_target, 0, std::array<int, 0>{}, amjuel_data,
          TestReactionKernels<0>(test_prop_map));

  auto descendant_particles = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_group->get_particle_spec(),
      particle_group->sycl_target);

  // Rate calculated based on ne=3e18, T=2eV, with the evolved quantity
  // normalised to 3e12
  auto expected_rate = 3.880728735562758;
  for (int i = 0; i < cell_count; i++) {

    test_reaction.run_rate_loop(particle_sub_group, i, i + 1);
    auto rate = particle_group->get_cell(Sym<REAL>("TOT_REACTION_RATE"), i);
    const int nrow = rate->nrow;

    for (int rowx = 0; rowx < nrow; rowx++) {
      EXPECT_DOUBLE_EQ(rate->at(rowx, 0), expected_rate); // 1e-15);
    }
  }

  particle_group->domain->mesh->free();
}