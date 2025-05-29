#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ParticleSpecBuilder, add_particle_spec) {
  auto particle_group = create_test_particle_group(100);

  auto particle_group_spec = particle_group->get_particle_spec();

  auto test_particle_spec_builder = ParticleSpecBuilder(2);

  test_particle_spec_builder.add_particle_spec(particle_group_spec);

  int num_position_props = 0;
  int num_cell_id_props = 0;
  for (auto prop :
       test_particle_spec_builder.get_particle_spec().properties_real) {
    if (prop == ParticleProp(Sym<REAL>("POSITION"), 2, true)) {
      num_position_props++;
    }
  }
  for (auto prop :
       test_particle_spec_builder.get_particle_spec().properties_int) {
    if (prop == ParticleProp(Sym<INT>("CELL_ID"), 1, true)) {
      num_cell_id_props++;
    }
  }
  EXPECT_EQ(num_position_props, 1);
  EXPECT_EQ(num_cell_id_props, 1);

  auto test_particle_spec = test_particle_spec_builder.get_particle_spec();

  for (auto prop : particle_group_spec.properties_int) {
    EXPECT_EQ(test_particle_spec.contains(prop), true);
  }

  for (auto prop : particle_group_spec.properties_real) {
    EXPECT_EQ(test_particle_spec.contains(prop), true);
  }

  particle_group->domain->mesh->free();
}