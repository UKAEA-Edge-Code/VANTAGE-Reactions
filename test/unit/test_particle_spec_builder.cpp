#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace Reactions;

TEST(ParticleSpecBuilder, add_particle_prop) {

  auto basic_spec = ParticleSpec{ParticleProp(Sym<REAL>("POSITION"), 2, true),
                                 ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto test_particle_spec_builder = ParticleSpecBuilder(basic_spec);

  auto props = default_properties;

  auto position_prop = Properties<REAL>(std::vector<int>{props.position});
  test_particle_spec_builder.add_particle_prop(position_prop, 2, true);

  auto cell_id_prop = Properties<INT>(std::vector<int>{props.cell_id});
  test_particle_spec_builder.add_particle_prop(cell_id_prop, 1, true);

  int num_position_props = 0;
  int num_cell_id_props = 0;
  // Check that adding "POSITION" doesn't change the number of instances of
  // "POSITION" inside test_particle_spec_builder.particle_spec
  for (auto prop :
       test_particle_spec_builder.get_particle_spec().properties_real) {
    if (prop == ParticleProp(Sym<REAL>("POSITION"), 2, true)) {
      num_position_props++;
    }
  }
  EXPECT_EQ(num_position_props, 1);

  // Check that adding "CELL_ID" doesn't change the number of instances of
  // "CELL_ID" inside test_particle_spec_builder.particle_spec
  for (auto prop :
       test_particle_spec_builder.get_particle_spec().properties_int) {
    if (prop == ParticleProp(Sym<INT>("CELL_ID"), 1, true)) {
      num_cell_id_props++;
    }
  }
  EXPECT_EQ(num_cell_id_props, 1);

  // General add_particle_prop test
  auto internal_state_prop = ParticleProp(Sym<INT>("INTERNAL_STATE"), 1);
  auto weight_prop = ParticleProp(Sym<REAL>("w"), 1);
  auto electron_temp_prop = ParticleProp(Sym<REAL>("ELECTRON_TEMPERATURE"), 1);

  auto int_props = Properties<INT>(std::vector<int>{props.internal_state});
  auto real_props = Properties<REAL>(std::vector<int>{props.weight},
                                     std::vector<Species>{Species("ELECTRON")},
                                     std::vector<int>{props.temperature});

  test_particle_spec_builder.add_particle_prop(int_props);

  auto new_map = default_map;
  new_map[props.weight] = "w";
  test_particle_spec_builder.add_particle_prop(real_props,1,false,new_map);

  auto test_particle_spec = test_particle_spec_builder.get_particle_spec();

  EXPECT_EQ(test_particle_spec.contains(internal_state_prop), true);
  EXPECT_EQ(test_particle_spec.contains(weight_prop), true);
  EXPECT_EQ(test_particle_spec.contains(electron_temp_prop), true);
}

TEST(ParticleSpecBuilder, add_particle_spec) {
  auto particle_group = create_test_particle_group(100);

  auto particle_group_spec = particle_group->get_particle_spec();

  auto test_particle_spec_builder = ParticleSpecBuilder();

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

  EXPECT_EQ(particle_group_spec.properties_int,
            test_particle_spec_builder.get_particle_spec().properties_int);
  EXPECT_EQ(particle_group_spec.properties_real,
            test_particle_spec_builder.get_particle_spec().properties_real);

  particle_group->domain->mesh->free();
}
