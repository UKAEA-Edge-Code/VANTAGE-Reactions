#include <reactions.hpp>
#include <neso_particles.hpp>
#include <gtest/gtest.h>
#include <mock_reactions.hpp>

using namespace NESO::Particles;
using namespace Reactions;

#include "./example_sources/example_marking_strategy.hpp"
#include "./example_sources/example_removal_strategy.hpp"
#include "./example_sources/example_zeroer_strategy.hpp"
#include "./example_sources/example_accumulator_strategy.hpp"
#include "./example_sources/example_composite_strategy.hpp"
#include "./example_sources/example_merging_strategy.hpp"
#include "./example_sources/example_transformation_wrapper.hpp"
#include "./example_sources/example_custom_properties.hpp"
#include "./example_sources/example_custom_property_map.hpp"
#include "./example_sources/example_property_container.hpp"
#include "./example_sources/example_spec_builder.hpp"
#include "./example_sources/example_linear_reaction_CX.hpp"
#include "./example_sources/example_fixed_coeff.hpp"
#include "./example_sources/example_amjuel1d.hpp"
#include "./example_sources/example_amjuel2d.hpp"
#include "./example_sources/example_amjuel2dH3.hpp"
#include "./example_sources/example_maxwellian_sampler.hpp"
#include "./example_sources/example_amjuel_cs.hpp"
#include "./example_sources/example_ionisation_kernels.hpp"
#include "./example_sources/example_electron_impact_ion.hpp"
#include "./example_sources/example_reaction_controller.hpp"
#include "./example_sources/example_new_reaction_data.hpp"

TEST(Examples, all){

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  marking_strategy_example(particle_group);
  removal_strategy_example(particle_group);
  zeroer_strategy_example(particle_group);
  accumulator_strategy_example(particle_group);
  composite_strategy_example(particle_group);
  merging_strategy_example(particle_group);
  transformation_wrapper_example(particle_group);
  custom_property_map_example();
  property_container_example();
  spec_builder_example();
  linear_reaction_CX_example(particle_group);
  fixed_rate_coeff_example();
  amjuel_1d_example();
  amjuel_2d_example();
  amjuel_2d_H3_example();
  maxwellian_sampler_example();
  amjuel_h1_cs_example();
  ionisation_kernels_example();
  electron_impact_ion_example(particle_group);
  reaction_controller_example(particle_group);

  particle_group->domain->mesh->free()
}
