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
}
