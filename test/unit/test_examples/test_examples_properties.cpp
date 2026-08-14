#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "../example_sources/example_custom_properties.hpp"
#include "../example_sources/example_custom_property_map.hpp"
#include "../example_sources/example_property_container.hpp"
#include "../example_sources/example_spec_builder.hpp"

TEST(Examples, properties) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  custom_property_map_example();
  property_container_example();
  spec_builder_example();

  particle_group->domain->mesh->free();
}
