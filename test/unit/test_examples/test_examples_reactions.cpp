#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "../example_sources/example_linear_reaction_CX.hpp"
#include "../example_sources/example_reaction_controller.hpp"
#include "../example_sources/example_reaction_data_accumulator_strategy.hpp"
#include "../example_sources/example_recombination_reaction.hpp"

TEST(Examples, reactions) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  linear_reaction_CX_example(particle_group);
  reaction_controller_example(particle_group);
  recombination_reaction_example(particle_group);
  reaction_data_accumulator_strategy_example(particle_group);

  particle_group->domain->mesh->free();
}
