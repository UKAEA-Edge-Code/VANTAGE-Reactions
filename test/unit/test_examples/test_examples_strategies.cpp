#include "../include/mock_particle_group.hpp"
#include "../include/test_common.hpp"

using namespace VANTAGE::Reactions;

#include "../example_sources/example_accumulator_strategy.hpp"
#include "../example_sources/example_cellwise_distributor_strategy.hpp"
#include "../example_sources/example_composite_strategy.hpp"
#include "../example_sources/example_direct_transformations.hpp"
#include "../example_sources/example_marking_strategy.hpp"
#include "../example_sources/example_merging_strategy.hpp"
#include "../example_sources/example_removal_strategy.hpp"
#include "../example_sources/example_simple_thinning_strategy.hpp"
#include "../example_sources/example_transformation_wrapper.hpp"
#include "../example_sources/example_uniform_velocity_binning.hpp"
#include "../example_sources/example_vranic_merging_strategy.hpp"
#include "../example_sources/example_zeroer_strategy.hpp"

TEST(Examples, strategies) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  removal_strategy_example(particle_group);
  zeroer_strategy_example(particle_group);
  accumulator_strategy_example(particle_group);
  composite_strategy_example(particle_group);
  merging_strategy_example(particle_group);
  transformation_wrapper_example(particle_group);
  distributor_strategy_example(particle_group);
  direct_marking_example(particle_group);
  direct_transformation_example(particle_group);

  particle_group->add_particle_dat(NP::Sym<INT>("REACTIONS_GROUPING_INDEX"), 1);
  particle_group->add_particle_dat(NP::Sym<INT>("REACTIONS_LINEAR_INDEX"), 1);
  vranic_merging_strategy_example(particle_group);
  simple_thinning_strategy_example(particle_group);
  uniform_velocity_binning_example(particle_group);

  particle_group->domain->mesh->free();
}
