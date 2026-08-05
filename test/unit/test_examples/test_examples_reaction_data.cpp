#include "../include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "../example_sources/example_amjuel1d.hpp"
#include "../example_sources/example_amjuel2d.hpp"
#include "../example_sources/example_amjuel2dH3.hpp"
#include "../example_sources/example_amjuel_cs.hpp"
#include "../example_sources/example_array_lookup.hpp"
#include "../example_sources/example_arrhenius.hpp"
#include "../example_sources/example_binary_array_transform_data.hpp"
#include "../example_sources/example_concatenator.hpp"
#include "../example_sources/example_extractor.hpp"
#include "../example_sources/example_fixed_coeff.hpp"
#include "../example_sources/example_interpolation.hpp"
#include "../example_sources/example_lambda_wrapper_array_transform_data.hpp"
#include "../example_sources/example_maxwellian_sampler.hpp"
#include "../example_sources/example_pipeline.hpp"
#include "../example_sources/example_sampler.hpp"
#include "../example_sources/example_trim_interpolation.hpp"
#include "../example_sources/example_unary_array_transform_data.hpp"

TEST(Examples, reaction_data) {

  const int N_total = 1000;

  auto particle_group = create_test_particle_group(N_total);

  fixed_rate_coeff_example();
  amjuel_1d_example();
  amjuel_2d_example();
  amjuel_2d_H3_example();
  amjuel_h1_cs_example();
  maxwellian_sampler_example();
  arrhenius_example();
  sampler_example();
  array_lookup_example(particle_group);
  extractor_example();
  concatenator_example();
  pipeline_example();
  unary_array_transform_examples();
  binary_array_transform_examples();
  lambda_wrapper_array_transform_examples();
  interpolation_example(particle_group);
  trim_interpolation_example(particle_group);

  particle_group->domain->mesh->free();
}
