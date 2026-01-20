#include "include/mock_particle_group.hpp"
#include <gtest/gtest.h>

using namespace NESO::Particles;
using namespace VANTAGE::Reactions;

#include "./example_sources/example_accumulator_strategy.hpp"
#include "./example_sources/example_amjuel1d.hpp"
#include "./example_sources/example_amjuel2d.hpp"
#include "./example_sources/example_amjuel2dH3.hpp"
#include "./example_sources/example_amjuel_cs.hpp"
#include "./example_sources/example_array_lookup.hpp"
#include "./example_sources/example_arrhenius.hpp"
#include "./example_sources/example_binary_array_transform_data.hpp"
#include "./example_sources/example_cartesian_basis_reflection.hpp"
#include "./example_sources/example_composite_strategy.hpp"
#include "./example_sources/example_concatenator.hpp"
#include "./example_sources/example_custom_properties.hpp"
#include "./example_sources/example_custom_property_map.hpp"
#include "./example_sources/example_cx_kernel_definition.hpp"
#include "./example_sources/example_electron_impact_ion.hpp"
#include "./example_sources/example_extractor.hpp"
#include "./example_sources/example_fixed_coeff.hpp"
#include "./example_sources/example_general_absorption_kernels.hpp"
#include "./example_sources/example_general_linear_scattering_kernels.hpp"
#include "./example_sources/example_ionisation_kernels.hpp"
#include "./example_sources/example_lambda_wrapper_array_transform_data.hpp"
#include "./example_sources/example_linear_reaction_CX.hpp"
#include "./example_sources/example_marking_strategy.hpp"
#include "./example_sources/example_maxwellian_sampler.hpp"
#include "./example_sources/example_merging_strategy.hpp"
#include "./example_sources/example_new_reaction_data.hpp"
#include "./example_sources/example_pipeline.hpp"
#include "./example_sources/example_property_container.hpp"
#include "./example_sources/example_reaction_controller.hpp"
#include "./example_sources/example_reaction_data_accumulator_strategy.hpp"
#include "./example_sources/example_recombination_kernels.hpp"
#include "./example_sources/example_recombination_reaction.hpp"
#include "./example_sources/example_removal_strategy.hpp"
#include "./example_sources/example_sampler.hpp"
#include "./example_sources/example_spec_builder.hpp"
#include "./example_sources/example_spherical_basis_reflection.hpp"
#include "./example_sources/example_transformation_wrapper.hpp"
#include "./example_sources/example_unary_array_transform_data.hpp"
#include "./example_sources/example_zeroer_strategy.hpp"

TEST(Examples, all) {

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
  recombination_kernels_example();
  recombination_reaction_example(particle_group);
  arrhenius_example();
  sampler_example();
  array_lookup_example(particle_group);
  extractor_example();
  concatenator_example();
  pipeline_example();
  unary_array_transform_examples();
  binary_array_transform_examples();
  lambda_wrapper_array_transform_examples();
  cartesian_basis_reflection_example();
  spherical_basis_reflection_example();
  general_absorption_kernels_example();
  general_linear_scattering_kernels_example(particle_group);
  reaction_data_accumulator_strategy_example(particle_group);

  particle_group->domain->mesh->free();
}
