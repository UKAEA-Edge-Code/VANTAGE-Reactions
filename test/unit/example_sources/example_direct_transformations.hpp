#include "example_common_functors.hpp"

void direct_transformation_example(ParticleGroupSharedPtr particle_group) {

  auto subgroup_low_weight = particle_sub_group(
      particle_group, LowWeightMark{}, Access::read(Sym<REAL>("WEIGHT")));

  // We can recreate the built-in removal strategy using a direct transformation
  // strategy

  auto removal_strategy_direct = make_lambda_transformation_strategy(
      "removal_lambda", // The name for this transformation strategy
      RemoveSubgroupTransform{}
      // The functor to be applied to any passed subgroup
  );

  removal_strategy_direct->transform(subgroup_low_weight);

  // Or we can apply a particle loop as a transformation strategy

  auto set_id_strategy = make_direct_transformation_strategy(
      "set_id_0",                   // Name of the strategy
      SetIdToZeroKernel{},          // The particle loop kernel to be applied
      Access::write(Sym<INT>("ID")) // Accessors for the loop
  );

  // Apply to the particle_group
  set_id_strategy->transform(particle_sub_group(particle_group));

  return;
}
