#include "example_common_functors.hpp"

void removal_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto subgroup_low_weight = particle_sub_group(
      particle_group, LowWeightMark{}, Access::read(Sym<REAL>("WEIGHT")));

  // The make_transformation_strategy helper function casts concrete
  // transformation strategies into std::shared_ptr<TransformationStrategy>.
  //
  // The simple removal strategy below requires no inputs, and just deletes
  // all particles in the passed subgroup
  auto removal_strategy =
      make_transformation_strategy<SimpleRemovalTransformationStrategy>();

  removal_strategy->transform(subgroup_low_weight);

  return;
}
