inline void removal_strategy_example(ParticleGroupSharedPtr particle_group) {

  // Marking strategies take in ParticleSubgroup shared pointers.
  // We trivially produce a whole group subgroup pointer:
  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // As in the marking strategy example we mark low weight particles by first
  // creating the marking strategy
  auto mark_low_weight =
      make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
          Sym<REAL>("WEIGHT"), 1e-6);

  // And then applying it
  auto subgroup_low_weight =
      mark_low_weight->make_marker_subgroup(input_subgroup);

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
