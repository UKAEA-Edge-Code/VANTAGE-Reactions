inline void marking_strategy_example(ParticleGroupSharedPtr particle_group) {

  // Marking strategies take in ParticleSubgroup shared pointers. 
  // We trivially produce a whole group subgroup pointer:
  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // To make a marking strategy that will mark all particles with the real-valued
  // particle dat "WEIGHT" < 1e-6 we use the ComparisonMarkerSingle strategy,
  // comparing a single REAL value with a less-than comparison function (LessThanComp is a simple device-safe wrapper)
  // 
  // The make_marking_strategy helper function casts marking strategies into 
  // a std::shared_ptr<MarkingStrategy>.
  //
  // The first argument in the case of the ComparisonMarkersSingle strategy is the single
  // ParticleDat name (as a NESO-Particles Sym) and the second is the fixed value these dats
  // will be compared against.
  auto mark_low_weight =
      make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
          Sym<REAL>("WEIGHT"), 1e-6);

  // Here we make a new marking strategy, which will mark all particles with ID=0
  auto mark_id_zero =
      make_marking_strategy<ComparisonMarkerSingle<INT, EqualsComp>>(
          Sym<INT>("ID"), 0);

  // The resulting subgroup will have only particles with ID=0
  auto subgroup_only_id_zero =
      mark_id_zero->make_marker_subgroup(input_subgroup);

  // The following subgroup will have only particles with ID=0 and WEIGHT<1e-6
  auto subgroup_id_zero_and_low_weight =
      mark_low_weight->make_marker_subgroup(subgroup_only_id_zero);

  return;
}
