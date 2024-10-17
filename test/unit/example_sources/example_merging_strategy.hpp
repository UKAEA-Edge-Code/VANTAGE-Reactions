inline void merging_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // To specify the merging strategy one need only supply the position, weight,
  // and velocity/momentum Syms, as well as set the correct dimensionality in
  // the template argument
  auto merging_strat =
      make_transformation_strategy<MergeTransformationStrategy<2>>(
          Sym<REAL>("POSITION"), Sym<REAL>("WEIGHT"), Sym<REAL>("VELOCITY"));

  merging_strat->transform(input_subgroup);

  return;
}
