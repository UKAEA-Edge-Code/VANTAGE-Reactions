void merging_strategy_example(NP::ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  auto merging_strat =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  merging_strat->transform(input_subgroup);

  return;
}
