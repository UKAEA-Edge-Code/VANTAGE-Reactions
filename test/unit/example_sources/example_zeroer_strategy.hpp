void zeroer_strategy_example(NP::ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<NP::ParticleSubGroup>(particle_group);

  // A ParticleDatZeroer zeroes NP::INT or NP::REAL particle dats and is
  // constructed by passing a vestor of strings with the dat names
  auto zeroer = make_transformation_strategy<ParticleDatZeroer<NP::REAL>>(
      std::vector<std::string>{"ELECTRON_SOURCE_DENSITY",
                               "ION_SOURCE_DENSITY"});

  zeroer->transform(input_subgroup);

  return;
}
