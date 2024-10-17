inline void zeroer_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // A ParticleDatZeroer zeroes INT or REAL particle dats and is constructed 
  // by passing a vestor of strings with the dat names
  auto zeroer =
      make_transformation_strategy<ParticleDatZeroer<REAL>>(
          std::vector<std::string>{"ELECTRON_SOURCE_DENSITY",
                                   "ION_SOURCE_DENSITY"});

  zeroer->transform(input_subgroup);

  return;
}
