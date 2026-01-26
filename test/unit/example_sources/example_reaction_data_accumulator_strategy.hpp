inline void reaction_data_accumulator_strategy_example(
    ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = particle_sub_group(particle_group);

  // For the reaction data accumulator, we need to define the reaction
  // data object whose results should be accumulated cellwise
  //
  // Here we use extractors and reaction data arithmetic to accumulate
  // weighted kinetic energy in each velocity dimension

  auto weight = extract<1>("WEIGHT");
  auto velocity = extract<2>("VELOCITY");

  auto kin_energy = weight * velocity * velocity;

  // The accumulator is then defined simply as

  auto accumulator =
      std::make_shared<CellwiseReactionDataAccumulator<decltype(kin_energy)>>(
          particle_group, kin_energy);

  // The accumulator can then be called like any other transform
  accumulator->transform(input_subgroup);

  // Upon accumulation, the accumulated data is stored in NESO-Particle
  // CellDatConst objects and can be retrieved easily
  auto accumulated_kin_energy = accumulator->get_cell_data();

  // The buffer can be zeroed as
  accumulator->zero_buffer();

  return;
}
