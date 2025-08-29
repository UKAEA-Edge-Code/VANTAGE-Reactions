inline void
accumulator_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // CellwiseAccumulators need to access some general data about the particle
  // group, so it needs to be available on construction
  //
  // Similarly to zeroers, accumulators take in the names of the particle dats
  // that they should accumulate values for cellwise. Here we use make_shared
  // instead of make_transformation_strategy in order to be able to call
  // accumulator-specific methods
  auto accumulator = std::make_shared<CellwiseAccumulator<REAL>>(
      particle_group, std::vector<std::string>{"ELECTRON_SOURCE_DENSITY",
                                               "ION_SOURCE_DENSITY"});

  // The accumulator, unlike other transforms, does not modify the particle
  // group. Instead, it modifies its own internal state.
  accumulator->transform(input_subgroup);

  // Upon accumulation, the accumulated data is stored in NESO-Particle
  // CellDatConst objects and can be retrieved easily
  auto accumulated_electron_source =
      accumulator->get_cell_data("ELECTRON_SOURCE_DENSITY");

  // Individual cell data buffers can be zeroed
  accumulator->zero_buffer("ELECTRON_SOURCE_DENSITY");

  // Or all buffers can be zeroed
  accumulator->zero_all_buffers();

  // A weighted accumulator transform is also available, taking in the REAL
  // particle dat to be used as the weight (should be a PartilceDat with 1
  // component) in the constructor
  auto weighted_accumulator =
      std::make_shared<WeightedCellwiseAccumulator<REAL>>(
          particle_group, std::vector<std::string>{"VELOCITY", "POSITION"},
          "WEIGHT");

  weighted_accumulator->transform(input_subgroup);

  auto weighted_velocities = weighted_accumulator->get_cell_data("VELOCITY");
  weighted_accumulator->zero_buffer("VELOCITY");

  // In addition to the standard accumulator methods, the weighted accumulator
  // also offers access to the accumulated weight dat
  auto accumulated_weight = weighted_accumulator->get_weight_cell_data();

  return;
}
