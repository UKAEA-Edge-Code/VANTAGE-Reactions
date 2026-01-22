inline void composite_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  // We wish to compose an accumulator and a particle dat zeroer to accumulate
  // some sources and then to reset the particle data that stored them
  //
  // The sources we wish to accumulate
  auto source_names =
      std::vector<std::string>{"ELECTRON_SOURCE_DENSITY", "ION_SOURCE_DENSITY"};

  // In order to have later access to the accumulator, we construct it using
  // make_shared
  auto accumulator =
      std::make_shared<CellwiseAccumulator<REAL>>(particle_group, source_names);

  // A composite transform can be constructed by passing a vector of
  // TransformationStrategy objects, so if we wish to include the accumulator,
  // it must be dynamically cast
  auto composite = std::make_shared<CompositeTransform>(
      std::vector<std::shared_ptr<TransformationStrategy>>{
          std::dynamic_pointer_cast<TransformationStrategy>(accumulator)});

  // We can also directly add TransformationStrategy objects to the composite to
  // be applied in sequence (in order of addition)

  composite->add_transformation(
      make_transformation_strategy<ParticleDatZeroer<REAL>>(source_names));

  // The composite can then be applied as one transformation
  composite->transform(input_subgroup);

  // Since the accumulator was added to the composite via a pointer
  // its interface is still accessible

  auto accumulated_electron_source =
      accumulator->get_cell_data("ELECTRON_SOURCE_DENSITY");

  accumulator->zero_all_buffers();

  return;
}
