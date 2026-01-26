inline void
distributor_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = particle_sub_group(particle_group);

  // CellwiseDistributors need to access some general data about the particle
  // group, so it needs to be available on construction
  //
  // Similarly to zeroers, distributors take in the names of the particle dats
  // that they should distribute values for cellwise. Here we use make_shared
  // instead of make_transformation_strategy in order to be able to call
  // distributor-specific methods
  auto distributor = std::make_shared<CellwiseAccumulator<REAL>>(
      particle_group, std::vector<std::string>{"ELECTRON_SOURCE_DENSITY",
                                               "ION_SOURCE_DENSITY"});

  // To set the values for distribution we can do the following

  auto buffer = distributor->get_cell_data("ELECTRON_SOURCE_DENSITY");
  auto num_cells = particle_group->domain->mesh->get_cell_count();

  for (int cellx = 0; cellx < num_cells; cellx++) {

    buffer[cellx]->at(0, 0) = 1.0; // at 0th component and 0th layer (see
                                   // NESO-Particles documentation)
  }
  distributor->set_cell_data("ELECTRON_SOURCE_DENSITY", buffer);
  distributor->transform(input_subgroup);

  // Distributor buffer zeroing, if needed, can be done in the same way as for
  // the accumulator
  distributor->zero_buffer("ELECTRON_SOURCE_DENSITY");

  distributor->zero_all_buffers();

  return;
}
