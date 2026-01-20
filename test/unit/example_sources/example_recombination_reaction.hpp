inline void
recombination_reaction_example(ParticleGroupSharedPtr particle_group) {

  // In case we would like to remap the used Syms
  auto used_map = get_default_map();

  auto electron_species = Species("ELECTRON", 5.5e-4, -1.0);

  auto marker_species = Species(
      "ION", 2.0, 0.0,
      -1); // This is the target species, i.e. the ID corresponds to the
           // marker species and the species name corresponds to the ion fluid

  auto neutral_species = Species(
      "ION", 2.0, 0.0,
      0); // This is the recombined species, i.e. the ID corresponds to the
          // neutral species and the species name corresponds to the ion fluid

  auto reaction_rate = FixedRateData(1.0); // Reaction rate
                                           // NOTE: this would in reality
                                           // have to account for the background
                                           // electrons and ions

  // Following the recombination_kernels_example
  auto reaction_energy_rate = FixedRateData(1.0);
  auto ion_vel_x = FixedRateData(1.0);
  auto ion_vel_y = FixedRateData(2.0);
  auto normalised_potential_energy = 1.0;
  auto data_calculator =
      DataCalculator<decltype(reaction_energy_rate), decltype(ion_vel_x),
                     decltype(ion_vel_y)>(reaction_energy_rate, ion_vel_x,
                                          ion_vel_y);

  // Recombination with 2 velocity dimensions
  auto recombination_reaction =
      Recombination<decltype(reaction_rate), decltype(data_calculator), 2>(
          particle_group
              ->sycl_target, // Reactions need to know the used SYCL target
          reaction_rate,     // Reaction rate data object
          data_calculator, // Data calculator containing the energy rate and the
                           // velocity sampling
          marker_species,  // Marker pseudo-particle species - ingoing particle
                           // representing ions
          electron_species, // Electron species - will have energy loss set by
                            // given rate and potential energy
          neutral_species,  // Species into which the ions recombine - product
                            // particle
          normalised_potential_energy // Normalised ionisation potential energy
                                      // (to m_0 v_0^2)
      );

  return;
}
