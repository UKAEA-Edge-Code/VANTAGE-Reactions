inline void recombination_kernels_example() {

  // In case we would like to remap the used Syms
  auto used_map = get_default_map();

  auto electron_species = Species("ELECTRON", // name
                                  5.5e-4,        // electron mass in amu
                                  -1.0       // charge
  );
  auto target_species = Species("ION", 2.0, 0.0, -1); // This is the target species, i.e. the ID corresponds to the 
                                                     // marker species and the species name corresponds to the ion fluid
  auto reaction_energy_rate = FixedRateData(1.0); // Energy rate for 
                                                  // projectile energy loss 
                                                  // per recombination event 

  // Assuming the ions are a beam with given x and y speeds
  auto ion_vel_x = FixedRateData(1.0);
  auto ion_vel_y = FixedRateData(2.0);

  auto normalised_potential_energy = 1.0; // Set for convenience, otherwise should
                                         // be normalised to m_0 * v_0^2 
                                         // where m_0 is the mass normalisation (usually amu)
                                         // and v_0 is the velocity normalisation
  // Used data calculator
  auto data_calculator =
      DataCalculator<decltype(reaction_energy_rate), 
                     decltype(ion_vel_x),
                     decltype(ion_vel_y)>(reaction_energy_rate, 
                             ion_vel_x, 
                             ion_vel_y);

  auto recombination_kernels = RecombReactionKernels<2, // velocity dat dimensionality
                                           2 // momentum source dat dimensionality (defaults to the velocity dat dimensionality)
                                          >(
      target_species, // target species - marker species corresponding to the ions
      electron_species, // projectile species 
      normalised_potential_energy, // normalised ionisation potential to be included in projectile energy source
      used_map // Optional map for remapping property names
      );

  return;
}
