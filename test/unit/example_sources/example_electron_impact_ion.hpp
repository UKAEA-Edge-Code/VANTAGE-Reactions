inline void electron_impact_ion_example(ParticleGroupSharedPtr particle_group) {

  auto used_map = default_map;

  auto electron_species = Species("ELECTRON", // name
                                  5.5e-4,        // electron mass in amu
                                  -1.0       // charge
  );
  auto target_species = Species("ION", 2.0, 0.0, 1); // This is the target species, i.e. the ID corresponds to the 
                                                     // neutral being ionised and the species name corresponds to the ion fluid 
  auto test_data = FixedRateData(1.0); // Example fixed rate data
  auto ionise_reaction = ElectronImpactIonisation<FixedRateData, //Reaction data class used for the reaction rate
                                                  FixedRateData // Reaction data class used for the energy rate
                                                 >(
      particle_group->sycl_target, // Reactions need access to the SYCL target 
      Sym<REAL>(used_map[default_properties.tot_reaction_rate]), // Total reaction data sym used to construct the reaction 
      Sym<REAL>(used_map[default_properties.weight]), // Particle weight sym 
      test_data, // Reaction rate data
      test_data, // Energy rate data
      target_species, // Ionisation target species
      electron_species, // Electron species object (projectile) 
      particle_group->particle_spec // Particle spec needed in the construction of reaction data
      );
  return;
}
