inline void ionisation_kernels_example() {

  // In case we would like to remap the used Syms
  auto used_map = default_map;

  auto electron_species = Species("ELECTRON", // name
                                  5.5e-4,        // electron mass in amu
                                  -1.0       // charge
  );
  auto target_species = Species("ION", 2.0, 0.0, 1); // This is the target species, i.e. the ID corresponds to the 
                                                     // neutral being ionised and the species name corresponds to the ion fluid 
  auto ion_kernels = IoniseReactionKernels<2, // velocity dat dimensionality
                                           2, // momentum source dat dimensionality (defaults to the velocity dat dimensionality)
                                           false // set to true if there is an expected momentum source rate data in the data calculator // 
                                          >(
      target_species, // target species - neutral species and resulting ion species
      electron_species, // electron species - in general used only to store a density source
      electron_species, // projectile species - energy and momentum sources (so here electrons will have an energy and a particle source contribution)
      used_map // Optional map for remapping property names
      );

  return;
}
