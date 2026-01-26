inline void general_linear_scattering_kernels_example(
    ParticleGroupSharedPtr particle_group) {

  // For this kernel, we do need to remap the sources in general
  // For example we might want to use these kernels to represent absorption at a
  // surface
  //
  // In this example we will indeed set up a specular reflection kernel using
  // the general linear scattering kernel and specular reflection data
  auto properties_map = PropertiesMap();

  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "SURFACE_SOURCE_DENSITY";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "SURFACE_SOURCE_DENSITY";

  auto projectile_species = Species("ION", 2.0, 0.0,
                                    -1); // This is the projectile species

  // Implementing specular reflection on ingoing particle velocities
  auto velocity_data = ExtractorData<2>(Sym<REAL>("VELOCITY"));

  auto specular_reflection = SpecularReflectionData<2>();

  auto pipeline = pipe(velocity_data, specular_reflection);
  // Wrapping the pipeline in a DataCalculator
  auto data_calculator = DataCalculator<decltype(pipeline)>(pipeline);

  auto scattering_kernel =
      LinearScatteringKernels<2 // velocity dat dimensionality
                              >(projectile_species,      // Scattered species
                                properties_map.get_map() // Our remapped sources
      );

  // The above can then be used to creat a specular reflection reaction
  // that can then be applied on particles that have hit a surface
  //
  // Here we set a constant rate and reflect the particle in the same internal
  // state

  auto specular_reflection_reaction =
      LinearReactionBase<1, FixedRateData, decltype(scattering_kernel),
                         decltype(data_calculator)>(
          particle_group->sycl_target, 0, std::array<int, 1>{0},
          FixedRateData(1.0), scattering_kernel, data_calculator);

  return;
}
