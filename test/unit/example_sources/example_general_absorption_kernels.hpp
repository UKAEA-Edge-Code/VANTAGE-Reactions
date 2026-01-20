inline void general_absorption_kernels_example() {

  // For this kernel, we do need to remap the sources in general
  // For example we might want to use these kernels to represent absorption at a
  // surface
  auto properties_map = PropertiesMap();

  properties_map[VANTAGE::Reactions::default_properties.source_density] =
      "SURFACE_SOURCE_DENSITY";
  properties_map[VANTAGE::Reactions::default_properties.source_momentum] =
      "SURFACE_SOURCE_DENSITY";
  properties_map[VANTAGE::Reactions::default_properties.source_energy] =
      "SURFACE_SOURCE_DENSITY";

  auto target_species = Species("ION", 2.0, 0.0,
                                -1); // This is absorbed species

  auto absorption_kernels =
      GeneralAbsorptionKernels<2 // velocity dat dimensionality
                               >(
          target_species,          // Absorbed species
          properties_map.get_map() // Our remapped sources
      );

  return;
}
