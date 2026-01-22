inline void amjuel_2d_example() {

  // In case we wish to remap the default weight, fluid_temperature,
  // fluid_density
  auto used_map = get_default_map();

  // alpha coefficients (the inner array size is the number of density
  // coefficients, the outer is temperature)
  auto coeffs = std::array<std::array<REAL, 2>, 2>{
      std::array<REAL, 2>{1.0, 0.02}, std::array<REAL, 2>{0.01, 0.02}};

  // We pass the number of fit coefficients to the constructor as a template
  // parameter
  auto amjuel_data =
      AMJUEL2DData<2, 2>(1.0,       // The normalisation of the evolved quantity
                                    // (density, energy, particle weight, etc.)
                         1e19,      // Normalisation of density in m^{-3}
                         1.0,       // Temperature normalistion in eV
                         1e-8,      // Time normalisation in seconds
                         coeffs,    // fit coefficients
                         used_map); // Optional property map
  return;
}
