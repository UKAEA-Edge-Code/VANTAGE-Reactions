inline void amjuel_1d_example() {

  // In case we wish to remap the default weight, fluid_temperature,
  // fluid_density
  auto used_map = get_default_map();

  auto coeffs = std::array<REAL, 3>{1.0, 1.0, 1.0}; // b_n coefficients

  // We pass the number of fit coefficients to the constructor as a template
  // parameter
  auto amjuel_data =
      AMJUEL1DData<3>(1.0,       // The normalisation of the evolved quantity
                                 // (density, energy, particle weight, etc.)
                      1e19,      // Normalisation of density in m^{-3}
                      1.0,       // Temperature normalistion in eV
                      1e-8,      // Time normalisation in seconds
                      coeffs,    // fit coefficients
                      used_map); // Optional property map
  return;
}
