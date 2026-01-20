inline void arrhenius_example() {

  // In case we wish to remap the default weight, fluid_temperature
  auto used_map = get_default_map();

  // The following will calculate the rate as 2 * T^3 * w
  auto arrhenius_data =
      ArrheniusData(2.0,       // The a coefficient in the Arrhenius formula
                    3.0,       // The b coefficient in the Arrhenius formula
                    used_map); // Optional property map
  return;
}
