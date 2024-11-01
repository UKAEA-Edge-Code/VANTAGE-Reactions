inline void fixed_rate_coeff_example() {

  // In case we would like to remap the used weight Sym
  auto used_map = default_map;

  auto fixed_coeff_rate = FixedCoefficientData(5.0, // K - fixed rate coefficient
                                               used_map);

  return;
}
