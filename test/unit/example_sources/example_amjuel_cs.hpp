inline void amjuel_h1_cs_example() {

  // These are example values
  auto coeffs = std::array<REAL, 3>{1.0, 1.0, 1.0}; // a_n coefficients
  auto l_coeffs =
      std::array<REAL, 3>{1.0, 1.0, 1.0}; // left asymptote a_n coefficients
  auto r_coeffs =
      std::array<REAL, 3>{1.0, 1.0, 1.0}; // right asymptote a_n coefficients

  REAL E_lab_max = 1e3;  // energy value after which the r_coeffs are used
  REAL E_lab_min = 1e-1; // energy value after which the l_coeffs are used

  REAL reduced_mass_amu = 1.0;

  auto cs = AMJUELFitCrossSection<
      3,      // Dimensionality of bulk energy fit
      3,      // Dimensionality of left asymptote fit - no asymptote if 0
      3       // Dimensionality of right asymptote fit - no asymptote if 0
      >(1e6,  // velocity normalisation
        1e-4, // cross-section normalisation in m^2
        reduced_mass_amu,
        coeffs,    // Bulk fit coefficients
        l_coeffs,  // Left asymptote coefficients - set to std::array<REAL,0>{}
                   // if no left asymptote
        r_coeffs,  // Right asymptote coefficents - set to std::array<REAL,0>{}
                   // if no right asymptote
        E_lab_min, // Left asymptote energy threshold - ignored if l_coeffs of
                   // size 0
        E_lab_max, // Right asymptote energy threshold - ignored if r_coeffs of
                   // size 0
        1e5); // this->Maximum expected energy - used to evaluate the maximum
              // value of the cross-section for rejection sampling - after this
              // value, the cross-section decays as 1/v_rel

  return;
}
