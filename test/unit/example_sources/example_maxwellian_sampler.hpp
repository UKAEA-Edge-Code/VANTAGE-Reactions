inline void maxwellian_sampler_example() {

  // In case we wish to remap the fluid_temperature, fluid_flow_speed, or
  // velocity
  auto used_map = get_default_map();

  // Default cross-section object - results in sampling from an unfiltered
  // drifting Maxwellian
  auto default_cs = ConstantRateCrossSection(1.0);

  // The sampler needs a NESO-Particles rng_kernel
  // In general, it will need the atomic block kernel (see NESO-Particles
  // documentation) This is because rejection sampling has an a priori unknown
  // number of samples
  //
  // Here we use an arbitrary lambda, but this should in general be a standard
  // uniform distribution
  auto rng_lambda = [&]() -> REAL { return 0.5; };
  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1000);

  // The sampler is templated against velocity space dimensionality - here 2D
  auto sampler_data = FilteredMaxwellianSampler<2>(
      1.0,        // This is the ratio between kT_0 and mv_0^2,
                  // where T_0 is the temperature normalisation,
                  // m is the mass of the species whose distribution is sampled,
                  // and v_0 is the velocity normalisation
      default_cs, // Optional cross-section object - here explicitly defaulted
      rng_kernel, // RNG kernel used to perform Box-Muller sampling and
                  // rejection sampling
      used_map);  // Optional property map

  return;
}
