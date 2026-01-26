inline void sampler_example() {

  // In case we wish to remap the default panic flag - used in case the sampling
  // fails
  auto used_map = get_default_map();

  // The sampler needs a NESO-Particles rng_kernel
  //
  // Here we use an arbitrary lambda, but this can be anything
  auto rng_lambda = [&]() -> REAL { return 0.5; };
  auto rng_kernel = host_atomic_block_kernel_rng<REAL>(rng_lambda, 1000);

  // The following will just sample from the above kernel
  auto sampler_data = SamplerData(rng_kernel,
                                  used_map); // Optional property map
  return;
}
