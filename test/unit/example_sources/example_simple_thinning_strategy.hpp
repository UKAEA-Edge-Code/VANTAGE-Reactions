inline void
simple_thinning_strategy_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop_map = get_default_map();

  // Below is a placeholder rng kernel, in practice this would be a uniformly
  // sampled value - here we just use a constant number
  auto rng_lambda = [&]() -> REAL { return 0.01; };

  auto rng_kernel = host_per_particle_block_rng<REAL>(rng_lambda, 1);

  auto thinning_strat = make_simple_thinning_strategy(
      particle_group, // The particle group
      0.1, // The thinning ratio - will on average keep 10% of the particles
      rng_kernel, // The uniform random variate rng kernel
      prop_map    // Property map used for
                  // remapping the weight, and the panic flag
  );

  thinning_strat->transform(input_subgroup);

  return;
}
