inline void direct_marking_example(ParticleGroupSharedPtr particle_group) {

  // Here we create a marking strategy marking low weight particles
  auto marking_strategy = make_direct_marking_strategy(
      "test_strategy", // Name of the strategy for profiling purposes
      [](auto w) { return w[0] < 1e-6; }, // Marking kernel
      Access::read(Sym<REAL>("WEIGHT"))   // Accessors for the kernel
  );

  // The subgroup can then be created as follows from another subgroup

  auto subgroup_low_weight = marking_strategy->make_marker_subgroup(
      particle_sub_group(particle_group));

  // Contrast the above with the full constructor call from NESO-Particles
  auto subgroup_low_weight_from_NP = particle_sub_group(
      particle_group, [](auto w) { return w[0] < 1e-6; },
      Access::read(Sym<REAL>("WEIGHT")));

  return;
}
