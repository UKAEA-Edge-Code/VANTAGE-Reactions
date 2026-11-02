void vranic_merging_strategy_example(
    NP::ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = NP::particle_sub_group(particle_group);

  auto prop_map = PropertiesMap();

  auto merging_strat =
      make_vranic_merging_strategy<2 // The velocity space dimensionality
                                   >(particle_group, // The particle group
                                     1, // The number of downsampling groups
                                        // (e.g. velocity bins, etc.)
                                     prop_map // Property map used for
                                     // remapping the grouping index, weight,
                                     // linear index, and velocity properties
      );

  merging_strat->transform(input_subgroup);

  return;
}
