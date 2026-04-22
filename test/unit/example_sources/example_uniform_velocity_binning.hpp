inline void
uniform_velocity_binning_example(ParticleGroupSharedPtr particle_group) {

  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);

  auto prop_map = get_default_map();

  auto velocity_bin =
      uniform_velocity_bin_transform<2 // Velocity space dimensionality
                                     >(
          std::array<REAL, 2>{3.0,
                              3.0}, // Extent in each dimension of
                                    // the main binning cells - corresponding to
                                    // ranges (-1.5,1.5]x(-1.5,1.5]
          std::array<INT, 2>{10, 10}, // Number of main binning cells in each
                                      // dimension - will result in 12 x 12
                                      // total cells, accounting for guard cells
          Sym<INT>("REACTIONS_GROUPING_INDEX"), // The linear velocity indexing
                                                // Sym
          Sym<REAL>("VELOCITY")                 // The velocity Sym
      );

  velocity_bin->transform(input_subgroup);

  return;
}
