inline void linear_reaction_CX_example(ParticleGroupSharedPtr particle_group) {

  auto particle_spec = particle_group->get_particle_spec();

  // We take two species, one with internal state ID = 0 and one with ID = 1
  // We shall treat the ID = 0 species as the projectile in a CX event
  auto projectile_species = Species("ION", 1.2, 0.0, 0);
  auto target_species = Species("ION2", 2.0, 0.0, 1);

  // All reactions are applied to a subgroup. Here we intend to apply the
  // reaction only to those particles with ID = 0, since those are the
  // projectiles
  //
  // We use the following marking strategy, but avoid hardcoding the internal
  // state ID by using the default map

  auto prop_map = get_default_map();

  auto mark_id_zero = make_marking_strategy<
      ComparisonMarkerSingle<INT, EqualsComp>>(
      Sym<INT>(
          prop_map[default_properties.internal_state]), // this will result in
                                                        // "INTERNAL_STATE" when
                                                        // using the default map
      projectile_species.get_id()); // projectile species internal state id = 0

  // The resulting subgroup will have only particles with ID=0
  auto input_subgroup = std::make_shared<ParticleSubGroup>(particle_group);
  auto particle_sub_group = mark_id_zero->make_marker_subgroup(input_subgroup);

  // For this example, we will use the FixedRateData reaction data class
  // it simply sets the rate to a fixed number

  auto rate_data =
      FixedRateData(1.0); // the values used here are somewhat arbitrary,
                          // normally they would depend on the normalisation

  // Since the CX kernels require ndim values per particle in additional
  // reaction data we build a DataCalculator that will produce ndim values
  //
  // Here we choose ndim = 2, and the two data values per particle expected by
  // the 2D CX kernel are the x and y values of the ion velocities sampled from
  // the ion distribution
  //
  // For this example, we mimic a beam in 2D, by using two FixedRateData objects
  // in the DataCalculator
  //
  // In this case, the beam would flow to the bottom right of the domain
  auto vx_beam_data = FixedRateData(1.0);
  auto vy_beam_data = FixedRateData(-1.0);

  // DataCalculators are templated against their contents, so in this case
  auto data_calculator =
      DataCalculator<FixedRateData, FixedRateData>(vx_beam_data, vy_beam_data);

  // Finally, the CXReactionKernels class only requires the dimensionality of
  // the velocity space, as well as the two species

  auto cx_kernel = CXReactionKernels<2>(
      target_species, projectile_species,
      prop_map // The property map used to remap any of the required properties
               // in the kernel The CX kernel requires
               // default_properties.weight, default_properties.velocity as well
               // as some of the source properies (see the CXReactionKernels
               // docs)
  );

  // We can now assemble the linear reaction using the base class constructor
  auto cx_reaction = LinearReactionBase<
      1, // The number of outgoing particles - CX will produce one neutral of
         // the target species
      FixedRateData,        // The reaction data class to be used for the rate
                            // calculation
      CXReactionKernels<2>, // The kernel class used
      DataCalculator<FixedRateData, FixedRateData> // DataCalculator class used
      >(particle_group
            ->sycl_target, // The reaction class needs access to the sycl_target
                           // of the group whose subgroups it's to be applied to
        projectile_species.get_id(), // Ingoing partice state id
        std::array<int, 1>{static_cast<int>(
            target_species
                .get_id())}, // State IDs of all the products - here just one
        rate_data,           // Reaction data used for the rate calculation
        cx_kernel,           // Reaction kernel object to be used
        data_calculator,
        prop_map // Map used for getting weight and total rate syms
  );

  // The following is normally handled by the ReactionController
  //
  // We will loop over all cells and generate products
  int cell_count = particle_group->domain->mesh->get_cell_count();
  auto product_group = std::make_shared<ParticleGroup>(
      particle_group->domain, particle_spec, particle_group->sycl_target);

  for (int i = 0; i < cell_count; i++) {
    // This is the rate loop, here the reaction rates are calculated,
    // they are added to a total reaction rate, and the DataCalculator
    // performs any calculations in needs to
    cx_reaction.run_rate_loop(particle_sub_group, i, i + 1);

    // For the product loop, the reaction needs to know the timestep (here
    // arbitrarily set to 0.1) and the product group
    //
    // The timestep is used to calculate the total particle weight participating
    // in the reaction as rate * timestep
    cx_reaction.descendant_product_loop(particle_sub_group, i, i + 1, 0.1,
                                        product_group);
  }

  return;
}
