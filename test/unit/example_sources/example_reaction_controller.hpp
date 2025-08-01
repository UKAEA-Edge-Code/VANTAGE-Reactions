inline void reaction_controller_example(ParticleGroupSharedPtr particle_group) {

  auto particle_spec = particle_group->get_particle_spec();

  auto prop_map = get_default_map();

  // We shall generate a CX and two ionisation reactions quickly, in order to
  // populate the controller with some reactions

  auto electron_species = Species("ELECTRON", 5.5e-4, -1.0);
  auto ion_species_1 = Species("ION", 1.2, 0.0, 0);
  auto ion_species_2 = Species("ION2", 2.0, 0.0, 1);

  // We'll use dummy rate data to simplify the setup
  auto rate_data = FixedRateData(1.0);

  // CX beam data
  auto vx_beam_data = FixedRateData(1.0);
  auto vy_beam_data = FixedRateData(-1.0);

  auto data_calculator =
      DataCalculator<FixedRateData, FixedRateData>(vx_beam_data, vy_beam_data);

  auto cx_kernel = CXReactionKernels<2>(ion_species_1, ion_species_2, prop_map);

  // CX reaction
  auto cx_reaction = std::make_shared<
      LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                         DataCalculator<FixedRateData, FixedRateData>>>(
      particle_group->sycl_target, ion_species_1.get_id(),
      std::array<int, 1>{static_cast<int>(ion_species_2.get_id())}, rate_data,
      cx_kernel, data_calculator);

  // Ionisation reactions
  auto ionise_reaction_1 =
      std::make_shared<ElectronImpactIonisation<FixedRateData, FixedRateData>>(
          particle_group->sycl_target, rate_data, rate_data, ion_species_1,
          electron_species);

  auto ionise_reaction_2 =
      std::make_shared<ElectronImpactIonisation<FixedRateData, FixedRateData>>(
          particle_group->sycl_target, rate_data, rate_data, ion_species_2,
          electron_species);

  // We can now initialise a reaction controller and populate it with the above
  // reactions We start off by creating transformations for the children and
  // parents
  //
  // We will remove any children/parents below some set weight using the
  // following transform

  auto remove_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>(prop_map[default_properties.weight]), 1e-6)},
      make_transformation_strategy<SimpleRemovalTransformationStrategy>());
  // But will first try merge any children/parents below a higher weight
  // threshold

  auto merge_transform =
      make_transformation_strategy<MergeTransformationStrategy<2>>();

  auto merge_wrapper = std::make_shared<TransformationWrapper>(
      std::vector<std::shared_ptr<MarkingStrategy>>{
          make_marking_strategy<ComparisonMarkerSingle<REAL, LessThanComp>>(
              Sym<REAL>(prop_map[default_properties.weight]), 1e-2)},
      merge_transform);

  auto reaction_controller = ReactionController(
      std::vector{
          merge_wrapper,
          remove_wrapper}, // the order matters! this will first merge parents,
                           // then remove any remaining small particles
      std::vector{merge_wrapper, remove_wrapper}
      // this will do the same to the children
      // before merging them into the parents
  );

  reaction_controller.set_cell_block_size(
      256); // This is the greedy default value, reduce this if memory issues
            // are found
  reaction_controller.set_max_particles_per_cell(
      16384); // This is the default maximum (average) number of particles per
              // cell for the use in reaction data buffers, modify as needed

  // Now we can add the reaction simply
  reaction_controller.add_reaction(cx_reaction);
  reaction_controller.add_reaction(ionise_reaction_1);
  reaction_controller.add_reaction(ionise_reaction_2);

  // We can now request an Euler step of 0.01 time units
  reaction_controller.apply_reactions(particle_group, 0.01);

  return;
}
