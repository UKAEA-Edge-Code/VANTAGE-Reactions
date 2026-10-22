void source_reaction_example(NP::ParticleGroupSharedPtr particle_group) {
  auto marker_species = Species("MARKER", 1.0, 0.0, 0);
  auto product_species = Species("PRODUCT", 2.0, 0.0, 1);

  auto velocity_data = FixedArrayData<2>(std::array<REAL, 2>{1.0, 0.0});

  // SourceReaction uses rate 1.0 and preserves marker weights.
  auto source_reaction = SourceReaction<FixedArrayData<2>>(
      particle_group->sycl_target, velocity_data, marker_species,
      product_species);

  return;
}
