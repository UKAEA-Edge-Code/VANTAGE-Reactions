inline void spec_builder_example() {

  // A spec builder can be initialised with an existing spec
  auto basic_spec = ParticleSpec{ParticleProp(Sym<REAL>("POSITION"), 2, true),
                                 ParticleProp(Sym<INT>("CELL_ID"), 1, true)};
  auto particle_spec_builder = ParticleSpecBuilder(basic_spec);

  // Default properties alias
  auto props = default_properties;

  // We can create multiple property containers
  auto int_props = Properties<INT>(std::vector<int>{props.internal_state});
  auto scalar_real_props = Properties<REAL>(
      std::vector<int>{props.weight}, std::vector<Species>{Species("ELECTRON")},
      std::vector<int>{props.temperature});

  auto vector_real_props =
      Properties<REAL>(std::vector<int>{props.velocity},
                       std::vector<Species>{Species("ELECTRON")},
                       std::vector<int>{props.flow_speed});

  particle_spec_builder.add_particle_prop(int_props);

  // We can also remap enum properties as we add them using property maps
  auto new_map = get_default_map();
  new_map[props.weight] = "w";
  particle_spec_builder.add_particle_prop(
      scalar_real_props, // The added properties
      1,                 // property dimensionality
      false,  // whether the property is a position property or not (most often
              // false)
      new_map // the used property map - defaults to default_map
  );

  // Here we add the vector (2D) properties
  particle_spec_builder.add_particle_prop(vector_real_props, 2);

  // Finally we can merge the particle spec in the builder with other specs
  // (will ignore duplicates)
  particle_spec_builder.add_particle_spec(ParticleSpec{ParticleProp(Sym<REAL>("w"),1,false), ParticleProp(Sym<REAL>("NEW_PROP"),2,false)});

  auto built_spec = particle_spec_builder.get_particle_spec();
  return;
}
