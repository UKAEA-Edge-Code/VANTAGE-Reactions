inline void spec_builder_example() {

  // The recommended way of initialising a ParticleSpecBuilder 
  // is by using the following constructor, which will 
  // make sure that the properties present satisfy requirements in the 
  // library 
  
  auto particle_spec_builder_default = ParticleSpecBuilder(2 // the dimensionality of the default 
                                                             // position and velocity props 
                                                          ); // A map (see below) can additionally i
                                                             // be passed here to remap
                                                             // some of the default property Syms

  // Alternatively, if full control is required 
  // a spec builder can be initialised with an existing spec,
  // which will not add any of the default properties.
  // The user is responsible then for ensuring that
  // all required properties are available when requested.
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
