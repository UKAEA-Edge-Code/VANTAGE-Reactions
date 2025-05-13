inline void property_container_example() {

  // We first construct a few species objects. Here we only need the species
  // name, so we set the mass, charge, and ID to arbitrary values

  auto electron_species = Species("ELECTRON", // name
                                  1.0,        // mass
                                  -1.0,       // charge
                                  -1          // ID
  );
  // Only the species name is required, since the the main use of the Species
  // class is property name construction

  auto ion_species = Species("ION");

  // We can encapsulate a few integer properties directly. We refer to
  // properties using a property enum, here the default one

  auto int_props = Properties<INT>(std::vector<int>{
      default_properties.id, default_properties.internal_state});

  // We can do the same for real properties, but this time we also store some
  // species information
  auto real_props = Properties<REAL>(
      std::vector<int>{default_properties.velocity, default_properties.weight},
      std::vector<Species>{electron_species, ion_species},
      std::vector<int>{default_properties.temperature,
                       default_properties.density});

  // In order to convert property information into NESO-Particle Sym names,
  // we require a property map. This defaults to the default_map object, but
  // users can supply their own, usually to the constructor of the object that
  // own the properies.

  auto req_int_prop_names = int_props.simple_prop_names(
      get_default_map()); // explicitly passing the default map - could be a
                    // user-defined map
  auto req_simple_real_prop_names =
      real_props.simple_prop_names(); // map defaults to default_map

  // When requesting species property names the species name is concatenated to
  // the property name In the example here, the required species real properties
  // using the default map will contain:
  //  - "ELECTRON_TEMPERATURE"
  //  - "ELECTRON_DENSITY"
  //  - "ION_TEMPERATURE"
  //  - "ION_DENSITY"

  auto req_species_real_prop_names = real_props.species_prop_names();

  return;
}
