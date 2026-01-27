inline void array_lookup_example(ParticleGroupSharedPtr particle_group) {

  // The arrays can be any size
  // The default array is what is returned if the key value isn't
  // found
  auto default_array = std::array<REAL, 1>{1.0};
  std::map<int, std::array<REAL, 1>> lookup_table_map;

  lookup_table_map[0] = std::array<REAL, 1>{2.0};
  lookup_table_map[1] = std::array<REAL, 1>{3.0};

  // The following data will check the first component of the particle's
  // INTERNAL_STATE value, use it as the lookup key in the above map, and
  // return the corresponding map value if found, otherwise returning
  // the default array
  auto array_lookup_data =
      ArrayLookupData<1>(Sym<INT>("INTERNAL_STATE"), // The key Sym to
                                                     // be used for lookup
                         0,                          // The component of the
                                                     // ParticleDat referred
                                                     // to by the key Sym to
                                                     // be used for the key
                                                     // value
                         lookup_table_map, default_array,
                         particle_group->sycl_target); // A sycl target is
                                                       // needed to store the
                                                       // on-device lookup table
  return;
}
