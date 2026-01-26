inline void cartesian_basis_reflection_example() {

  // In case we wish to remap the default boundary normal or velocity name
  auto used_map = get_default_map();

  // Reflect in the direction normal to the surface, and back into the domain
  std::array<REAL, 3> coords{0, 0, 1};
  // Here we just use a fixed reflection value, but this can
  // be calculated using any other reaction data object
  auto coord_data = FixedArrayData<3>(coords);

  auto spherical_reflection = CartesianBasisReflectionData();

  // The pipeline object will first evaluate the reflected coordinate (fixed
  // here) and will then pipe this into the reflection data object.
  //
  // The reflection data object will use the particle velocity and the surface
  // normal to generate the correct post-reflection basis, where the
  // post-reflection velocity is given with local cartesian coordinates
  //
  // NOTE: This will only work with 3-dimensional data
  auto pipeline = pipe(coord_data, spherical_reflection);

  return;
}
