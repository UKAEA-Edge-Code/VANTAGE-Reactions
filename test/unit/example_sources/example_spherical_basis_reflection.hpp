inline void spherical_basis_reflection_example() {

  // In case we wish to remap the default boundary normal or velocity name
  auto used_map = get_default_map();

  // v = 2, theta = pi/4, phi = 3*pi/4
  std::array<REAL, 3> coords{2.0, M_PI / 4, 3 * M_PI / 4};
  // Here we just use a fixed reflection value, but this can
  // be calculated using any other reaction data object
  auto coord_data = FixedArrayData<3>(coords);

  auto spherical_reflection = SphericalBasisReflectionData();

  // The pipeline object will first evaluate the reflected coordinate (fixed
  // here) and will then pipe this into the reflection kernel.
  //
  // The reflection kernel will use the particle velocity and the surface normal
  // to generate the correct post-reflection basis, where the v, theta, and phi
  // coordinate values would be used
  //
  // NOTE: This will only work with 3-dimensional data
  auto pipeline = pipe(coord_data, spherical_reflection);

  return;
}
