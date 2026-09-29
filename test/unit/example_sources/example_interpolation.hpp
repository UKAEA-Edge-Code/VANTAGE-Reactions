inline void interpolation_example(NP::ParticleGroupSharedPtr particle_group) {
  // Number of dimensions of the pre-calculated grid
  static constexpr int ndim = 3;

  // Example coordinates for dimension 0.
  std::vector<REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                  5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

  // Example coordinates for dimension 1.
  std::vector<REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  // Example coordinates for dimension 2.
  std::vector<REAL> dim2_range = {-98.5, -94.,  -86.5, -76., -62.5,
                                  -46.,  -26.5, -4.,   21.5, 50.,
                                  81.5,  116.,  153.5, 194., 237.5};

  // Example lambda defining the values of the grid function at each coordinate.
  static constexpr auto grid_func_lambda =
      [](const std::array<REAL, ndim> &vals) {
        return (vals[0] * vals[1] * vals[2]);
      };

  // Helper class to construct pre-requisites for CartesianGridData.
  auto grid_descriptor = GridDescriptor<ndim>(
      {dim0_range, dim1_range, dim2_range}, grid_func_lambda);

  auto dims_vec = grid_descriptor.get_interp_dims();

  auto coords_vec = grid_descriptor.get_flat_coords();

  // Helper class that provides a calc_data() that retrieves values from the
  // pre-calculated grid in grid_descriptor.
  auto grid_func_data =
      CartesianGridData<ndim>(grid_descriptor, particle_group->sycl_target);

  // Construction of InterpolateData. There is a calc_data() within the
  // on-device object associated with this class which may be used directly if
  // the correct input_array is given. But it is meant to be used in a pipeline
  // with results from another ReactionData-derived object acting as inputs for
  // InterpolateData.
  auto interpolate_data = InterpolateData<1, ndim, decltype(grid_func_data)>(
      dims_vec, coords_vec, particle_group->sycl_target, grid_func_data);
  // If a different extrapolation other than continue_linear is needed, then
  // just add either:
  //  - ExtrapolationType::clamp_to_edge
  //  - ExtrapolationType::clamp_to_zero
  // as the last argument when constructing InterpolateData.

  // Example of construction of a ReactionData-derived object which has an
  // output that matches the type of the expected input of the calc_data() from
  // InterpolateDataOnDevice.
  auto prop0_extract = extract<1>("PROP0");
  auto prop1_extract = extract<1>("PROP1");
  auto prop2_extract = extract<1>("PROP2");
  auto concatenator =
      ConcatenatorData(prop0_extract, prop1_extract, prop2_extract);

  // Pipeline that handles the pass-through of values.
  auto pipeline = pipe(concatenator, interpolate_data);

  // Wrapping in a DataCalculator allows the extraction of values from
  // interpolate_data via a NP::NDLocalArray buffer.
  auto data_calc = DataCalculator(pipeline);

  return;
}
