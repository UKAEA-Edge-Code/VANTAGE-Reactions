
#include <random>
#include <tuple>
inline void
trim_interpolation_example(NP::ParticleGroupSharedPtr particle_group) {
  // Number of dimensions of the pre-calculated grid
  static constexpr int interp_ndim = 2;

  // Number of dimensions associated with the TRIM tables
  static constexpr int trim_ndim = 3;

  static constexpr int trim_dim0 = 5;
  static constexpr int trim_dim1 = 5;
  static constexpr int trim_dim2 = 5;

  static constexpr auto trim_dims_arr =
      std::array<size_t, trim_ndim>{trim_dim0, trim_dim1, trim_dim2};

  // Example coordinates for dimension 0.
  std::vector<NP::REAL> dim0_range = {1.0e+18, 2.0e+18, 3.0e+18, 4.0e+18,
                                      5.0e+18, 6.0e+18, 7.0e+18, 8.0e+18};

  // Example coordinates for dimension 1.
  std::vector<NP::REAL> dim1_range = {
      1.00000000e+01, 2.78255940e+01, 7.74263683e+01, 2.15443469e+02,
      5.99484250e+02, 1.66810054e+03, 4.64158883e+03, 1.29154967e+04,
      3.59381366e+04, 1.00000000e+05};

  // Example lambda for the TRIM tables.
  static constexpr auto trim_grid_func_lambda =
      [](const std::array<NP::REAL, interp_ndim> &vals) {
        // This calculates the correct length for the output.
        // \sum_{i = 0}^{n}(\prod_{j=0}^{i} trim_dims_arr[j])
        static constexpr size_t trim_table_length = [&] {
          NP::INT s = 0;
          NP::INT p = 1;
          for (NP::INT x : trim_dims_arr) {
            p *= x;
            s += p;
          }
          return s;
        }();
        std::array<NP::REAL, trim_table_length> result;
        for (size_t i = 0; i < trim_table_length; i++) {
          result[i] = std::pow((i + 50), 3.0);
        }

        return result;
      };

  // Helper class to construct pre-requisites for TrimEvalData.
  auto grid_descriptor = GridDescriptor<interp_ndim, trim_ndim>(
      {dim0_range, dim1_range}, {trim_dim0, trim_dim1, trim_dim2},
      trim_grid_func_lambda);

  auto dims_vec = grid_descriptor.get_interp_dims();

  auto coords_vec = grid_descriptor.get_flat_coords();

  // Helper class that provides a calc_data() that retrieves values from the
  // pre-calculated grid in grid_descriptor.
  auto grid_func_data = TrimEvalData<interp_ndim + trim_ndim>(
      grid_descriptor, particle_group->sycl_target);

  // Given that the "grid" will have (interp_ndim + trim_ndim) dimensions, it's
  // necessary to specify the indices of the dimensions that are to be
  // interpolated.
  std::array<size_t, interp_ndim> interp_indices = {0, 1};

  // Construction of InterpolateData. There is a calc_data() within the
  // on-device object associated with this class which may be used directly if
  // the correct input_array is given. But it is meant to be used in a pipeline
  // with results from another ReactionData-derived object acting as inputs for
  // InterpolateData.
  auto interpolate_data =
      InterpolateData<trim_ndim, interp_ndim, decltype(grid_func_data),
                      trim_ndim>(dims_vec, coords_vec, interp_indices,
                                 particle_group->sycl_target, grid_func_data);
  // If a different extrapolation other than continue_linear is needed, then
  // just add either:
  //  - ExtrapolationType::clamp_to_edge
  //  - ExtrapolationType::clamp_to_zero
  // as the last argument when constructing InterpolateData.

  // Example of construction of a ReactionData-derived object which has an
  // output that matches the type of the expected input of the calc_data() from
  // InterpolateDataOnDevice.
  auto props_extract = extract<interp_ndim>("PROPS");

  const int rank = particle_group->sycl_target->comm_pair.rank_parent;
  auto rng = std::mt19937(52234126 + rank);
  std::uniform_real_distribution<NP::REAL> uniform_dist_2(0.0, 1.0);

  auto rng_lambda = [&]() -> NP::REAL {
    NP::REAL rng_sample = 0.0;
    do {
      rng_sample = uniform_dist_2(rng);
    } while (rng_sample == 0.0);
    return rng_sample;
  };

  auto trim_rng_kernel =
      NP::host_per_particle_block_rng<NP::REAL>(rng_lambda, 1);

  auto trim_sampler = SamplerData(trim_rng_kernel);
  // This is hard-coded here to avoid bloated general implementation for
  // arbitrary number of samplers. (quite easy in C++20)
  auto trim_sampler_concat =
      ConcatenatorData(trim_sampler, trim_sampler, trim_sampler);

  auto concatenator = ConcatenatorData(props_extract, trim_sampler_concat);

  // Pipeline that handles the pass-through of values.
  auto pipeline = pipe(concatenator, interpolate_data);

  // Wrapping in a DataCalculator allows the extraction of values from
  // interpolate_data via a NP::NDLocalArray buffer.
  auto data_calc = DataCalculator(pipeline);

  return;
}
