#ifndef REACTIONS_TRIM_EVAL_DATA_H
#define REACTIONS_TRIM_EVAL_DATA_H

#include "../../reactions/neso_particles_namespace_alias.hpp"
#include "../../reactions/neso_test_assert.hpp"
#include "../interp_utils.hpp"
#include "../particle_properties_map.hpp"
#include "../reaction_data.hpp"
#include "../utils.hpp"
#include <array>
#include <memory>

#include "grid_descriptors.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation evaluating a tabulated
 * distribution by computing grid indices for interpolation dimensions and
 * binning the remaining TRIM dimensions against nested table values.
 *
 * TRIM = TRansport of Ions in Matter.
 *
 * An input coordinate is split into two parts. The first interp_ndim components
 * (where interp_ndim = input_ndim - output_ndim) are interpolation coordinates.
 * For each such component an index for it in the corresponding coordinate
 * vector is computed, exactly as in CartesianGridDataOnDevice. These
 * per-dimension indices are flattened with row-major ordering into a flat grid
 * index, and the base data offset is flat_index * grid_stride, (details of the
 * grid_stride calculation are in the TrimEvalData docstrings).
 *
 * The remaining output_ndim components are TRIM coordinates between 0.0
 * and 1.0. Each is uniformly binned against the corresponding entry in
 * d_trim_dims. The grid data for a single interpolation point is a
 * concatenation of nested arrays. Data corresponding to the first output
 * dimension occupies a 1-D array of length d_trim_dims[0]; the second occupies
 * a 2-D array of size d_trim_dims[0] * d_trim_dims[1] starting immediately
 * after the first; the third occupies a 3-D array of size d_trim_dims[0] *
 * d_trim_dims[1] * d_trim_dims[2] starting immediately after the second; in
 * general the table for output dimension idim has size:
 * product(d_trim_dims[jdim] for jdim = 0 to idim). To read output idim, the
 * flattened nested data at d_grid[grid_access_point] has to be accessed by
 * calculating a field_access_point and a field_stride and adding those to the
 * binned_input[idim]. This total is then added to grid_access_point to get the
 * index of the trim_vals[idim].
 *
 * The setup of the nested tables and the details of accessing the elements can
 * be found in the EIRENE documentation (in section 4):
 * https://www.eirene.de/old_eirene/trim.pdf
 *
 * @tparam input_ndim Total input dimensionality (interpolation plus TRIM
 * dimensions).
 * @tparam output_ndim Number of TRIM dimensions (size of the returned value
 * array).
 */
template <int input_ndim>
struct TrimEvalDataOnDevice
    : public ReactionDataBaseOnDevice<3, DEFAULT_RNG_KERNEL, input_ndim> {

  static constexpr int output_ndim = 3;

  TrimEvalDataOnDevice() {
    static_assert(
        input_ndim >= output_ndim,
        "For TrimEvalDataOnDevice, input_ndim >= output_ndim must be true.");
  }

  /**
   * @brief Constructor for TrimEvalDataOnDevice.
   *
   * @param d_grid Device buffer containing the tabulated distribution data.
   * @param d_coords Device buffer containing coordinate boundaries for the
   * interpolation dimensions.
   * @param d_dims Device buffer containing grid dimensions for the
   * interpolation axes.
   * @param d_trim_dims Device buffer containing TRIM grid dimensions.
   */
  TrimEvalDataOnDevice(
      const std::shared_ptr<NP::BufferDevice<NP::REAL>> &d_grid,
      const std::shared_ptr<NP::BufferDevice<NP::REAL>> &d_coords,
      const std::shared_ptr<NP::BufferDevice<size_t>> &d_dims,
      const std::shared_ptr<NP::BufferDevice<size_t>> &d_trim_dims)
      : TrimEvalDataOnDevice() {
    this->d_grid_ptr = d_grid->ptr;
    this->d_coords_ptr = d_coords->ptr;
    this->d_dims_ptr = d_dims->ptr;
    this->d_trim_dims_ptr = d_trim_dims->ptr;
  }

  /**
   * @brief Function to evaluate the tabulated TRIM distribution. Computes grid
   * indices for the interpolation dimensions, bins the TRIM dimensions, and
   * returns the values for the computed flat index from the nested data at the
   * interpolation point.
   *
   * @param input The input coordinate array of size input_ndim.
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation. The panic counter is
   * incremented when a TRIM coordinate falls outside 0.0 and 1.0.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation (unused here).
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation (unused here).
   *
   * @return A NP::REAL-valued array of size output_ndim containing the TRIM
   * values at the interpolation point.
   */
  std::array<NP::REAL, output_ndim>
  calc_data(const std::array<NP::REAL, input_ndim> &input,
            const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            [[maybe_unused]] const NP::Access::SymVector::Read<NP::REAL>
                &req_real_props,
            [[maybe_unused]] DEFAULT_RNG_KERNEL::KernelType &rng_kernel) const {

    std::array<NP::REAL, output_ndim> input_to_bin;
    std::array<NP::INT, output_ndim> trim_dims_arr;
    for (size_t i = 0; i < output_ndim; i++) {
      input_to_bin[i] = input[i + interp_ndim];

      req_int_props.at(this->panic_ind, index, i) +=
          ((input_to_bin[i] < 0.0) || (input_to_bin[i] >= 1.0)) ? 1 : 0;

      input_to_bin[i] = ((input_to_bin[i] < 0.0) || (input_to_bin[i] >= 1.0))
                            ? 0.0
                            : input_to_bin[i];

      trim_dims_arr[i] = this->d_trim_dims_ptr[i];
    }

    std::array<NP::INT, output_ndim> binned_inputs =
        interp_utils::bin_uniform_indices(input_to_bin, trim_dims_arr);

    std::array<NP::INT, interp_ndim> grid_indices;
    grid_indices[0] = interp_utils::calc_floor_point_index(
        input[0], this->d_coords_ptr, this->d_dims_ptr[0]);
    size_t aggregate_dims = 0;
    for (size_t i = 1; i < interp_ndim; i++) {
      aggregate_dims += this->d_dims_ptr[i - 1];
      grid_indices[i] = interp_utils::calc_floor_point_index(
          input[i], this->d_coords_ptr + aggregate_dims, this->d_dims_ptr[i]);
    }

    auto grid_indices_ptr = grid_indices.data();
    NP::INT grid_flat_index = interp_utils::coeff_index_on_device(
        grid_indices_ptr, this->d_dims_ptr, (input_ndim - output_ndim));

    auto grid_access_point = grid_flat_index * this->grid_stride;

    std::array<NP::INT, output_ndim> trim_indices;
    std::array<NP::REAL, output_ndim> trim_vals;

    for (int idim = 0; idim < output_ndim; idim++) {
      int field_access_point = 0;
      int field_stride = 0;
      int aggregate_dim = 1;
      int offset_factor = 1;

      for (int jdim = 0; jdim <= idim; jdim++) {
        aggregate_dim *= this->d_trim_dims_ptr[jdim];
      }

      for (int jdim = 0; jdim < idim; jdim++) {
        offset_factor *= this->d_trim_dims_ptr[jdim];
        field_access_point += offset_factor;
        // TODO try to optimize out the integer division
        field_stride += binned_inputs[jdim] * (aggregate_dim / offset_factor);
      }

      trim_indices[idim] =
          field_access_point + field_stride + binned_inputs[idim];
      trim_vals[idim] =
          this->d_grid_ptr[grid_access_point + trim_indices[idim]];
    }
    return trim_vals;
  }

public:
  int grid_stride;

  static constexpr int interp_ndim = input_ndim - output_ndim;

  NP::REAL const *d_grid_ptr;
  NP::REAL const *d_coords_ptr;
  size_t const *d_dims_ptr;
  size_t const *d_trim_dims_ptr;

  int panic_ind;
};

/**
 * @brief Reaction rate data calculation managing buffers for grid, coords,
 * dims, and trim_dims, enabling on-device tabulated distribution evaluation.
 *
 * The evaluation works with the NP::BufferDevice objects that are constructed
 * for the input vectors (grid, coords_vec, dims_vec, trim_dims_vec). All input
 * vectors are 1D vectors and are accessed using the logic in the on-device
 * calc_data(...). The interpolated points are calculated with the same indexing
 * as in CartesianGridDataOnDevice. The TRIM dimensions are uniformly binned.
 * The TRIM grid data for each interpolation point is a concatenation of nested
 * tables whose sizes are determined by cumulative products of trim_dims_vec
 * entries. The grid_stride member stores the total size of this concatenation
 * and is precomputed in the constructor. It's effectively a two-stage
 * calculation for itrim_dim:
 * sum(product(trim_dims_vec[jtrim_dim] for jtrim_dim= 0 to jtrim_dim =
 * itrim_dim) for itrim_dim = 0 to itrim_dim = output_ndim).
 *
 * @tparam input_ndim Total input dimensionality (interpolation plus TRIM
 * dimensions).
 * @tparam output_ndim Number of TRIM dimensions (size of the returned value
 * array).
 */
template <int input_ndim>
struct TrimEvalData
    : public ReactionDataBase<TrimEvalDataOnDevice<input_ndim>> {

  static constexpr int output_ndim = 3;
  static constexpr int interp_ndim = input_ndim - output_ndim;

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_int_props = {props.panic};
  /**
   * @brief Constructor for TrimEvalData.
   *
   * @param grid Flat vector of grid values (tabulated distribution data).
   * @param coords_vec Coordinate boundaries for the interpolation dimensions
   * (used for index computation).
   * @param dims_vec Grid dimensions for the interpolation axes.
   * @param trim_dims_vec Trim grid dimensions (ie. number of bins per TRIM
   * axis).
   * @param sycl_target SYCL target shared pointer used for buffer
   * allocation.
   * @param properties_map Map of property indices to names.
   */
  TrimEvalData(const std::vector<NP::REAL> &grid,
               const std::vector<NP::REAL> &coords_vec,
               const std::vector<size_t> &dims_vec,
               const std::vector<size_t> &trim_dims_vec,
               NP::SYCLTargetSharedPtr sycl_target,
               std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<TrimEvalDataOnDevice<input_ndim>>(
            Properties<NP::INT>(required_simple_int_props), properties_map) {

    auto dims_size = dims_vec.size();
    NESOASSERT((dims_size == interp_ndim),
               "Invalid size of input dims vector.");

    auto trim_dims_size = trim_dims_vec.size();
    NESOASSERT((trim_dims_size == output_ndim),
               "Invalid size of input TRIM dims vector.");

    auto grid_stride = 0;
    int aggregate_dim = 1;
    for (auto &trim_dim : trim_dims_vec) {
      aggregate_dim *= trim_dim;
      grid_stride += aggregate_dim;
    }

    auto grid_size = grid.size();
    auto coords_size = coords_vec.size();

    auto expected_grid_size = 1;
    auto expected_coords_size = 0;

    for (auto &idim : dims_vec) {
      expected_grid_size *= idim;
      expected_coords_size += idim;
    }

    expected_grid_size *= grid_stride;

    NESOASSERT((coords_size == expected_coords_size),
               "Invalid size of input coords vector.");
    NESOASSERT((grid_size == expected_grid_size),
               "Invalid size of input grid.");

    this->d_grid = utils::make_buffer_device_ptr(sycl_target, grid);
    this->d_coords = utils::make_buffer_device_ptr(sycl_target, coords_vec);
    this->d_dims = utils::make_buffer_device_ptr(sycl_target, dims_vec);
    this->d_trim_dims =
        utils::make_buffer_device_ptr(sycl_target, trim_dims_vec);

    this->on_device_obj =
        TrimEvalDataOnDevice<input_ndim>(d_grid, d_coords, d_dims, d_trim_dims);

    this->on_device_obj->grid_stride = grid_stride;

    this->index_on_device_obj();
  };

  /**
   * \overload
   * @brief Construct from a GridDescriptor object.
   */
  TrimEvalData(const GridDescriptor<interp_ndim, output_ndim> &grid_descriptor,
               NP::SYCLTargetSharedPtr sycl_target,
               std::map<int, std::string> properties_map = get_default_map())
      : TrimEvalData(
            grid_descriptor.get_flat_grid(), grid_descriptor.get_flat_coords(),
            grid_descriptor.get_interp_dims(),
            grid_descriptor.get_output_dims(), sycl_target, properties_map) {
    static_assert(interp_ndim == input_ndim - output_ndim,
                  "GridDescriptor interpolation dimensions must match "
                  "input_ndim - output_ndim");
  }

  void index_on_device_obj() {
    this->on_device_obj->panic_ind = this->required_int_props.find_index(
        this->properties_map.at(props.panic));
  };

public:
  std::shared_ptr<NP::BufferDevice<NP::REAL>> d_coords;
  std::shared_ptr<NP::BufferDevice<size_t>> d_dims;
  std::shared_ptr<NP::BufferDevice<size_t>> d_trim_dims;
  std::shared_ptr<NP::BufferDevice<NP::REAL>> d_grid;
};

} // namespace VANTAGE::Reactions

#endif
