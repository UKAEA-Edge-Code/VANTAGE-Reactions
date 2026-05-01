#ifndef REACTIONS_TRIM_EVAL_DATA_H
#define REACTIONS_TRIM_EVAL_DATA_H

#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include "reactions_lib/particle_properties_map.hpp"
#include <array>
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief On device: ReactionData evaluating a TRIM (Tabulated Representation
 * of Internal Modes) distribution by binning velocity components and looking
 * up tabulated values from a grid, combining interpolation with trimming.
 *
 * @tparam input_ndim Total input dimensionality (interp + trim dimensions).
 * @tparam output_ndim Number of trim/velocity dimensions (number of output
 * values).
 */
template <int input_ndim, int output_ndim>
struct TrimEvalOnDevice
    : public ReactionDataBaseOnDevice<output_ndim, DEFAULT_RNG_KERNEL,
                                      input_ndim, REAL, INT> {

  TrimEvalOnDevice() {
    static_assert(
        input_ndim >= output_ndim,
        "For TrimEvalOnDevice, input_ndim >= output_ndim must be true.");
  };

  /**
   * @brief Function to evaluate the TRIM distribution. Bins the trim
   * dimensions of the input, computes grid indices for the interpolation
   * dimensions, and returns the tabulated trim values.
   *
   * @param input Input coordinate array of size input_ndim (first interp_ndim
   * components are interpolation coordinates, remaining output_ndim components
   * are trim coordinates).
   * @param index Read-only accessor to a loop index for a ParticleLoop inside
   * which calc_data is called (unused for this data type).
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation (unused for this data
   * type).
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation (unused for this data
   * type).
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation (unused for this data type).
   *
   * @return A REAL-valued array of size output_ndim containing the tabulated
   * trim values.
   */
  std::array<REAL, output_ndim> calc_data(
      const std::array<REAL, input_ndim> &input,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]] DEFAULT_RNG_KERNEL::KernelType &rng_kernel) const {

    std::array<REAL, output_ndim> input_to_bin;
    std::array<INT, output_ndim> trim_dims_arr;
    for (size_t i = 0; i < output_ndim; i++) {
      input_to_bin[i] = input[i + interp_ndim];

      req_int_props.at(this->panic_ind, index, i) +=
          ((input_to_bin[i] < 0) || (input_to_bin[i] >= 1)) ? 1 : 0;

      input_to_bin[i] = ((input_to_bin[i] < 0) || (input_to_bin[i] >= 1))
                            ? 0
                            : input_to_bin[i];

      trim_dims_arr[i] = this->d_trim_dims[i];
    }

    std::array<INT, output_ndim> binned_inputs =
        interp_utils::bin_uniform_indices(input_to_bin, trim_dims_arr);

    std::array<INT, interp_ndim> grid_indices;
    grid_indices[0] = interp_utils::calc_floor_point_index(
        input[0], this->d_ranges, this->d_dims[0]);
    size_t aggregate_dims = 0;
    for (size_t i = 1; i < interp_ndim; i++) {
      aggregate_dims += this->d_dims[i - 1];
      grid_indices[i] = interp_utils::calc_floor_point_index(
          input[i], this->d_ranges + aggregate_dims, this->d_dims[i]);
    }

    auto grid_indices_ptr = grid_indices.data();
    INT grid_flat_index =
        interp_utils::coeff_index_on_device(grid_indices_ptr, this->d_dims, 2);

    auto grid_access_point = grid_flat_index * this->grid_stride;

    std::array<INT, output_ndim> trim_indices;
    std::array<REAL, output_ndim> trim_vals;

    int field_access_point;
    int field_stride;
    int aggregate_dim;
    int offset_factor;
    for (int idim = 0; idim < output_ndim; idim++) {
      field_access_point = 0;
      field_stride = 0;
      aggregate_dim = 1;
      offset_factor = 1;

      for (int jdim = 0; jdim <= idim; jdim++) {
        aggregate_dim *= d_trim_dims[jdim];
      }

      for (int jdim = 0; jdim < idim; jdim++) {
        offset_factor *= d_trim_dims[jdim];
        field_access_point += offset_factor;
        field_stride += binned_inputs[jdim] * (aggregate_dim / offset_factor);
      }

      trim_indices[idim] =
          field_access_point + field_stride + binned_inputs[idim];
      trim_vals[idim] = this->d_grid[grid_access_point + trim_indices[idim]];
    }
    return trim_vals;
  }

public:
  int grid_stride;

  static constexpr int interp_ndim = input_ndim - output_ndim;

  REAL const *d_ranges;
  size_t const *d_dims;
  size_t const *d_trim_dims;
  REAL const *d_grid;

  int panic_ind;
};

/**
 * @brief Host-side ReactionDataBase managing SYCL buffers for grid, ranges,
 * dims, and trim_dims used in TRIM evaluation.
 *
 * @tparam input_ndim Total input dimensionality (interp + trim dimensions).
 * @tparam output_ndim Number of trim/velocity dimensions (number of output
 * values).
 */
template <int input_ndim, int output_ndim>
struct TrimEval
    : public ReactionDataBase<TrimEvalOnDevice<input_ndim, output_ndim>> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_int_props = {props.panic};
  /**
   * @brief Constructor for TrimEval.
   *
   * @param grid Flat vector of grid values containing the tabulated trim data.
   * @param ranges_vec Range boundaries for the interpolation dimensions (used
   * for floor-point index computation).
   * @param dims_vec Grid dimensions for the interpolation axes.
   * @param trim_dims_vec Trim/velocity grid dimensions (number of bins per
   * trim axis).
   * @param sycl_target SYCL target shared pointer used for buffer allocation.
   */
  TrimEval(const std::vector<REAL> &grid, const std::vector<REAL> &ranges_vec,
           const std::vector<size_t> &dims_vec,
           const std::vector<size_t> &trim_dims_vec,
           SYCLTargetSharedPtr sycl_target,
           std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<TrimEvalOnDevice<input_ndim, output_ndim>>(
            Properties<INT>(required_simple_int_props), properties_map) {
    this->on_device_obj = TrimEvalOnDevice<input_ndim, output_ndim>();

    this->h_grid = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;

    this->h_ranges =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->d_ranges = this->h_ranges->ptr;

    this->h_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->d_dims = this->h_dims->ptr;

    int aggregate_dim = 1;
    this->on_device_obj->grid_stride = 0;
    for (auto &trim_dim : trim_dims_vec) {
      aggregate_dim *= trim_dim;
      this->on_device_obj->grid_stride += aggregate_dim;
    }

    this->h_trim_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, trim_dims_vec);
    this->on_device_obj->d_trim_dims = this->h_trim_dims->ptr;

    this->index_on_device_obj();
  };

  void index_on_device_obj() {
    this->on_device_obj->panic_ind = this->required_int_props.find_index(
        this->properties_map.at(props.panic));
  };

public:
  std::shared_ptr<BufferDevice<REAL>> h_ranges;
  std::shared_ptr<BufferDevice<size_t>> h_dims;
  std::shared_ptr<BufferDevice<size_t>> h_trim_dims;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
};

} // namespace VANTAGE::Reactions

#endif
