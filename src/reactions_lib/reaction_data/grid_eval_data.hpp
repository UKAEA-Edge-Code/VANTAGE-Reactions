#ifndef REACTIONS_GRID_EVAL_DATA_H
#define REACTIONS_GRID_EVAL_DATA_H

#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include "reactions_lib/utils.hpp"
#include <array>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/device_buffers.hpp>
#include <neso_particles/typedefs.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation evaluating a lookup grid by
 * computing floor-point grid indices and returning the grid value at the flat
 * index.
 *
 * An input coordinate is mapped to a flat grid index as follows. The range
 * vector for dimension idim begins at offset: sum(d_dims[jdim] for jdim = 0 to
 * idim-1) and contains d_dims[idim] elements. Given an input value:
 * input[idim], the floor-point index: grid_indices[idim], is calculated. This
 * is the largest index satisfying input[idim] >= d_ranges[offset +
 * grid_indices[idim]]. The per-dimension grid_indices are combined via
 * row-major ordering into a single flat index, and the returned value is
 * d_grid[grid_flat_index].
 *
 * @tparam input_ndim The number of input dimensions for the grid lookup.
 */
template <int input_ndim>
struct CartesianGridDataOnDevice
    : public ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL, input_ndim> {
  // Alternative to static_assert for input and output type checks is to
  // just direct developers to set IN_TYPE and VAL_TYPE from
  // ReactionDataBaseOnDevice as the types for the input and return arrays of
  // calc_data. This effectively kicks the can upstream to the point when
  // calc_data is called and produces a less informative compile-time error (eg.
  // "no match between array<REAL, ...> and array<VAL_TYPE,...>") but is easier
  // for developers to implement. Happy to go with either approach.
  //
  // using IN_TYPE =
  //     typename
  //     CartesianGridDataOnDevice::ReactionDataBaseOnDevice::INPUT_TYPE;
  // using VAL_TYPE =
  //     typename
  // CartesianGridDataOnDevice::ReactionDataBaseOnDevice::VALUE_TYPE;

  /**
   * @brief Default constructor for CartesianGridDataOnDevice which contains
   * checks for signature and return type of calc_data.
   */
  CartesianGridDataOnDevice() {
    using Base = CartesianGridDataOnDevice::ReactionDataBaseOnDevice;

    using input_t =
        const std::array<typename Base::INPUT_TYPE, Base::INPUT_DIM> &;

    static_assert(
        is_calc_data_callable_v<CartesianGridDataOnDevice, input_t,
                                const Access::LoopIndex::Read &,
                                const Access::SymVector::Write<INT> &,
                                const Access::SymVector::Read<REAL> &,
                                typename DEFAULT_RNG_KERNEL::KernelType &>,
        "CartesianGridDataOnDevice::calc_data parameter signature mismatch");

    static_assert(check_calc_data_return_type<
                      CartesianGridDataOnDevice,
                      std::array<typename Base::VALUE_TYPE, Base::DIM>, input_t,
                      const Access::LoopIndex::Read &,
                      const Access::SymVector::Write<INT> &,
                      const Access::SymVector::Read<REAL> &,
                      typename DEFAULT_RNG_KERNEL::KernelType &>(),
                  "CartesianGridDataOnDevice::calc_data return type mismatch");
  };

  /**
   * @brief Constructor for CartesianGridDataOnDevice.
   *
   * @param h_grid Host buffer containing the tabulated data.
   * @param h_ranges Host buffer containing range boundaries for the
   * interpolation dimensions.
   * @param h_dims Host buffer containing grid dimensions for the
   * interpolation axes.
   */
  CartesianGridDataOnDevice(const std::shared_ptr<BufferDevice<REAL>> &h_grid,
                            const std::shared_ptr<BufferDevice<REAL>> &h_ranges,
                            const std::shared_ptr<BufferDevice<size_t>> &h_dims)
      : CartesianGridDataOnDevice() {
    d_grid = h_grid->ptr;
    d_ranges = h_ranges->ptr;
    d_dims = h_dims->ptr;
  }

  /**
   * @brief Function to compute floor-point grid indices from the input
   * coordinate and return the grid value at the computed index.
   *
   * @param input The input coordinate array of size input_ndim.
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
   * @return A REAL-valued array of size 1 containing the grid value at the
   * computed flat index.
   */
  std::array<REAL, 1> calc_data(
      const std::array<REAL, input_ndim> &input,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]] DEFAULT_RNG_KERNEL::KernelType &rng_kernel) const {
    std::array<INT, input_ndim> grid_indices;
    grid_indices[0] = interp_utils::calc_floor_point_index(
        input[0], this->d_ranges, this->d_dims[0]);
    size_t aggregate_dims = 0;
    for (size_t i = 1; i < input_ndim; i++) {
      aggregate_dims += this->d_dims[i - 1];
      grid_indices[i] = interp_utils::calc_floor_point_index(
          input[i], this->d_ranges + aggregate_dims, this->d_dims[i]);
    }

    auto grid_indices_ptr = grid_indices.data();
    INT grid_flat_index = interp_utils::coeff_index_on_device(
        grid_indices_ptr, this->d_dims, input_ndim);

    return std::array<REAL, 1>{this->d_grid[grid_flat_index]};
  }

public:
  size_t const *d_dims;
  REAL const *d_ranges;
  REAL const *d_grid;
};

/**
 * @brief Reaction rate data calculation managing buffers for grid,
 * ranges, and dims, enabling on-device grid evaluation.
 *
 * The grid evaluation works with the BufferDevice objects that are constructed
 * for the input vectors (grid, ranges_vec, dims_vec). As such, there are
 * constraints on the format of the vectors. All input vectors are 1D vectors
 * and are accessed using the logic in the on-device calc_data(...). For a given
 * input coordinate, the floor-point index in each dimension is found by
 * locating the index of the closest point in the range for that dimension that
 * is less than the input coordinate value. These per-dimension indices are then
 * flattened via row-major ordering into a single index, and the corresponding
 * value is retrieved from d_grid (device-side pointer to a host-side
 * BufferDevice that is constructed from std::vector<REAL> grid).
 *
 * @tparam input_ndim The number of input dimensions for the grid lookup.
 */
template <int input_ndim>
struct CartesianGridData
    : public ReactionDataBase<CartesianGridDataOnDevice<input_ndim>> {
  /**
   * @brief Constructor for CartesianGridData.
   *
   * @param grid Flat vector of grid values (tabulated data).
   * @param ranges_vec Range boundaries for each dimension (used for
   * floor-point index computation).
   * @param dims_vec Grid dimensions (number of grid points per axis).
   * @param sycl_target SYCL target shared pointer used for buffer
   * allocation.
   */
  CartesianGridData(const std::vector<REAL> &grid,
                    const std::vector<REAL> &ranges_vec,
                    const std::vector<size_t> &dims_vec,
                    SYCLTargetSharedPtr sycl_target) {

    auto dims_size = dims_vec.size();
    NESOASSERT((dims_size == input_ndim), "Invalid size of input dims vector.");

    auto grid_size = grid.size();
    auto ranges_size = ranges_vec.size();

    auto expected_grid_size = 1;
    auto expected_ranges_size = 0;
    for (auto &idim : dims_vec) {
      expected_grid_size *= idim;
      expected_ranges_size += idim;
    }

    NESOASSERT((ranges_size == expected_ranges_size),
               "Invalid size of input ranges vector.");
    NESOASSERT((grid_size == expected_grid_size),
               "Invalid size of input grid.");

    this->h_grid = utils::make_buffer_device_ptr(sycl_target, grid);
    this->h_ranges = utils::make_buffer_device_ptr(sycl_target, ranges_vec);
    this->h_dims = utils::make_buffer_device_ptr(sycl_target, dims_vec);

    this->on_device_obj = CartesianGridDataOnDevice<input_ndim>(
        this->h_grid, this->h_ranges, this->h_dims);
  };

public:
  std::shared_ptr<BufferDevice<REAL>> h_grid;
  std::shared_ptr<BufferDevice<REAL>> h_ranges;
  std::shared_ptr<BufferDevice<size_t>> h_dims;
};

} // namespace VANTAGE::Reactions

#endif // REACTIONS_GRID_EVAL_DATA_H
