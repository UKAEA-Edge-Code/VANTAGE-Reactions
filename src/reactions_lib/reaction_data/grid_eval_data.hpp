#ifndef REACTIONS_GRID_EVAL_DATA_H
#define REACTIONS_GRID_EVAL_DATA_H

#include "../reaction_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <array>
#include <memory>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief On device: ReactionData evaluating a lookup grid by computing
 * floor-point grid indices and returning the grid value at the flat index.
 *
 * @tparam input_ndim The number of input dimensions for the grid lookup.
 */
template <int input_ndim>
struct GridEvalOnDevice
    : public ReactionDataBaseOnDevice<1, DEFAULT_RNG_KERNEL, input_ndim> {

  GridEvalOnDevice() = default;

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
   * computed index.
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
 * @brief Host-side ReactionDataBase managing SYCL buffers for grid,
 * ranges, and dims, enabling on-device grid evaluation.
 *
 * @tparam input_ndim The number of input dimensions for the grid lookup.
 */
template <int input_ndim>
struct GridEval : public ReactionDataBase<GridEvalOnDevice<input_ndim>> {
  /**
   * @brief Constructor for GridEval.
   *
   * @param grid Flat vector of grid values (tabulated data).
   * @param ranges_vec Range boundaries for each dimension (used for
   * floor-point index computation).
   * @param dims_vec Grid dimensions (number of cells per axis).
   * @param sycl_target SYCL target shared pointer used for buffer
   * allocation.
   */
  GridEval(const std::vector<REAL> &grid, const std::vector<REAL> &ranges_vec,
           const std::vector<size_t> &dims_vec,
           SYCLTargetSharedPtr sycl_target) {
    this->on_device_obj = GridEvalOnDevice<input_ndim>();

    this->h_dims =
        std::make_shared<BufferDevice<size_t>>(sycl_target, dims_vec);
    this->on_device_obj->d_dims = this->h_dims->ptr;

    this->h_ranges =
        std::make_shared<BufferDevice<REAL>>(sycl_target, ranges_vec);
    this->on_device_obj->d_ranges = this->h_ranges->ptr;

    this->h_grid = std::make_shared<BufferDevice<REAL>>(sycl_target, grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;
  };

public:
  std::shared_ptr<BufferDevice<size_t>> h_dims;
  std::shared_ptr<BufferDevice<REAL>> h_ranges;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
};

} // namespace VANTAGE::Reactions

#endif // REACTIONS_GRID_EVAL_DATA_H
