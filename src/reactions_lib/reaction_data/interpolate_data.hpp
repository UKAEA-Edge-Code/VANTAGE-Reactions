/**
 * The following host and device classes provide a method of interpolating
 * values on an ND grid. The algorithm used follows the method described in:
 * https://www.nas.nasa.gov/assets/nas/pdf/staff/Murman_S_apnum_jun13.pdf
 *
 * In short, the algorithm finds where the interpolation points lie in the
 * discretized ND space that the grid is defined on. It then proceeds to
 * construct a hypercube around the interpolation points. The grid values at
 * each vertex of the hypercube are retrieved. With the vertex positions and
 * grid points, a recursive contraction of the hypercube, 1 dimension at a time,
 * is performed. The final interpolated function evaluation is returned (within
 * a 1D array for compatibility with ReactionDataBaseOnDevice).
 *
 * An illustrative example of contracting from 3D to 0D is shown here.
 * Each vertex is 1 index apart so if V1 is defined as an origin (0,0,0) then
 * V2 is (1,0,0), V3 is (0,1,0), etc. all the way to V8 being (1,1,1). This way
 * only the location of the origin point in the ND space is needed to find the
 * locations of the rest of the vertices. After contraction to 2D, P1 is now the
 * "origin" at (0,0) and P3 is (1,1). This quadritlateral can be thought of as a
 * slice of the preceding 3D hypercube around the point (x). Figure 2 of the
 * provided Murman paper provides another illustration of how the algorithm
 * works.
 *
 *       V7-----------V8
 *      /|           / |
 *     V5-----------V6 |
 *     | |    (x)    | |
 *     | V3----------|V4
 *     |/            |/
 *     V1-----------V2
 *
 *            |
 *            |
 *            v
 *
 *     P4-----------P3
 *      |           |
 *      |    (x)    |
 *      |           |
 *     P1-----------P2
 *
 *            |
 *            |
 *            v
 *
 *            L2
 *            |
 *           (x)
 *            |
 *            L1
 *
 *            |
 *            |
 *            v
 *
 *           (x)
 *
 * The underlying maths of the contraction (simplified here) is:
 * f(P1) = linear_interp(x(0), V1, V5, f(V1), f(V5))
 * f(P4) = linear_interp(x(0), V3, V7, f(V3), f(V7))
 * f(P2) = linear_interp(x(0), V2, V6, f(V2), f(V6))
 * f(P3) = linear_interp(x(0), V4, V8, f(V4), f(V8))
 *
 * then
 *
 * f(L1) = linear_interp(x(1), P1, P2, f(P1), f(P2))
 * f(L2) = linear_interp(x(1), P4, P3, f(P4), f(P3))
 *
 * finally
 *
 * f(x) = linear_interp(x(2), L1, L2, f(L1), f(L2))
 *
 * Note x(0), x(1) and x(2) simply refers to the components of the 3D vector x
 * corresponding to the dimension that's being contracted.
 */

#ifndef REACTIONS_COMPOSITE_INTERPOLATE_DATA_H
#define REACTIONS_COMPOSITE_INTERPOLATE_DATA_H
#include "reactions_lib/composite_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <array>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/containers/tuple.hpp>
#include <neso_particles/device_buffers.hpp>
#include <neso_particles/typedefs.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {
constexpr auto INF_INTERP_DOUBLE = std::numeric_limits<double>::infinity();

/**
 * @brief Enum class containing possible modes for extrapolation for
 * InterpolateData.
 */
enum class ExtrapolationType { continue_linear, clamp_to_zero, clamp_to_edge };

/**
 * @brief On device: ReactionData calculating an interpolated function
 * evaluation given a set of interpolation points (provided on-device by the
 * particles on a per-particle basis) and a pre-calculated grid of function
 * evaluations.
 *
 * @tparam input_ndim The number of dimensions that correspond to the number of
 * interpolation points.
 */
template <size_t output_ndim, size_t interp_ndim, size_t rng_ndim,
          typename OUT_TYPE, typename IN_TYPE, typename DATATYPE>
struct InterpolateDataOnDevice
    : public CompositeDataOnDevice<output_ndim, interp_ndim + rng_ndim,
                                   OUT_TYPE, IN_TYPE, DATATYPE> {

  InterpolateDataOnDevice() = default;
  /**
   * @brief Constructor for InterpolateDataOnDevice.
   *
   * @param extrapolation_type The extrapolation type to fall back on if
   * interpolation is not possible for a set of points.
   */
  InterpolateDataOnDevice(
      DATATYPE interp_data,
      ExtrapolationType extrapolation_type = ExtrapolationType::continue_linear)
      : CompositeDataOnDevice<output_ndim, interp_ndim + rng_ndim, OUT_TYPE,
                              IN_TYPE, DATATYPE>(interp_data) {
    switch (extrapolation_type) {
    case VANTAGE::Reactions::ExtrapolationType::continue_linear:
      this->continue_linear = true;
      break;
    case VANTAGE::Reactions::ExtrapolationType::clamp_to_zero:
      this->clamp_to_zero = true;
      break;
    case VANTAGE::Reactions::ExtrapolationType::clamp_to_edge:
      this->clamp_to_edge = true;
      break;
    }
  };

  /**
   * @brief Function to calculate interpolated function evaluations.
   *
   * @param interpolation_points An array containing the interpolation points
   * at which a function evaluation is desired (for example the points may be
   * provided by the values of properties on a particle for which calc_data is
   * being run).
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size 1 that contains the interpolated
   * function evaluation at the given interpolation points.
   */
  std::array<REAL, output_ndim> calc_data(
      const std::array<REAL, interp_ndim + rng_ndim> &interpolation_points,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]]
      typename TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>>::
          KernelType &kernel) const {

    std::array<REAL, interp_ndim> mut_interpolation_points;
    for (int i = 0; i < interp_ndim; i++) {
      mut_interpolation_points[i] = interpolation_points[i];
    }

    std::array<INT, interp_ndim> origin_indices;

    std::array<REAL, initial_num_points * output_ndim> vertex_func_evals;
    std::array<INT, initial_num_points> vertex_coord;

    std::array<REAL, initial_num_points * output_ndim> output_evals;
    std::array<INT, initial_num_points> varying_dim;

    for (size_t i = 0; i < interp_ndim; i++) {
      origin_indices[i] = 0;
    }

    for (size_t i = 0; i < initial_num_points; i++) {
      vertex_func_evals[i] = 0.0;
      vertex_coord[i] = 0;

      output_evals[i] = 0.0;
      varying_dim[i] = 0;
    }

    // Counter
    INT num_points = this->initial_num_points;

    // Calculation of the indices that will form the "origin" of the
    // hypercube. These are the smallest indices in each dimension that
    // still have coordinate values that are less than the interpolation
    // values in that dimension (ie. to the left of the interpolation point in
    // every dimension).
    // Note that the actual ranges for the dimensions used in this calculation
    // are the "extended" versions which are padded with -INF_INTERP_DOUBLE
    // below their lower bounds and +INF_INTERP_DOUBLE above their upper
    // bounds. This is for the sake of aiding in extrapolation handling and is
    // reset after extrapolation handling.
    for (size_t i = 0; i < interp_ndim; i++) {
      origin_indices[i] = interp_utils::calc_floor_point_index(
          mut_interpolation_points[i],
          this->d_extended_ranges_vec + this->d_extended_ranges_strides[i],
          this->d_extended_dims_vec[i] - 1);
    }

    // Out-of-range clamping handling
    bool out_of_range = false;
    bool above_range = false;
    bool below_range = false;

    REAL above_clamp_to_edge = 0.0;
    REAL below_clamp_to_edge = 0.0;

    bool out_of_range_clamp_to_zero = false;

    for (size_t i = 0; i < interp_ndim; i++) {
      above_range = (origin_indices[i] == (this->d_extended_dims_vec[i] - 2));
      below_range = (origin_indices[i] == 0);

      out_of_range = (above_range || below_range);

      above_clamp_to_edge = this->d_ranges_vec[this->d_dims_vec[i] - 1 +
                                               this->d_ranges_strides[i]];
      below_clamp_to_edge = this->d_ranges_vec[this->d_ranges_strides[i]];

      mut_interpolation_points[i] = (above_range && this->clamp_to_edge)
                                        ? above_clamp_to_edge
                                        : ((below_range && this->clamp_to_edge)
                                               ? below_clamp_to_edge
                                               : mut_interpolation_points[i]);

      out_of_range_clamp_to_zero = (out_of_range && this->clamp_to_zero);
    }

    // Limit origin_indices to be between the standard dimensional ranges.
    // Note that the upper limit is set by this->d_dims_vec[i] - 2 since that
    // represents the penultimate element in the standard dimensional range
    // which is the last left-most index that can be selected such that the
    // linear gradient can be calculated.
    for (size_t i = 0; i < interp_ndim; i++) {
      origin_indices[i]--;
      origin_indices[i] = Kernel::min(Kernel::max(origin_indices[i], 0),
                                      this->d_dims_vec[i] - 2);
    }

    // Necessary for using the interp_utils functions.
    auto mut_interpolation_points_ptr = mut_interpolation_points.data();
    auto origin_indices_ptr = origin_indices.data();

    auto vertex_func_evals_ptr = vertex_func_evals.data();
    auto vertex_coord_ptr = vertex_coord.data();

    auto output_evals_ptr = output_evals.data();
    auto varying_dim_ptr = varying_dim.data();

    auto interp_data = Tuple::get<0>(this->data);

    std::array<REAL, rng_ndim> trim_indices;
    for (int i = 0; i < rng_ndim; i++) {
      trim_indices[i] = interpolation_points[i + interp_ndim];
    }

    // Initial function evaluation (ie values of the coeffs_vec) based on the
    // origin_indices and the vertices of the hypercube.
    interp_utils::initial_func_eval_on_device<decltype(interp_data),
                                              output_ndim, rng_ndim>(
        vertex_func_evals_ptr, vertex_coord_ptr, interp_data,
        this->d_hypercube_vertices, origin_indices_ptr, this->d_dims_vec,
        interp_ndim, num_points, trim_indices, this->d_trim_dims, index,
        req_int_props, req_real_props, kernel);

    // Array of length 1 to maintain compatibility with pipelining interface
    // for ReactionData objects.
    std::array<REAL, output_ndim> calculated_interpolated_vals;
    for (int i = 0; i < output_ndim; i++) {
      calculated_interpolated_vals[i] = 0.0;
    }

    // Loop until the last dimension (down to 0D)
    // Note that despite the dim_index = input_ndim assignment, the loop
    // actually starts at dim_index = (input_ndim - 1) , as desired, due to
    // the decrement and store during the first check against 0.
    for (size_t dim_index = interp_ndim; dim_index-- > 0;) {

      // Contract the hypercube vertices and evaluations by performing linear
      // interpolation on the current dimension (denoted by dim_index). The
      // contraction is expressed by the reduction in the length of the
      // output_evals (ie. for 3D to 2D, the vertex_func_evals would be of
      // length 8 and output_evals would be of length 4).
      interp_utils::contract_hypercube_on_device<output_ndim>(
          mut_interpolation_points_ptr, dim_index, this->d_hypercube_vertices,
          origin_indices_ptr, vertex_func_evals_ptr, this->d_ranges_vec,
          this->d_dims_vec, output_evals_ptr, varying_dim_ptr,
          vertex_coord_ptr);

      // This now accounts for the smaller size of output_evals, and makes
      // sure that any loops in future contract_hypercube(...) invocations
      // remain consistent.
      num_points = num_points >> 1;

      // Reset vertex_func_evals for the next contraction
      for (int i = 0; i < num_points; i++) {
        for (int idim = 0; idim < output_ndim; idim++) {
          vertex_func_evals[(i * output_ndim) + idim] =
              output_evals[(i * output_ndim) + idim];
        }
      }
    }

    // Return either 0 or vertex_func_evals[0] depending on the
    // out_of_range_clamp_to_zero boolean value.
    for (int idim = 0; idim < output_ndim; idim++) {
      calculated_interpolated_vals[idim] =
          out_of_range_clamp_to_zero ? calculated_interpolated_vals[idim]
                                     : vertex_func_evals[idim];
    }

    return calculated_interpolated_vals;
  }

public:
  INT const *d_hypercube_vertices;
  size_t const *d_dims_vec;
  REAL const *d_ranges_vec;
  REAL const *d_grid;
  REAL const *d_extended_ranges_vec;
  size_t const *d_extended_dims_vec;
  size_t const *d_ranges_strides;
  size_t const *d_extended_ranges_strides;
  INT const *d_trim_dims;

  static constexpr INT initial_num_points = 1 << interp_ndim;
  static constexpr INT num_randoms = DATATYPE::INPUT_DIM;

  bool continue_linear = false;
  bool clamp_to_zero = false;
  bool clamp_to_edge = false;
};

/**
 * @brief ReactionData calculating an interpolated function evaluation given a
 * set of interpolation points (provided on-device by the particles on a
 * per-particle basis) and a pre-calculated grid of function evaluations.
 *
 * @tparam input_ndim The number of dimensions that correspond to the number of
 * interpolation points.
 */

template <size_t output_ndim, size_t interp_ndim, typename OUT_TYPE,
          typename IN_TYPE, typename DATATYPE, size_t rng_ndim = 0>
struct InterpolateData
    : public CompositeData<InterpolateDataOnDevice<
                               output_ndim, interp_ndim, rng_ndim, OUT_TYPE,
                               IN_TYPE, typename DATATYPE::ON_DEVICE_OBJ_TYPE>,
                           output_ndim, interp_ndim + rng_ndim, DATATYPE> {
  /**
   * @brief Constructor for InterpolateData
   *
   * @param dims_vec A vector containing the lengths of each dimension that
   * defines the grid of pre-computed values.
   * @param ranges_vec A vector that contains the range of values for
   * each axis that defines the grid of pre-computed values. The values in
   * ranges_vec can be thought of as a set of concatenated arrays where each
   * segment's length within the 1D ranges_vec is defined in dims_vec.
   * @param grid A contiguous row-major vector that contains the pre-computed
   * function evaluations on a grid of input_ndim dimensions. The flattening
   * of this data follows the standard C-style flattening of multi-dimensional
   * data into a 1D array. An explanation of this flattening can be found
   * either:
   * https://en.wikipedia.org/wiki/Array_(data_structure)#Multidimensional_arrays
   * or
   * https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param extrapolation_type The extrapolation type to fall back on if
   * interpolation is not possible for a set of points. Either
   * continue_linear, clamp_to_zero or clamp_to_edge.
   */
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::vector<REAL> &grid,
                  SYCLTargetSharedPtr sycl_target, const DATATYPE &interp_data,
                  const ExtrapolationType &extrapolation_type =
                      ExtrapolationType::continue_linear,
                  const std::vector<INT> &trim_dims_vec = {})
      : CompositeData<InterpolateDataOnDevice<
                          output_ndim, interp_ndim, rng_ndim, OUT_TYPE, IN_TYPE,
                          typename DATATYPE::ON_DEVICE_OBJ_TYPE>,
                      output_ndim, interp_ndim + rng_ndim, DATATYPE>(
            interp_data),
        dims_vec(dims_vec), ranges_vec(ranges_vec), grid(grid),
        trim_dims_vec(trim_dims_vec), sycl_target(sycl_target),
        extrapolation_type(extrapolation_type) {
    this->post_init();
    NESOASSERT(trim_dims_vec.size() == rng_ndim,
               "Ensure that the number of elements in the trim_dims_vec "
               "argument matches the rng_ndim template parameter.");
  };

  void index_on_device_object() override {
    /**
     * A series of vertices of an N-Dimensional
     * hypercube are constructed here. Note that the hypercube vertices
     * produced don't actually have any direct mapping onto the values of the
     * dimensions of the grid (ie. the x-values or y-values), rather it can be
     * thought of as a stencil to be used with the origin_indices array that's
     * calculated in InterpolateDataOnDevice.calc_data. More details on the
     * construct_initial_hypercube can be found in the docstring in
     * interp_utils.hpp.
     */
    auto initial_hypercube =
        interp_utils::construct_initial_hypercube(interp_ndim);

    this->on_device_obj =
        InterpolateDataOnDevice<output_ndim, interp_ndim, rng_ndim, OUT_TYPE,
                                IN_TYPE, typename DATATYPE::ON_DEVICE_OBJ_TYPE>(
            std::get<0>(this->data).get_on_device_obj(),
            this->extrapolation_type);

    // BufferDevice<REAL> mock setup
    this->h_dims_vec = std::make_shared<BufferDevice<size_t>>(this->sycl_target,
                                                              this->dims_vec);
    this->on_device_obj->d_dims_vec = this->h_dims_vec->ptr;

    std::vector<size_t> ranges_strides(interp_ndim);
    std::vector<size_t> extended_dims_vec(interp_ndim);
    std::vector<size_t> extended_ranges_strides(interp_ndim);
    for (size_t idim = 0; idim < interp_ndim; idim++) {
      extended_dims_vec[idim] = this->dims_vec[idim] + 2;
      for (size_t j = 0; j < idim; j++) {
        ranges_strides[idim] += this->dims_vec[j];
        extended_ranges_strides[idim] += extended_dims_vec[j];
      }
    }

    std::vector<REAL> extended_ranges_vec;
    for (size_t idim = 0; idim < interp_ndim; idim++) {
      extended_ranges_vec.push_back(-INF_INTERP_DOUBLE);
      for (size_t irange = 0; irange < this->dims_vec[idim]; irange++) {
        extended_ranges_vec.push_back(
            ranges_vec[irange + ranges_strides[idim]]);
      }
      extended_ranges_vec.push_back(INF_INTERP_DOUBLE);
    }

    this->h_extended_ranges_vec = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, extended_ranges_vec);
    this->on_device_obj->d_extended_ranges_vec =
        this->h_extended_ranges_vec->ptr;

    this->h_extended_dims_vec = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, extended_dims_vec);
    this->on_device_obj->d_extended_dims_vec = this->h_extended_dims_vec->ptr;

    this->h_ranges_vec = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                              this->ranges_vec);
    this->on_device_obj->d_ranges_vec = this->h_ranges_vec->ptr;

    this->h_ranges_strides = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, ranges_strides);
    this->on_device_obj->d_ranges_strides = this->h_ranges_strides->ptr;

    this->h_extended_ranges_strides = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, extended_ranges_strides);
    this->on_device_obj->d_extended_ranges_strides =
        this->h_extended_ranges_strides->ptr;

    this->h_grid =
        std::make_shared<BufferDevice<REAL>>(this->sycl_target, this->grid);
    this->on_device_obj->d_grid = this->h_grid->ptr;

    this->h_hypercube_vertices = std::make_shared<BufferDevice<INT>>(
        this->sycl_target, initial_hypercube);
    this->on_device_obj->d_hypercube_vertices = this->h_hypercube_vertices->ptr;

    this->h_trim_dims = std::make_shared<BufferDevice<INT>>(
        this->sycl_target, this->trim_dims_vec);
    this->on_device_obj->d_trim_dims = this->h_trim_dims->ptr;
  }

  SYCLTargetSharedPtr sycl_target;
  std::vector<size_t> dims_vec;
  std::vector<REAL> ranges_vec;
  std::vector<REAL> grid;
  std::vector<INT> trim_dims_vec;
  ExtrapolationType extrapolation_type;

  std::shared_ptr<BufferDevice<size_t>> h_dims_vec;
  std::shared_ptr<BufferDevice<REAL>> h_ranges_vec;
  std::shared_ptr<BufferDevice<REAL>> h_extended_ranges_vec;
  std::shared_ptr<BufferDevice<size_t>> h_extended_dims_vec;
  std::shared_ptr<BufferDevice<size_t>> h_ranges_strides;
  std::shared_ptr<BufferDevice<size_t>> h_extended_ranges_strides;
  std::shared_ptr<BufferDevice<REAL>> h_grid;
  std::shared_ptr<BufferDevice<INT>> h_hypercube_vertices;
  std::shared_ptr<BufferDevice<INT>> h_trim_dims;
};
}; // namespace VANTAGE::Reactions
#endif
