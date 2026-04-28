/**
 * The following host and device classes provide a method of interpolating
 * values on an ND grid (represented by evaluations of a given ReactionData
 * derived object). The algorithm used follows the method described in:
 * https://www.nas.nasa.gov/assets/nas/pdf/staff/Murman_S_apnum_jun13.pdf
 *
 * In short, the algorithm finds where the interpolation points lie in the
 * discretized ND space that the grid is defined on. It then proceeds to
 * construct a hypercube around the interpolation points. The grid values(which
 * may be multi-dimensional) at each vertex of the hypercube are retrieved. With
 * the vertex coordinates and function evaluations at the vertices, a recursive
 * contraction of the hypercube, 1 dimension at a time, is performed. The final
 * interpolated function evaluation is returned.
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

// Type discipline for indices:
//   size_t  — API boundaries, container sizes, stride values
//   INT     — internal device computation (subtraction, decrement, clamp)
#ifndef REACTIONS_COMPOSITE_INTERPOLATE_DATA_H
#define REACTIONS_COMPOSITE_INTERPOLATE_DATA_H
#include "reactions_lib/composite_data.hpp"
#include "reactions_lib/interp_utils.hpp"
#include <algorithm>
#include <memory>
#include <neso_particles.hpp>

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
 * evaluation given a set of interpolation points and a ReactionDataBaseOnDevice
 * derived object.
 *
 * @tparam output_ndim The number of dimensions that correspond to the output of
 * calc_data from DATATYPE.
 * @tparam interp_ndim The number of dimensions that correspond to the number of
 * interpolation points.
 * @tparam non_interp_ndim The number of dimensions that are not interpolated
 * and are used by calc_data from DATATYPE.
 * @tparam DATATYPE ReactionDataBaseOnDevice derived type corresponding to the
 * on-device grid-function evaluation reaction data object.
 */
template <size_t output_ndim, size_t interp_ndim, size_t non_interp_ndim,
          typename DATATYPE>
struct InterpolateDataOnDevice
    : public CompositeDataOnDevice<output_ndim, interp_ndim + non_interp_ndim,
                                   REAL, REAL, DATATYPE> {

  InterpolateDataOnDevice() = default;
  /**
   * @brief Constructor for InterpolateDataOnDevice.
   *
   * @param interp_data ReactionDataBaseOnDevice derived object corresponding to
   * the grid-function evaluation reaction data.
   * @param interp_indices Indices that correspond to interpolation dimensions
   * of the full input array that will be passed to calc_data.
   * @param extrapolation_type The extrapolation type to fall back on if
   * interpolation is not possible for a set of points.
   */
  InterpolateDataOnDevice(
      DATATYPE interp_data,
      const std::array<size_t, interp_ndim> &interp_indices,
      ExtrapolationType extrapolation_type = ExtrapolationType::continue_linear)
      : interp_indices(interp_indices),
        CompositeDataOnDevice<output_ndim, interp_ndim + non_interp_ndim, REAL,
                              REAL, DATATYPE>(interp_data) {
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

    if constexpr (non_interp_ndim > 0) {
      size_t j = 0;
      for (size_t i = 0; i < this->total_ndim; i++) {
        if (std::find(interp_indices.begin(), interp_indices.end(), i) ==
            interp_indices.end()) {
          this->non_interp_indices[j++] = i;
        }
      }
    }
  };

  /**
   * @brief Function to calculate interpolated function evaluations.
   *
   * @param interpolation_points An array containing all of the values needed
   * for grid-function evaluation. (Both the interpolation points as well as
   * pass-through values)
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation (the kernel type is inherited from the kernel type for
   * DATATYPE)
   *
   * @return A REAL-valued array of size output_ndim that contains the
   * interpolated function evaluation at the given interpolation points.
   */
  std::array<REAL, output_ndim> calc_data(
      const std::array<REAL, interp_ndim + non_interp_ndim> &input_array,
      [[maybe_unused]] const Access::LoopIndex::Read &index,
      [[maybe_unused]] const Access::SymVector::Write<INT> &req_int_props,
      [[maybe_unused]] const Access::SymVector::Read<REAL> &req_real_props,
      [[maybe_unused]]
      typename TupleRNG<std::shared_ptr<typename DATATYPE::RNG_KERNEL_TYPE>>::
          KernelType &kernel) const {

    std::array<REAL, interp_ndim> mut_interpolation_points;
    for (size_t i = 0; i < interp_ndim; i++) {
      mut_interpolation_points[i] = input_array[this->interp_indices[i]];
    }

    std::array<REAL, non_interp_ndim> non_interpolation_points;
    for (size_t i = 0; i < non_interp_ndim; i++) {
      non_interpolation_points[i] = input_array[non_interp_indices[i]];
    }

    std::array<INT, interp_ndim> origin_indices;

    std::array<REAL, initial_num_points * output_ndim> vertex_func_evals;
    std::array<INT, interp_ndim> vertex_coord;

    std::array<REAL, initial_num_points * output_ndim> output_evals;
    std::array<INT, initial_num_points> varying_dim;

    for (size_t i = 0; i < interp_ndim; i++) {
      origin_indices[i] = 0;
      vertex_coord[i] = 0;
    }

    for (size_t i = 0; i < initial_num_points; i++) {
      vertex_func_evals[i] = 0.0;

      output_evals[i] = 0.0;
      varying_dim[i] = 0;
    }

    // Counter
    INT num_points = static_cast<INT>(this->initial_num_points);

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
      origin_indices[i] = static_cast<INT>(interp_utils::calc_floor_point_index(
          mut_interpolation_points[i],
          this->d_extended_ranges_vec_ptr +
              this->d_extended_ranges_strides_ptr[i],
          this->d_extended_dims_vec_ptr[i] - 1));
    }

    // Out-of-range clamping handling
    bool out_of_range = false;
    bool above_range = false;
    bool below_range = false;

    REAL above_clamp_to_edge = 0.0;
    REAL below_clamp_to_edge = 0.0;

    bool out_of_range_clamp_to_zero = false;

    for (size_t i = 0; i < interp_ndim; i++) {
      above_range = (origin_indices[i] ==
                     static_cast<INT>(this->d_extended_dims_vec_ptr[i] - 2));
      below_range = (origin_indices[i] == 0);

      out_of_range = (above_range || below_range);

      above_clamp_to_edge =
          this->d_ranges_vec_ptr[this->d_dims_vec_ptr[i] - 1 +
                                 this->d_ranges_strides_ptr[i]];
      below_clamp_to_edge =
          this->d_ranges_vec_ptr[this->d_ranges_strides_ptr[i]];

      mut_interpolation_points[i] = (above_range && this->clamp_to_edge)
                                        ? above_clamp_to_edge
                                        : ((below_range && this->clamp_to_edge)
                                               ? below_clamp_to_edge
                                               : mut_interpolation_points[i]);

      out_of_range_clamp_to_zero = (out_of_range && this->clamp_to_zero);
    }

    // Limit origin_indices to be between the standard dimensional ranges.
    // Note that the upper limit is set by this->d_dims_vec_ptr[i] - 2 since
    // that represents the penultimate element in the standard dimensional range
    // which is the last left-most index that can be selected such that the
    // linear gradient can be calculated.
    for (size_t i = 0; i < interp_ndim; i++) {
      origin_indices[i]--;
      origin_indices[i] =
          Kernel::clamp(origin_indices[i], static_cast<INT>(0),
                        static_cast<INT>(this->d_dims_vec_ptr[i] - 2));
    }

    // Necessary for using the interp_utils functions.

    auto mut_interpolation_points_ptr = mut_interpolation_points.data();

    auto origin_indices_ptr = origin_indices.data();

    auto vertex_func_evals_ptr = vertex_func_evals.data();
    auto vertex_coord_ptr = vertex_coord.data();

    auto output_evals_ptr = output_evals.data();
    auto varying_dim_ptr = varying_dim.data();

    auto interp_data = Tuple::get<0>(this->data);

    // Initial function evaluation (ie values of the coeffs_vec) based on the
    // origin_indices and the vertices of the hypercube and any data needed for
    // DATATYPE.calc_data(...).
    interp_utils::initial_func_eval_on_device<
        decltype(interp_data), output_ndim, interp_ndim, non_interp_ndim>(
        vertex_func_evals_ptr, vertex_coord_ptr, interp_data,
        origin_indices_ptr, this->d_hypercube_vertices_ptr,
        this->d_ranges_vec_ptr, non_interpolation_points, this->interp_indices,
        this->non_interp_indices, this->d_dims_vec_ptr, index, req_int_props,
        req_real_props, kernel);

    std::array<REAL, output_ndim> calculated_interpolated_vals;
    for (size_t i = 0; i < output_ndim; i++) {
      calculated_interpolated_vals[i] = 0.0;
    }

    // Loop until the last dimension (down to 0D)
    // Starts at dim_index = (interp_ndim - 1) and decrements to 0.
    for (INT dim_index = static_cast<INT>(interp_ndim) - 1; dim_index >= 0;
         dim_index--) {

      // Contract the hypercube vertices and evaluations by performing linear
      // interpolation on the current dimension (denoted by dim_index). The
      // contraction is expressed by the reduction in the length of the
      // output_evals (ie. for 3D to 2D, the vertex_func_evals would be of
      // length 8 and output_evals would be of length 4).
      interp_utils::contract_hypercube_on_device<output_ndim>(
          mut_interpolation_points_ptr, dim_index,
          this->d_hypercube_vertices_ptr, origin_indices_ptr,
          vertex_func_evals_ptr, this->d_ranges_vec_ptr, this->d_dims_vec_ptr,
          output_evals_ptr, varying_dim_ptr, vertex_coord_ptr);

      // This now accounts for the smaller size of output_evals, and makes
      // sure that any loops in future contract_hypercube(...) invocations
      // remain consistent.
      num_points = num_points >> 1;

      // Reset vertex_func_evals for the next contraction
      for (size_t i = 0; i < static_cast<size_t>(num_points); i++) {
        for (size_t idim = 0; idim < output_ndim; idim++) {
          vertex_func_evals[(i * output_ndim) + idim] =
              output_evals[(i * output_ndim) + idim];
        }
      }
    }

    for (size_t idim = 0; idim < output_ndim; idim++) {
      calculated_interpolated_vals[idim] =
          out_of_range_clamp_to_zero ? calculated_interpolated_vals[idim]
                                     : vertex_func_evals[idim];
    }

    return calculated_interpolated_vals;
  }

public:
  size_t const *d_hypercube_vertices_ptr;
  size_t const *d_dims_vec_ptr;
  REAL const *d_ranges_vec_ptr;
  REAL const *d_extended_ranges_vec_ptr;
  size_t const *d_extended_dims_vec_ptr;
  size_t const *d_ranges_strides_ptr;
  size_t const *d_extended_ranges_strides_ptr;

  std::array<size_t, interp_ndim> interp_indices;
  std::array<size_t, non_interp_ndim> non_interp_indices;

  static constexpr size_t initial_num_points = 1 << interp_ndim;
  static constexpr size_t total_ndim = interp_ndim + non_interp_ndim;

  bool continue_linear = false;
  bool clamp_to_zero = false;
  bool clamp_to_edge = false;
};

/**
 * @brief ReactionData calculating an interpolated function evaluation given a
 * set of interpolation points and a ReactionDataBase derived object.
 *
 * The input vector contains both the interpolation points and
 * the pass-through values. The interpolation points are selected using
 * interp_indices. The remaining entries in the input vector are passed through
 * unchanged to the underlying reaction data object.
 *
 * @tparam output_ndim The number of dimensions that correspond to the output of
 * calc_data from DATATYPE.
 * @tparam interp_ndim The number of dimensions that correspond to the number of
 * interpolation points.
 * @tparam DATATYPE ReactionDataBase derived type corresponding to the
 * grid-function evaluation reaction data object.
 * @tparam non_interp_ndim The number of dimensions that are not interpolated
 * and are used by calc_data from DATATYPE.
 */

template <size_t output_ndim, size_t interp_ndim, typename DATATYPE,
          size_t non_interp_ndim = 0>
struct InterpolateData
    : public CompositeData<
          InterpolateDataOnDevice<output_ndim, interp_ndim, non_interp_ndim,
                                  typename DATATYPE::ON_DEVICE_OBJ_TYPE>,
          output_ndim, interp_ndim + non_interp_ndim, DATATYPE> {
  /**
   * @brief Constructor for InterpolateData
   *
   * @param dims_vec A vector containing the lengths of each dimension that
   * defines the grid of pre-computed values.
   * @param ranges_vec A vector that contains the range of values for
   * each axis that defines the grid of pre-computed values. The values in
   * ranges_vec can be thought of as a set of concatenated arrays where each
   * segment's length within the 1D ranges_vec is defined in dims_vec.
   * @param interp_indices An array of indices that correspond to the indices of
   * the full input array that will be passed to calc_data that are to be
   * interpolated.
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param interp_data ReactionDataBase derived object corresponding to
   * the grid-function evaluation reaction data object.
   * @param extrapolation_type The extrapolation type to fall back on if
   * interpolation is not possible for a set of points. Either
   * continue_linear, clamp_to_zero or clamp_to_edge.
   */
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::array<size_t, interp_ndim> &interp_indices,
                  SYCLTargetSharedPtr sycl_target, const DATATYPE &interp_data,
                  const ExtrapolationType &extrapolation_type)
      : CompositeData<
            InterpolateDataOnDevice<output_ndim, interp_ndim, non_interp_ndim,
                                    typename DATATYPE::ON_DEVICE_OBJ_TYPE>,
            output_ndim, interp_ndim + non_interp_ndim, DATATYPE>(interp_data),
        dims_vec(dims_vec), ranges_vec(ranges_vec),
        interp_indices(interp_indices), sycl_target(sycl_target),
        extrapolation_type(extrapolation_type) {
    this->post_init();
  };

  /**
   * \overload
   * @brief Constructor for InterpolateData that takes the usual arguments but
   * without extrapolation_type (set to continue_linear)
   *
   * @param dims_vec A vector containing the lengths of each dimension that
   * defines the grid of pre-computed values.
   * @param ranges_vec A vector that contains the range of values for
   * each axis that defines the grid of pre-computed values. The values in
   * ranges_vec can be thought of as a set of concatenated arrays where each
   * segment's length within the 1D ranges_vec is defined in dims_vec.
   * @param interp_indices An array of indices that correspond to the indices
   * of the full input array that will be passed to calc_data that are to be
   * interpolated.
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param interp_data ReactionDataBase derived object corresponding to
   * the grid-function evaluation reaction data object.
   */
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  const std::array<size_t, interp_ndim> &interp_indices,
                  SYCLTargetSharedPtr sycl_target, const DATATYPE &interp_data)
      : InterpolateData(dims_vec, ranges_vec, interp_indices, sycl_target,
                        interp_data, ExtrapolationType::continue_linear) {};

  /**
   * \overload
   * @brief Constructor for InterpolateData that takes the usual arguments but
   * without interp_indices (set to an array with values from 0 to inter_ndim,
   * ie. all dimensions are to be interpolated) and extrapolation_type (set to
   * continue_linear).
   *
   * @param dims_vec A vector containing the lengths of each dimension that
   * defines the grid of pre-computed values.
   * @param ranges_vec A vector that contains the range of values for
   * each axis that defines the grid of pre-computed values. The values in
   * ranges_vec can be thought of as a set of concatenated arrays where each
   * segment's length within the 1D ranges_vec is defined in dims_vec.
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param interp_data ReactionDataBase derived object corresponding to
   * the grid-function evaluation reaction data object.
   */
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  SYCLTargetSharedPtr sycl_target, const DATATYPE &interp_data)
      : InterpolateData(dims_vec, ranges_vec, std::array<size_t, interp_ndim>(),
                        sycl_target, interp_data) {
    for (size_t i = 0; i < interp_ndim; i++)
      this->interp_indices[i] = i;
  };

  /**
   * \overload
   * @brief Constructor for InterpolateData that takes the usual arguments but
   * without interp_indices (set to an array with values from 0 to inter_ndim,
   * ie. all dimensions are to be interpolated).
   *
   * @param dims_vec A vector containing the lengths of each dimension that
   * defines the grid of pre-computed values.
   * @param ranges_vec A vector that contains the range of values for
   * each axis that defines the grid of pre-computed values. The values in
   * ranges_vec can be thought of as a set of concatenated arrays where each
   * segment's length within the 1D ranges_vec is defined in dims_vec.
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param interp_data ReactionDataBase derived object corresponding to
   * the grid-function evaluation reaction data object.
   * @param extrapolation_type The extrapolation type to fall back on if
   * interpolation is not possible for a set of points. Either
   * continue_linear, clamp_to_zero or clamp_to_edge.
   */
  InterpolateData(const std::vector<size_t> &dims_vec,
                  const std::vector<REAL> &ranges_vec,
                  SYCLTargetSharedPtr sycl_target, const DATATYPE &interp_data,
                  const ExtrapolationType &extrapolation_type)
      : InterpolateData(dims_vec, ranges_vec, std::array<size_t, interp_ndim>(),
                        sycl_target, interp_data, extrapolation_type) {
    for (size_t i = 0; i < interp_ndim; i++)
      this->interp_indices[i] = i;
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
        InterpolateDataOnDevice<output_ndim, interp_ndim, non_interp_ndim,
                                typename DATATYPE::ON_DEVICE_OBJ_TYPE>(
            std::get<0>(this->data).get_on_device_obj(), this->interp_indices,
            this->extrapolation_type);

    // BufferDevice<REAL> mock setup
    this->d_dims_vec = std::make_shared<BufferDevice<size_t>>(this->sycl_target,
                                                              this->dims_vec);
    this->on_device_obj->d_dims_vec_ptr = this->d_dims_vec->ptr;

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

    this->d_extended_ranges_vec = std::make_shared<BufferDevice<REAL>>(
        this->sycl_target, extended_ranges_vec);
    this->on_device_obj->d_extended_ranges_vec_ptr =
        this->d_extended_ranges_vec->ptr;

    this->d_extended_dims_vec = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, extended_dims_vec);
    this->on_device_obj->d_extended_dims_vec_ptr =
        this->d_extended_dims_vec->ptr;

    this->d_ranges_vec = std::make_shared<BufferDevice<REAL>>(this->sycl_target,
                                                              this->ranges_vec);
    this->on_device_obj->d_ranges_vec_ptr = this->d_ranges_vec->ptr;

    this->d_ranges_strides = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, ranges_strides);
    this->on_device_obj->d_ranges_strides_ptr = this->d_ranges_strides->ptr;

    this->d_extended_ranges_strides = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, extended_ranges_strides);
    this->on_device_obj->d_extended_ranges_strides_ptr =
        this->d_extended_ranges_strides->ptr;

    this->d_hypercube_vertices = std::make_shared<BufferDevice<size_t>>(
        this->sycl_target, initial_hypercube);
    this->on_device_obj->d_hypercube_vertices_ptr =
        this->d_hypercube_vertices->ptr;
  }

  SYCLTargetSharedPtr sycl_target;
  std::vector<size_t> dims_vec;
  std::vector<REAL> ranges_vec;
  std::array<size_t, interp_ndim> interp_indices;
  ExtrapolationType extrapolation_type;

  std::shared_ptr<BufferDevice<size_t>> d_dims_vec;
  std::shared_ptr<BufferDevice<REAL>> d_ranges_vec;
  std::shared_ptr<BufferDevice<REAL>> d_extended_ranges_vec;
  std::shared_ptr<BufferDevice<size_t>> d_extended_dims_vec;
  std::shared_ptr<BufferDevice<size_t>> d_ranges_strides;
  std::shared_ptr<BufferDevice<size_t>> d_extended_ranges_strides;
  std::shared_ptr<BufferDevice<size_t>> d_hypercube_vertices;
};
}; // namespace VANTAGE::Reactions
#endif
