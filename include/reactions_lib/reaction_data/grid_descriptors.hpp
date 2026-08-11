#ifndef REACTIONS_GRID_DESCRIPTOR_H
#define REACTIONS_GRID_DESCRIPTOR_H

#include "../../reactions/neso_particles_namespace_alias.hpp"
#include "../../reactions/neso_test_assert.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

#include <type_traits>
#include <vector>

namespace VANTAGE::Reactions {

namespace grid_utils {

/**
 * @brief Type trait to check if a type is a std::array of NP::REAL.
 *
 * Primary template: yields std::false_type for all other types.
 *
 * @tparam T Type to check.
 */
template <typename T> struct is_std_array_of_real : std::false_type {};

/**
 * @brief Partial specialization for std::array<NP::REAL, N>.
 *
 * Yields std::true_type when std::array of NP::REAL values is inferred
 * implicitly.
 *
 * @tparam N Number of elements in the array.
 */
template <std::size_t N>
struct is_std_array_of_real<std::array<NP::REAL, N>> : std::true_type {};

/**
 * @brief Helper variable template for is_std_array_of_real.
 *
 * @tparam T Type to check.
 */
template <typename T>
inline constexpr bool is_std_array_of_real_v = is_std_array_of_real<T>::value;

/**
 * @brief Iterate over all grid points in row-major order (dimension 0 varies
 * fastest) and execute a function at the calculated coordinates of each
 * grid point.
 *
 * @tparam ndim Number of dimensions.
 * @tparam FUNC Type of callable taking const std::array<NP::REAL, ndim> &
 * (coordinate values).
 * @param coords Per-dimension coordinate vectors defining the grid.
 * @param func Callable invoked once per grid point with the coordinate array.
 */
template <int ndim, typename FUNC>
inline void
iterate_points(const std::array<std::vector<NP::REAL>, ndim> &coords,
               const FUNC &func) {
  std::array<size_t, ndim> num_points;
  for (int i = 0; i < ndim; i++) {
    num_points[i] = coords[i].size();
  }

  size_t total = 1;
  for (auto i : num_points) {
    total *= i;
  }

  std::array<size_t, ndim> idx;
  idx.fill(0);
  for (size_t flat_idx = 0; flat_idx < total; flat_idx++) {
    std::array<NP::REAL, ndim> point_coords;
    for (int i = 0; i < ndim; i++) {
      point_coords[i] = coords[i][idx[i]];
    }
    func(point_coords);

    // Counter incrementing and resetting:
    // When idx[i] overflows num_points[i] it resets to 0 and carries
    // to the next dimension, preserving row-major order without division.
    for (int i = 0; i < ndim; i++) {
      if (++idx[i] < num_points[i])
        break;
      idx[i] = 0;
    }
  }
}

/**
 * @brief Append elements from a container into the flat grid buffer at the
 * current offset.
 *
 * @tparam Container Valid types: std::array<NP::REAL, N>,
 * std::vector<NP::REAL>.
 * @param ptr Pointer to the first value of the data that is to be copied to.
 * @param offset Location to specify where in ptr to copy data to. This is
 * updated post-copy so subsequent calls have the right offset.
 * @param data Container of NP::REAL values to copy.
 */
template <typename Container>
inline void append(NP::REAL *ptr, size_t &offset, const Container &data) {
  // Changed from std::enable_if_t for a nicer/more useful error message.
  static_assert(
      std::is_same_v<Container, std::vector<NP::REAL>> ||
          grid_utils::is_std_array_of_real_v<Container>,
      "If passing a container to append, it must be a "
      "std::vector<NP::REAL> or a std::array<NP::REAL, N>. Alternatively "
      "pass a pointer to the start of a container and the size of "
      "the data to append.");
  std::copy(data.begin(), data.end(), ptr + offset);
  offset += data.size();
}

/**
 * @brief Append n NP::REAL values from a raw pointer into the flat grid buffer
 * at the current offset.
 *
 * @param ptr Pointer to the first value of the data that is to be copied to.
 * @param offset Location to specify where in ptr to copy data to. This is
 * updated post-copy so subsequent calls have the right offset.
 * @param data Pointer to the first NP::REAL value to copy.
 * @param n Number of elements to copy. MUST be less than or equal to the size
 * of data.
 */
void append(NP::REAL *ptr, size_t &offset, const NP::REAL *data, size_t n);
} // namespace grid_utils

/**
 * @brief Struct for describing the underlying grid that will be used by either
 * CartesianGridData or TrimEvalData. It generates the grid at construction and
 * it also handles flattening of data relating to interpolation coordinates,
 * interpolation dimensions, trim dimensions, and the nested per-point tables
 * (depending on the value of output_ndim).
 *
 * When output_ndim = 0, the struct operates in "CartesianGridData" mode where
 * the func that is passed to the constructor is expected to provide a single
 * NP::REAL value and the grid that is calculated has a single value at each
 * grid point.
 *
 * When output_ndim = 3, the struct operates in "TrimEvalData" mode where the
 * func that is passed is expected to provide a std::array<NP::REAL, (trim_dim0
 * + (trim_dim0 * trim_dim1) + (trim_dim0 * trim_dim1 * trim_dim2)> where
 * trim_dim0, trim_dim1, trim_dim2 are the 3 trim dimensions (eg. 5, 5, 5 for
 * EIRENE TRIM data so the size of the function result array would be 155) as an
 * output and subsequently the grid that is calculated will have multiple values
 * per grid point.
 *
 * @tparam interp_ndim Number of interpolation dimensions.
 * @tparam output_ndim Number of output dimensions (default is 0).
 */
template <int interp_ndim, int output_ndim = 0> struct GridDescriptor {
  /**
   * @brief Construct from interpolation coordinates and a
   * generator function (with optional additional context).
   *
   * The generator function is called once per interpolation point in row-major
   * order. It must return a NP::REAL value. Each value is appended to the
   * internal flat grid buffer.
   *
   * @tparam FUNC Generator callable type.
   * @tparam Context Type names of any additional context data needed for the
   * generator.
   * @param coords_in Per-dimension interpolation coordinate vectors.
   * @param func Generator callable with signature
   *   auto(const std::array<NP::REAL, interp_ndim> &coords, Context... context)
   * returning a NP::REAL value.
   */
  // The extra std::is_same_v condition is due to Clang limitation not allowing
  // deactivation using enable_if_t when the condition only uses the template
  // parameter from the enclosing struct.
  template <typename FUNC, typename... Context,
            std::enable_if_t<(output_ndim == 0) && std::is_same_v<FUNC, FUNC>,
                             int> = 0>
  GridDescriptor(
      const std::array<std::vector<NP::REAL>, interp_ndim> &coords_in,
      const FUNC &func, const Context &...context)
      : coords(coords_in) {
    using FUNC_RETURN_TYPE =
        std::invoke_result_t<decltype(func), std::array<NP::REAL, interp_ndim>,
                             decltype(context)...>;
    static_assert(std::is_same_v<FUNC_RETURN_TYPE, NP::REAL>,
                  "Return type of func must be NP::REAL.");

    // Compute total number of interpolation points and allocate the grid vector
    size_t num_points = 1;
    for (const auto &r : this->coords) {
      num_points *= r.size();
    }
    this->grid.assign(num_points, 0.0);

    // Iterate over all interpolation points, evaluate the passed function, and
    // append the func results to the flat grid buffer.
    size_t point_idx = 0;
    grid_utils::iterate_points<interp_ndim>(
        this->coords, [&](const std::array<NP::REAL, interp_ndim> &coords) {
          size_t offset = 0;
          grid_utils::append(&(this->grid[point_idx]), offset,
                             std::array<NP::REAL, 1>{func(coords, context...)});
          ++point_idx;
        });

    flatten_coords();
    flatten_interp_dims();
  }

  /**
   * @brief Construct from interpolation coordinates, trim dimensions, and a
   * generator function (with optional additional context).
   *
   * The generator function is called once per interpolation point in row-major
   * order. It must return a NP::REAL std::array/std::vector of size (trim_dim0
   * + (trim_dim0 * trim_dim1) + (trim_dim0 * trim_dim1 * trim_dim2)) . Each
   * array/vector is appended to the flat grid buffer.
   *
   * @tparam FUNC Generator callable type.
   * @tparam Context Type names of any additional context data needed for the
   * generator.
   * @param coords Per-dimension interpolation coordinate vectors.
   * @param trim_dims_arr TRIM grid dimensions per output axis.
   * @param func Generator callable with signature
   *   auto(const std::array<NP::REAL, interp_ndim> &coords, Context... context)
   * returning a NP::REAL std::array
   */
  // The extra std::is_same_v condition is due to Clang limitation not allowing
  // deactivation using enable_if_t when the condition only uses the template
  // parameter from the enclosing struct.
  template <typename FUNC, typename... Context,
            std::enable_if_t<(output_ndim == 3) && std::is_same_v<FUNC, FUNC>,
                             int> = 0>
  GridDescriptor(const std::array<std::vector<NP::REAL>, interp_ndim> &coords,
                 const std::array<size_t, output_ndim> &trim_dims_arr,
                 const FUNC &func, const Context &...context)
      : coords(coords), output_dims(trim_dims_arr) {
    using FUNC_RETURN_TYPE =
        std::invoke_result_t<decltype(func), std::array<NP::REAL, interp_ndim>,
                             decltype(context)...>;
    static_assert(grid_utils::is_std_array_of_real_v<FUNC_RETURN_TYPE>,
                  "Return type of func must be std::array<NP::REAL>.");

    // Compute grid_stride exactly as TrimEval does
    int agg = 1;
    int grid_stride = 0;
    for (auto &d : this->output_dims) {
      agg *= static_cast<int>(d);
      grid_stride += agg;
    }

    // Compute total number of grid points and allocate the grid vector
    size_t num_points = 1;
    for (const auto &r : this->coords) {
      num_points *= r.size();
    }
    num_points *= grid_stride;
    this->grid.assign(num_points, 0.0);

    // Iterate over all interpolation points, evaluate the passed function, and
    // append the nested per-point tables to the flat grid buffer.
    size_t point_idx = 0;
    size_t grid_access_index = 0;

    grid_utils::iterate_points<interp_ndim>(
        this->coords, [&](const std::array<NP::REAL, interp_ndim> &coords) {
          grid_access_index = point_idx * grid_stride;
          size_t offset = 0;
          grid_utils::append(&(this->grid[grid_access_index]), offset,
                             func(coords, context...));
          NESOASSERT(
              offset == grid_stride,
              "GridGenerator: per-point data size does not match grid_stride.");

          ++point_idx;
        });

    flatten_coords();
    flatten_interp_dims();
    flatten_output_dims();
  }

  /**
   * @brief Return the flattened vector of all of the coordinates for each
   * dimension of the grid.
   */
  const std::vector<NP::REAL> &get_flat_coords() const {
    return this->flat_coords;
  }

  /**
   * @brief Return the vector containing the sizes of each dimension of the
   * grid.
   */
  const std::vector<size_t> &get_interp_dims() const {
    return this->interp_dims_vec;
  }

  /**
   * @brief Return the vector containing the per-interpolation point output
   * dimensions. (Disabled if output_ndim != 3)
   */
  const std::vector<size_t> &get_output_dims() const {
    static_assert(
        (output_ndim == 3),
        "This function is only callable when GridDescriptor has been "
        "constructed with output_ndim = 3 as the template parameter.");
    return this->output_dims_vec;
  }

  /**
   * @brief Return the flat grid data vector.
   */
  const std::vector<NP::REAL> &get_flat_grid() const { return this->grid; }

private:
  /**
   * @brief Flatten the per-dimension interpolation coordinate vectors into a
   * single contiguous vector.
   */
  void flatten_coords() {
    size_t total = 0;
    for (const auto &r : this->coords) {
      total += r.size();
    }

    for (const auto &r : this->coords) {
      this->flat_coords.insert(this->flat_coords.end(), r.begin(), r.end());
    }
  }

  /**
   * @brief Fills the interp_dims vector.
   */
  void flatten_interp_dims() {
    for (const auto &r : this->coords) {
      this->interp_dims_vec.push_back(r.size());
    }
  }

  /**
   * @brief Fills the output dimensions vector. (Disabled if output_ndim != 3)
   */
  void flatten_output_dims() {
    static_assert(
        (output_ndim == 3),
        "This function is only callable when GridDescriptor has been "
        "constructed with output_ndim = 3 as the template parameter.");
    this->output_dims_vec =
        std::vector<size_t>(this->output_dims.begin(), this->output_dims.end());
  }

  std::array<std::vector<NP::REAL>, interp_ndim>
      coords; //!< Array containing vectors that define coordinates for each
              //!< dimension of the grid.
  std::vector<NP::REAL> flat_coords; //!< Vector containing the contiguous
                                     //!< per-dimension coordinates
  std::vector<size_t>
      interp_dims_vec; //!< Vector containing the size of each interpolation
                       //!< dimension for the grid.
  std::array<size_t, output_ndim>
      output_dims; //!< Array containing the size of each output dimension for
                   //!< tables at each interpolation point (eg. {5, 5, 5} for
                   //!< EIRENE-style TRIM tables).
  std::vector<size_t> output_dims_vec; // Vector version of output_dims.
  std::vector<NP::REAL> grid;          //!< Vector containing the flat grid.
};

} // namespace VANTAGE::Reactions

#endif // REACTIONS_GRID_DESCRIPTOR_H
