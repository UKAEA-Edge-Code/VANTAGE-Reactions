#ifndef REACTIONS_GRID_GENERATORS_H
#define REACTIONS_GRID_GENERATORS_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <neso_particles.hpp>
#include <type_traits>
#include <utility>
#include <vector>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

namespace grid_detail {

/**
 * @brief Type trait to check if a type is a std::array of REAL.
 *
 * Primary template: yields std::false_type for all other types.
 *
 * @tparam T Type to check.
 */
template <typename T> struct is_std_array_of_real : std::false_type {};

/**
 * @brief Partial specialization for std::array<REAL, N>.
 *
 * Yields std::true_type when T is a std::array of REAL values.
 *
 * @tparam N Number of elements in the array.
 */
template <std::size_t N>
struct is_std_array_of_real<std::array<REAL, N>> : std::true_type {};

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
 * @tparam FUNC Type of callable taking const std::array<REAL, ndim> &
 * (coordinate values).
 * @param ranges Per-dimension range vectors defining the grid coordinates.
 * @param func Callable invoked once per grid point with the coordinate array.
 */
template <int ndim, typename FUNC>
inline void iterate_points(const std::array<std::vector<REAL>, ndim> &ranges,
                           const FUNC &func) {
  std::array<size_t, ndim> num_points;
  for (int i = 0; i < ndim; i++) {
    num_points[i] = ranges[i].size();
  }

  size_t total = 1;
  for (auto i : num_points) {
    total *= i;
  }

  std::array<size_t, ndim> idx{};
  for (size_t flat_idx = 0; flat_idx < total; flat_idx++) {
    std::array<REAL, ndim> coords;
    for (int i = 0; i < ndim; i++) {
      coords[i] = ranges[i][idx[i]];
    }
    func(coords);

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
 * @brief Implementation detail for apply_coords.
 *
 * Unpacks a std::array as individual arguments to a callable using an index
 * sequence.
 *
 * @tparam FUNC Callable type.
 * @tparam T Array element type.
 * @tparam N Array size.
 * @tparam Is Index sequence (0, 1, ..., N-1).
 * @tparam Context Extra argument types forwarded after the array elements.
 * @param func Callable to invoke.
 * @param arr Array whose elements are unpacked as arguments.
 * @param context Extra arguments forwarded after the unpacked elements.
 * @return decltype(auto) Result of invoking func(arr[0], ..., arr[N-1],
 * context...).
 */
template <typename FUNC, typename T, size_t N, size_t... Is,
          typename... Context>
inline decltype(auto)
apply_coords_impl(const FUNC &func, const std::array<T, N> &arr,
                  std::index_sequence<Is...>, const Context &...context) {
  return func(arr[Is]..., context...);
}

/**
 * @brief Unpack a std::array as individual arguments to a callable.
 *
 * Optionally appends extra context arguments after the unpacked elements.
 *
 * @tparam FUNC Callable type.
 * @tparam T Array element type.
 * @tparam N Array size.
 * @tparam Context Extra argument types forwarded after the array elements.
 * @param func Callable to invoke.
 * @param arr Array whose elements are unpacked as arguments.
 * @param context Extra arguments forwarded after the unpacked elements.
 * @return decltype(auto) Result of invoking func(arr[0], ..., arr[N-1],
 * context...).
 */
template <typename FUNC, typename T, size_t N, typename... Context>
inline decltype(auto) apply_coords(const FUNC &func,
                                   const std::array<T, N> &arr,
                                   const Context &...context) {
  return apply_coords_impl(func, arr, std::make_index_sequence<N>{},
                           context...);
}

/* @brief Compute the return type of a callable when invoked with unpacked
 * std::array<REAL, N> elements plus trailing context arguments.
 */
template <typename FUNC, size_t N, typename... Context, size_t... Is>
auto invoke_result_unpacked_impl(std::index_sequence<Is...>)
    -> std::invoke_result_t<
        FUNC,
        decltype(std::get<Is>(std::declval<const std::array<REAL, N> &>()))...,
        Context...>;
template <typename FUNC, size_t N, typename... Context>
using invoke_result_unpacked =
    decltype(invoke_result_unpacked_impl<FUNC, N, Context...>(
        std::make_index_sequence<N>{}));

/**
 * @brief Helper for appending nested tables into a flat grid buffer.
 *
 * Usage consists of passing a table (either in a vector/array or as a
 * pointer to and size of an vector/array) to append(...) which appends the
 * table (as a flat buffer) to the buffer that is pointed to by REAL *ptr.
 * (Note the grid buffer pointed to by REAL *ptr, is written to directly by
 * append(...) so avoid concurrent calls to append(...) to prevent potential
 * data-races)
 */
struct TableWriter {
  TableWriter() = default;
  REAL *ptr = nullptr;
  int offset = 0;

  /**
   * @brief Append elements from a container into the flat grid buffer at the
   * current offset.
   *
   * @tparam Container Valid types: std::array<REAL, N>, std::vector<REAL>.
   * @param data Container of REAL values to copy.
   */
  template <typename Container>
  std::enable_if_t<std::is_same_v<Container, std::vector<REAL>> ||
                       grid_detail::is_std_array_of_real_v<Container>,
                   void>
  append(const Container &data) {
    std::copy(data.begin(), data.end(), this->ptr + this->offset);
    offset += static_cast<int>(data.size());
  }

  /**
   * @brief Append n REAL values from a raw pointer into the flat grid buffer
   * at the current offset.
   *
   * @param data Pointer to the first REAL value to copy.
   * @param n Number of elements to copy.
   */
  void append(const REAL *data, size_t n) {
    std::copy(data, data + n, this->ptr + this->offset);
    offset += static_cast<int>(n);
  }
};

/**
 * @brief Call a function on a given set of coords and append the result to a
 * flat grid buffer (via TableWriter::append).
 *
 * The result of the function call should be either a REAL value or a
 * std::array<REAL>/std::vector<REAL>.
 *
 * @tparam N Number of coordinate dimensions.
 * @tparam FUNC Callable type.
 * @tparam Context Extra argument types forwarded after the array elements.
 * @param writer Output TableWriter  that accumulates the flattened per-point
 * table data via append().
 * @param coords Coordinate array unpacked into each generator call.
 * @param func Callable to be passed to apply_coords
 * @param context Extra arguments needed for func besides coords.
 */
template <size_t N, typename FUNC, typename... Context>
inline void append_func_results(TableWriter &writer,
                                const std::array<REAL, N> &coords,
                                const FUNC &func, const Context &...context) {
  using FUNC_RETURN_TYPE = invoke_result_unpacked<FUNC, N, Context...>;
  auto func_result = apply_coords(func, coords, context...);

  if constexpr (std::is_same_v<FUNC_RETURN_TYPE, REAL>) {
    writer.append(std::array<REAL, 1>{func_result});
  } else if (grid_detail::is_std_array_of_real_v<FUNC_RETURN_TYPE> ||
             std::is_same_v<FUNC_RETURN_TYPE, std::vector<REAL>>) {
    writer.append(func_result);
  }
}

} // namespace grid_detail

/**
 * @brief Abstract base class with common elements for both versions of
 * GridGenerator.
 *
 * @tparam interp_ndim Number of interpolation dimensions.
 * @tparam output_ndim Number of output dimensions (default is 0).
 *
 */
template <int interp_ndim, int output_ndim = 0> struct AbstractGridGenerator {
  AbstractGridGenerator() = default;

  std::array<std::vector<REAL>, interp_ndim> ranges;
  std::array<size_t, output_ndim> output_dims;
  std::vector<REAL> grid;
  int grid_stride = 0;

  /**
   * @brief Flatten the per-dimension interpolation range vectors into a single
   * contiguous vector.
   *
   * @return Vector containing all range values concatenated in dimension order.
   */
  const std::vector<REAL> flatten_ranges() const {
    std::vector<REAL> flat;
    size_t total = 0;
    for (const auto &r : ranges) {
      total += r.size();
    }

    for (const auto &r : ranges) {
      flat.insert(flat.end(), r.begin(), r.end());
    }
    return flat;
  }

  /**
   * @brief Return the size of each interpolation dimension's range vector.
   *
   * @return Vector containing the number of interpolation grid points per
   * dimension.
   */
  const std::vector<size_t> flatten_interp_dims() const {
    std::vector<size_t> dims;
    for (const auto &r : ranges) {
      dims.push_back(r.size());
    }
    return dims;
  }

  /**
   * @brief Return the output dimensions.
   *
   * @return Vector containing the output dimensions.
   */
  const std::vector<size_t> flatten_output_dims() const {
    return std::vector<size_t>(output_dims.begin(), output_dims.end());
  }

  /**
   * @brief Return the per-interpolation-point data stride.
   *
   * @return The total number of values stored for each interpolation
   * point.
   */
  const int &get_grid_stride() const { return grid_stride; }

  /**
   * @brief Return the flat grid data vector.
   *
   * @return Const reference to the internally stored grid values.
   */
  const std::vector<REAL> &flatten_grid() const { return grid; }
};

/**
 * @brief Generator struct for CartesianGridData or TrimEval grid data that
 * handles flattening of interpolation ranges, trim dimensions, and the nested
 * per-point tables.
 *
 * @tparam interp_ndim Number of interpolation dimensions.
 * @tparam output_ndim Number of output dimensions (default is 0).
 */
template <int interp_ndim, int output_ndim = 0>
struct GridGenerator : AbstractGridGenerator<interp_ndim, output_ndim> {
  /**
   * @brief Construct from interpolation ranges and a
   * generator function (with optional additional context).
   *
   * The generator is called once per interpolation point in row-major order.
   * It must return a REAL value. Each value is appended to the internal flat
   * grid buffer.
   *
   * @tparam FUNC Generator callable type.
   * @tparam Context Type names of any additional context data needed for the
   * generator.
   * @param ranges_in Per-dimension interpolation range vectors.
   * @param func Generator callable with signature
   *   auto(const REAL &dim0_val, const REAL &dim1_val,... , Context... context)
   * returning a REAL value.
   */
  // The extra std::is_same_v condition is due to Clang limitation not allowing
  // deactivation using enable_if_t when the condition only uses the template
  // parameter from the enclosing struct.
  template <typename FUNC, typename... Context,
            std::enable_if_t<(output_ndim == 0) && std::is_same_v<FUNC, FUNC>,
                             int> = 0>
  GridGenerator(const std::array<std::vector<REAL>, interp_ndim> &ranges_in,
                const FUNC &func, const Context &...context)
      : AbstractGridGenerator<interp_ndim, output_ndim>{ranges_in} {

    // Compute total number of interpolation points and allocate the grid vector
    size_t num_points = 1;
    for (const auto &r : this->ranges) {
      num_points *= r.size();
    }
    this->grid.assign(num_points, 0.0);

    // Iterate over all interpolation points, evaluate the passed function, and
    // append the func results to the flat grid buffer.
    size_t point_idx = 0;
    grid_detail::iterate_points<interp_ndim>(
        this->ranges, [&](const std::array<REAL, interp_ndim> &coords) {
          grid_detail::TableWriter writer{&(this->grid[point_idx])};
          grid_detail::append_func_results(writer, coords, func, context...);
          ++point_idx;
        });
  }

  /**
   * @brief Construct from interpolation ranges, trim dimensions, and a
   * generator function (with optional additional context).
   *
   * The generator is called once per interpolation point in row-major order.
   * It must return a REAL std::array/std::vector of size (trim_dim0 +
   * (trim_dim0 * trim_dim1) + (trim_dim0 * trim_dim1 * trim_dim2)) . Each
   * array/vector is appended to the flat grid buffer.
   *
   * @tparam FUNC Generator callable type.
   * @tparam Context Type names of any additional context data needed for the
   * generator.
   * @param ranges_in Per-dimension interpolation range vectors.
   * @param trim_dims_arr TRIM grid dimensions per output axis.
   * @param func Generator callable with signature
   *   auto(const REAL &dim0_val, const REAL &dim1_val,... , Context... context)
   * returning a REAL std::array
   */
  // The extra std::is_same_v condition is due to Clang limitation not allowing
  // deactivation using enable_if_t when the condition only uses the template
  // parameter from the enclosing struct.
  template <typename FUNC, typename... Context,
            std::enable_if_t<(output_ndim == 3) && std::is_same_v<FUNC, FUNC>,
                             int> = 0>
  GridGenerator(const std::array<std::vector<REAL>, interp_ndim> &ranges_in,
                const std::array<size_t, output_ndim> &trim_dims_arr,
                const FUNC &func, const Context &...context)
      : AbstractGridGenerator<interp_ndim, output_ndim>{ranges_in,
                                                        trim_dims_arr} {

    // Compute grid_stride exactly as TrimEval does
    int agg = 1;
    for (auto &d : this->output_dims) {
      agg *= static_cast<int>(d);
      this->grid_stride += agg;
    }

    // Compute total number of grid points and allocate the grid vector
    size_t num_points = 1;
    for (const auto &r : this->ranges) {
      num_points *= r.size();
    }
    num_points *= this->grid_stride;
    this->grid.assign(num_points, 0.0);

    // Iterate over all interpolation points, evaluate the passed function, and
    // append the nested per-point tables to the flat grid buffer.
    size_t point_idx = 0;
    size_t grid_access_index = 0;

    grid_detail::iterate_points<interp_ndim>(
        this->ranges, [&](const std::array<REAL, interp_ndim> &coords) {
          grid_access_index = point_idx * this->grid_stride;
          grid_detail::TableWriter writer{&(this->grid[grid_access_index])};
          grid_detail::append_func_results(writer, coords, func, context...);
          NESOASSERT(
              writer.offset == this->grid_stride,
              "GridGenerator: per-point data size does not match grid_stride.");

          ++point_idx;
        });
  }
};

} // namespace VANTAGE::Reactions

#endif // REACTIONS_GRID_GENERATORS_H
