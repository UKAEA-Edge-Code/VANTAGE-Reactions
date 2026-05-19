#ifndef REACTIONS_GRID_DESCRIPTOR_H
#define REACTIONS_GRID_DESCRIPTOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <neso_particles.hpp>
#include <tuple>
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
 * @tparam FUNC Callable taking const std::array<REAL, ndim> & (coordinate
 * values).
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

/**
 * @brief Invoke each generator function from a tuple with unpacked
 * coordinates and context arguments, appending results to a writer.
 *
 * Used by TrimGridGenerator to evaluate per-point generators and store the
 * nested table data. The trailing std::index_sequence arguments are
 * compile-time tags that drive the parameter-pack expansion over the
 * generator functions and their context arguments.
 *
 * @tparam ContextOffset Offset in the tuple where context arguments begin.
 * @tparam Writer Type supporting an append() method.
 * @tparam N Number of coordinate dimensions.
 * @tparam Tuple Tuple-like type holding generator functions and context args.
 * @tparam FunctionIndices Index sequence selecting generator functions.
 * @tparam ContextIndices Index sequence selecting context arguments.
 * @param writer Output writer (e.g. TrimGridGenerator::TableWriter) that
 *   accumulates the flattened per-point table data via append().
 * @param coords Coordinate array unpacked into each generator call.
 * @param args Tuple containing generator functions and context arguments.
 */
template <size_t ContextOffset, typename Writer, size_t N, typename Tuple,
          size_t... FunctionIndices, size_t... ContextIndices>
inline void
append_func_results(Writer &writer, const std::array<REAL, N> &coords,
                    const Tuple &args, std::index_sequence<FunctionIndices...>,
                    std::index_sequence<ContextIndices...>) {
  (writer.append(
       apply_coords(std::get<FunctionIndices>(args), coords,
                    std::get<ContextOffset + ContextIndices>(args)...)),
   ...);
}

} // namespace grid_detail

/**
 * @brief Generator for Cartesian grid data that handles flattening of
 * dimensions, ranges and values.
 *
 * @tparam ndim Number of grid dimensions.
 */
template <int ndim> struct CartesianGridGenerator {
  /**
   * @brief Construct from per-dimension ranges and a generator function.
   *
   * The generator is called once per grid point in row-major order (dimension 0
   * varying fastest).
   *
   * @tparam FUNC Generator callable type.
   * @param ranges_in Per-dimension range vectors.
   * @param func Generator callable with signature
   *   REAL(const std::array<REAL, ndim>&).
   */
  template <typename FUNC, typename... Context>
  CartesianGridGenerator(const std::array<std::vector<REAL>, ndim> &ranges_in,
                         const FUNC &func, const Context &...context)
      : ranges(std::move(ranges_in)) {

    // Compute total number of grid points and reserve storage.
    size_t expected = 1;
    for (const auto &r : ranges) {
      expected *= r.size();
    }

    // Evaluate the generator at every grid point and store the results.
    grid_detail::iterate_points<ndim>(
        this->ranges, [&](const std::array<REAL, ndim> &coords) {
          this->grid.push_back(
              grid_detail::apply_coords(func, coords, context...));
        });

    // Difficult to test this as there is no external action from the user that
    // can corrupt the calculation of either expected or grid.size(). It is
    // still a runtime sanity check, in case of weird non-user triggered
    // scenarios (maybe bad memory access of this->grid inside func if that's
    // possible?) but might be worth removing if it seems redundant.
    NESOASSERT(this->grid.size() == expected,
               "CartesianGridGenerator: grid size does not match product of "
               "range sizes.");
  }

  /**
   * @brief Flatten the per-dimension range vectors into a single contiguous
   * vector.
   *
   * @return Vector containing all range values concatenated in dimension order.
   */
  const std::vector<REAL> flatten_ranges() const {
    std::vector<REAL> flat;
    size_t total = 0;
    for (const auto &r : this->ranges) {
      total += r.size();
    }

    for (const auto &r : this->ranges) {
      flat.insert(flat.end(), r.begin(), r.end());
    }
    return flat;
  }

  /**
   * @brief Return the size of each dimension's range vector.
   *
   * @return Vector containing the number of grid points per dimension.
   */
  const std::vector<size_t> flatten_dims() const {
    std::vector<size_t> dims;
    for (const auto &r : this->ranges) {
      dims.push_back(r.size());
    }
    return dims;
  }

  /**
   * @brief Return the flat grid data vector.
   *
   * @return Const reference to the internally stored grid values.
   */
  const std::vector<REAL> &flatten_grid() const { return this->grid; }

private:
  std::array<std::vector<REAL>, ndim> ranges;
  std::vector<REAL> grid;
};

/**
 * @brief Generator for TrimEval grid data that handles flattening of
 * interpolation ranges, trim dimensions, and the nested per-point tables.
 *
 * @tparam interp_ndim Number of interpolation dimensions.
 * @tparam output_ndim Number of TRIM output dimensions.
 */
template <int interp_ndim, int output_ndim> struct TrimGridGenerator {
  std::array<std::vector<REAL>, interp_ndim> ranges;
  std::array<size_t, output_ndim> trim_dims;
  std::vector<REAL> grid;
  int grid_stride = 0;

  /**
   * @brief Helper for appending nested tables into the flat grid buffer.
   *
   * Each interpolation point in the TrimGridGenerator owns a contiguous slice
   * of the flat grid. A TableWriter is passed to the user's generator
   * function so that nested per-output-dimension tables can be appended in
   * the order expected by TrimEval.
   */
  struct TableWriter {
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
      std::copy(data.begin(), data.end(), ptr + offset);
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
      std::copy(data, data + n, ptr + offset);
      offset += static_cast<int>(n);
    }
  };

  /**
   * @brief Construct from interpolation ranges, trim dimensions, and a
   * generator function.
   *
   * The generator is called once per interpolation point in row-major order.
   * It must return a tuple-like object (e.g. std::tuple, std::array) of
   * exactly output_ndim containers. Each container is appended in order to
   * the internal flat grid buffer.
   *
   * @tparam FUNC Generator callable type.
   * @param ranges_in Per-dimension interpolation range vectors.
   * @param trim_dims_in TRIM grid dimensions per output axis.
   * @param func Generator callable with signature
   *   auto(const std::array<REAL, interp_ndim>&) returning a tuple-like
   *   object of output_ndim containers.
   */
  template <typename... Args>
  TrimGridGenerator(const std::array<std::vector<REAL>, interp_ndim> &ranges_in,
                    const std::array<size_t, output_ndim> &trim_dims_in,
                    const Args &...args)
      : ranges(std::move(ranges_in)), trim_dims(trim_dims_in) {

    static_assert(sizeof...(Args) >= output_ndim,
                  "TrimGridGenerator: must provide at least output_ndim "
                  "functions");

    // Compute grid_stride exactly as TrimEval does
    int agg = 1;
    for (auto d : trim_dims) {
      agg *= static_cast<int>(d);
      grid_stride += agg;
    }

    // Compute total number of interpolation points and allocate the grid vector
    size_t num_points = 1;
    for (const auto &r : ranges) {
      num_points *= r.size();
    }
    for (size_t i = 0; i < (num_points * grid_stride); i++) {
      grid.push_back(0.0);
    }

    // Pack generators and trailing context arguments into a tuple for
    // indexed access during iteration.
    constexpr size_t context_count = sizeof...(Args) - output_ndim;
    auto args_tuple = std::make_tuple(args...);

    // Iterate over all interpolation points, evaluate the generators, and
    // append the nested per-point tables into the flat grid buffer.
    size_t point_idx = 0;
    grid_detail::iterate_points<interp_ndim>(
        ranges, [&](const std::array<REAL, interp_ndim> &coords) {
          TableWriter writer{&grid[point_idx * grid_stride]};
          grid_detail::append_func_results<output_ndim>(
              writer, coords, args_tuple,
              std::make_index_sequence<output_ndim>{},
              std::make_index_sequence<context_count>{});
          NESOASSERT(writer.offset == grid_stride,
                     "TrimGridGenerator: per-point data size does not match "
                     "grid_stride.");
          ++point_idx;
        });
  }

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
   * @brief Return the TRIM grid dimensions.
   *
   * @return Vector containing the TRIM output dimensions.
   */
  const std::vector<size_t> flatten_trim_dims() const {
    return std::vector<size_t>(trim_dims.begin(), trim_dims.end());
  }

  /**
   * @brief Return the per-interpolation-point data stride.
   *
   * @return The total number of REAL values stored for each interpolation
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

} // namespace VANTAGE::Reactions

#endif // REACTIONS_GRID_DESCRIPTOR_H
