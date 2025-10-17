#ifndef REACTIONS_UTILS_H
#define REACTIONS_UTILS_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <neso_particles.hpp>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions::utils {
/**
 * @brief Helper function to calculate the L2 norm of a vector of arithmetic
 * types.
 *
 * @tparam T Arithmetic type template parameter
 * @param vec Vector to take norm of
 * @return T sqrt(sum(x^2)) for x in vec
 */
template <typename T> T norm2(const std::vector<T> &vec) {
  static_assert(std::is_arithmetic<T>(),
                "Template type in norm2 must be arithmetic");
  return std::sqrt(std::accumulate(vec.begin(), vec.end(), T(),
                                   [](T a, T b) { return a + b * b; }));
}

template <typename T>
/**
 * @brief Helper function to compute vector cross product of two length 3
 * vectors.
 *
 * @param a first cross product argument
 * @param b second cross product argument
 * @return std::vector<T> a x b
 */
std::vector<T> cross_product(const std::vector<T> &a, const std::vector<T> &b) {
  static_assert(std::is_arithmetic<T>(),
                "Template type in cross_product must be arithmetic");
  if (a.size() != 3 || b.size() != 3) {
    assert("cross_product called with vectors not of size 3");
  }

  std::vector<T> result(3);

  result[0] = a[1] * b[2] - a[2] * b[1];
  result[1] = a[2] * b[0] - a[0] * b[2];
  result[2] = a[0] * b[1] - a[1] * b[0];

  return result;
}

/**
 * @brief Helper function to build a std::vector of Syms from a list of names.
 *
 * @tparam PROP_TYPE The property type associated with the syms that need to
 * be stored in the resulting vector (either INT or REAL).
 *
 * @param required_properties A vector of strings that contains the required
 * properties
 *
 * @return A std::vector of Syms of PROP_TYPE (ie.
 * std::vector<Syms<PROP_TYPE>>)
 */
template <typename PROP_TYPE>
std::vector<Sym<PROP_TYPE>>
build_sym_vector(std::vector<std::string> required_properties) {

  std::vector<Sym<PROP_TYPE>> syms = {};

  for (auto req_prop : required_properties) {
    syms.push_back(Sym<PROP_TYPE>(req_prop));
  }

  return syms;
}

/**
 * @brief Perform the standard deterministic Box-Muller transform and store the
 * two normal variates into an array (to avoid use of tuples/pairs in order to
 * maximise SYCL compatibility)
 *
 * @param u1 First uniformly distributed random number
 * @param u2 Second uniformly distributed random number
 *
 * @return A REAL-valued array of size 2 containing the calculated two normal
 * variates.
 */
inline std::array<REAL, 2> box_muller_transform(REAL u1, REAL u2) {
  constexpr REAL two_pi = 2 * M_PI;

  auto magnitude = Kernel::sqrt(-2 * Kernel::log(u1));
  REAL valuecos;
  const REAL valuesin = Kernel::sincos(two_pi * u2, &valuecos);
  return std::array<REAL, 2>{magnitude * valuecos, magnitude * valuesin};
};

/**
 * @brief Reflect an input array across a normalised reflection vector (e.g.
 * surface normal). output = input - 2 * dot_product(input,ref_vector) *
 * ref_vector
 *
 * @param input Input array to be reflected
 * @param ref_vector Normalised vector to reflect through
 * @return Reflected array
 */
template <size_t n_dim>
inline std::array<REAL, n_dim>
reflect_vector(std::array<REAL, n_dim> input,
               std::array<REAL, n_dim> ref_vector) {

  REAL proj_factor = 0.0;
  std::array<REAL, n_dim> output;

  for (int dim = 0; dim < n_dim; dim++) {

    proj_factor += 2 * input[dim] * ref_vector[dim];
  }

  for (int dim = 0; dim < n_dim; dim++) {

    output[dim] = input[dim] - proj_factor * ref_vector[dim];
  }

  return output;
};

template <std::size_t n_range_dim>
inline std::array<std::size_t, 2>
calc_closest_point_indices(const REAL &x_interp,
                           const std::array<REAL, n_range_dim> &dim_range) {

  REAL step_result_old = sycl::step(dim_range[0], x_interp);
  REAL step_result_new = step_result_old;

  for (std::size_t i = 0; i < n_range_dim; i++) {
    step_result_new = sycl::step(dim_range[i], x_interp);

    if ((step_result_new - step_result_old) != 0.0) {
      return {i - 1, i};
    }

    step_result_old = step_result_new;
  }

  // This return statement is not expected to be hit but is needed to avoid
  // compiler warnings (since it ensures all paths are defined).
  return {0, 0};
};

inline REAL linear_interp(const REAL &x_interp, const REAL &x0, const REAL &x1,
                          const REAL &f0, const REAL &f1) {
  REAL dfdx = (f1 - f0) / (x1 - x0);
  REAL c = f0 - (dfdx * x0);

  REAL f_interp = (dfdx * x_interp) + c;
  return f_interp;
};
} // namespace VANTAGE::Reactions::utils
#endif
