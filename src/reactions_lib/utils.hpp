#ifndef REACTIONS_UTILS_H
#define REACTIONS_UTILS_H
#include <cassert>
#include <cmath>
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
inline std::array<REAL, 2> box_muller_transform(const REAL &u1,
                                                const REAL &u2) {
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
reflect_vector(const std::array<REAL, n_dim> &input,
               const std::array<REAL, n_dim> &ref_vector) {

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

/**
 * @brief Return dot(input,proj_direction) * proj_direction. If proj_direction
 * is a unit vector this will be a projection of input onto proj_direction.
 *
 * @param input The input vector
 * @param proj_direction Direction onto which to project the input
 */
template <size_t n_dim>
inline std::array<REAL, n_dim>
project_vector(const std::array<REAL, n_dim> &input,
               const std::array<REAL, n_dim> &proj_direction) {

  REAL proj_factor = 0.0;
  std::array<REAL, n_dim> output;

  for (int dim = 0; dim < n_dim; dim++) {

    proj_factor += input[dim] * proj_direction[dim];
  }

  for (int dim = 0; dim < n_dim; dim++) {

    output[dim] = proj_factor * proj_direction[dim];
  }

  return output;
};

} // namespace VANTAGE::Reactions::utils
#endif
