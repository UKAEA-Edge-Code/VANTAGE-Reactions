#pragma once
#include <cassert>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace Reactions::utils {
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
 * @brief Helper function to build a std::vector of Syms.
 *
 * @tparam PROP_TYPE The property type associated with the syms that need to
 * be stored in the resulting vector (either INT or REAL).
 *
 * @param particle_spec ParticleSpec that contains properties that are matched
 * with required_properties to fill the vector of Syms.
 * @param required_properties A vector of strings that contains the required
 * properties that need to exist in particle_spec to fill the vector of
 * corresponding Syms.
 *
 * @return A std::vector of Syms of PROP_TYPE (ie.
 * std::vector<Syms<PROP_TYPE>>)
 */
template <typename PROP_TYPE>
std::vector<Sym<PROP_TYPE>>
build_sym_vector(ParticleSpec particle_spec,
                 std::vector<std::string> required_properties) {

  std::vector<Sym<PROP_TYPE>> syms = {};

  for (auto req_prop : required_properties) {
    if constexpr (std::is_same_v<PROP_TYPE, INT>) {
      for (auto &int_prop : particle_spec.properties_int) {
        if (int_prop.name == req_prop) {
          syms.push_back(Sym<INT>(int_prop.name));
        }
      }
    }
    if constexpr (std::is_same_v<PROP_TYPE, REAL>) {
      for (auto &real_prop : particle_spec.properties_real) {
        if (real_prop.name == req_prop) {
          syms.push_back(Sym<REAL>(real_prop.name));
        }
      }
    }
  }

  return syms;
}

} // namespace Reactions::utils
