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

template <int n_range_dim>
inline std::array<int, 2> calc_closest_point_indices(
    const REAL &x_interp,
    const std::array<REAL, n_range_dim> &dim_range) {

  std::array<int, 2> closest_indices{0, 0};
  std::array<int, 2> eq_indices{0, 0};

  std::array<REAL, n_range_dim> step_result;

  bool x_interp_is_exact = false;
  bool x_interp_is_sub_range = false;
  bool x_interp_is_sup_range = false;

  int sub_count = 0;
  int sup_count = 0;

  for (int i = 0; i < n_range_dim; i++) {
    if (sycl::fabs(x_interp - dim_range[i]) < 1e-12) {
      eq_indices = {i, i};
      x_interp_is_exact = true;
    }
    step_result[i] = sycl::step(dim_range[i], x_interp);
    if (step_result[i] == 0.0) {
      sub_count += 1;
    } else if (step_result[i] == 1.0) {
      sup_count += 1;
    }
  }

  x_interp_is_sub_range = sub_count == n_range_dim ? true : false;
  x_interp_is_sup_range = sup_count == n_range_dim ? true : false;

  for (int i = 1; i < n_range_dim; i++) {
    if (step_result[i] != step_result[i - 1]) {
      closest_indices[0] = i - 1;
      closest_indices[1] = i;
      // break;
    }
  }

  std::array<int, 2> result =
      x_interp_is_sub_range
          ? std::array<int, 2>{0, 0}
          : (x_interp_is_sup_range
                 ? std::array<int, 2>{n_range_dim - 1, n_range_dim - 1}
                 : (x_interp_is_exact ? eq_indices : closest_indices));

  return result;
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
