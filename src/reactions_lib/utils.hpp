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
 * @brief Wrapper class to provide default constructible lambdas for templating
 * device types that need them
 */
template <class F, size_t DIM = 1> struct LambdaWrapper {

  static const size_t OUTPUT_DIM = DIM;

  LambdaWrapper() = default;

  explicit LambdaWrapper(F &f) {

    static_assert(
        std::is_trivially_copyable<F>::value,
        "LambdaWrapper template parameter must be trivially copyable");
    static_assert(
        std::is_trivially_destructible<F>::value,
        "LambdaWrapper template parameter must be trivially destructible");
    ::new (static_cast<void *>(this->buf)) F(std::forward<F>(f));
  }

  const F &get() const {
    return *std::launder(reinterpret_cast<const F *>(this->buf));
  }

  template <class... Args>
  auto operator()(Args &...args) const
      -> decltype(std::declval<const F &>()(std::forward<Args>(args)...)) {
    return this->get()(std::forward<Args>(args)...);
  }

private:
  alignas(F) unsigned char buf[sizeof(F)];
};

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

/**
 * @brief Helper function to compute vector cross product of two length 3
 * vectors.
 *
 * @param a first cross product argument
 * @param b second cross product argument
 * @return std::vector<T> a x b
 */
template <typename T>
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

// TODO: refine and test

inline std::array<REAL, 9> get_normal_basis(const std::array<REAL, 3> &vel,
                                            const std::array<REAL, 3> &normal) {

  auto proj = project_vector(vel, normal);

  std::array<REAL, 9> result;

  for (auto i = 0; i < 3; i++) {

    result[i] = vel[i] - proj[i];
  }

  REAL norm = 0;

  for (auto i = 0; i < 3; i++) {

    norm += result[i] * result[i];
  }

  for (auto i = 0; i < 3; i++) {

    result[i] = result[i] / Kernel::sqrt(norm);
  }
  result[3] = normal[1] * result[2] - normal[2] * result[1];
  result[4] = normal[2] * result[0] - normal[0] * result[2];
  result[5] = normal[0] * result[1] - normal[1] * result[0];

  result[6] = normal[0];
  result[7] = normal[1];
  result[8] = normal[2];
  return result;
};

inline std::array<REAL, 3>
normal_basis_to_cartesian(const std::array<REAL, 3> &coords,
                          const std::array<REAL, 9> &basis) {

  REAL costheta;
  REAL theta = coords[1];
  const REAL sintheta = Kernel::sincos(theta, &costheta);

  REAL cosphi;
  REAL phi = coords[2];
  const REAL sinphi = Kernel::sincos(phi, &cosphi);

  std::array<REAL, 3> result;

  for (auto i = 0; i < 3; i++) {

    result[i] = coords[0] *
                (sintheta * cosphi * basis[i] +
                 sintheta * sinphi * basis[i + 3] + costheta * basis[i + 6]);
  }
}

} // namespace VANTAGE::Reactions::utils
#endif
