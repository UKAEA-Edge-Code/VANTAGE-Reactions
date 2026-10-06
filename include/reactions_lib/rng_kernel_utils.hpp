#ifndef REACTIONS_RNG_KERNEL_UTILS_H
#define REACTIONS_RNG_KERNEL_UTILS_H

#include "reactions/neso_particles_namespace_alias.hpp"

#include <memory>
#include <string>
#include <type_traits>

namespace VANTAGE::Reactions::rng_kernel_utils {

/**
 * @brief Check that a random number generator kernel provides at least the
 * required number of components.
 *
 * @tparam RNG_TYPE Random number generator kernel type.
 * @param rng_kernel Shared pointer to the RNG kernel.
 * @param case_name Name of the use case being checked. This is included in the
 * assertion message to aid debugging.
 * @param required_components Minimum number of RNG components required.
 */
template <typename RNG_TYPE>
inline void check_minimum_component_count(std::shared_ptr<RNG_TYPE> rng_kernel,
                                          const std::string &case_name,
                                          const int &required_components) {

  NESOASSERT(rng_kernel->num_components >= required_components,
             "RNG kernel does not provide enough components for " + case_name +
                 ". Required: " + std::to_string(required_components) +
                 ", provided: " + std::to_string(rng_kernel->num_components));
}

/**
 * @brief Enforce that RNG_TYPE is capable of sampling random numbers.
 *
 * This helper provides a compile-time check that the supplied RNG type is not
 * the default NullKernelRNG placeholder. It can be called from constructors or
 * helper functions associated with reaction data types that require sampling.
 *
 * This is not enforced in ReactionDataBase because some derived classes do not
 * require sampling and therefore legitimately use NullKernelRNG as a
 * placeholder type.
 *
 * @tparam RNG_TYPE Random number generator kernel type.
 */
template <typename RNG_TYPE> static constexpr void require_sampling_rng() {
  static_assert(!std::is_same_v<RNG_TYPE, NP::NullKernelRNG<REAL>>,
                "This reaction data requires a sampling RNG kernel. "
                "NullKernelRNG is only a placeholder.");
}

} // namespace VANTAGE::Reactions::rng_kernel_utils
#endif