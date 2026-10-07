#ifndef REACTIONS_RNG_KERNEL_COMPONENT_REQUIREMENT_H
#define REACTIONS_RNG_KERNEL_COMPONENT_REQUIREMENT_H

#include "reactions/neso_particles_namespace_alias.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>

namespace VANTAGE::Reactions {

/**
 * @brief Mixin exposing and validating an RNG kernel component requirement.
 *
 * @tparam RNG_TYPE Random number generator kernel type.
 * @tparam RequiredComponents Minimum number of components required.
 */
template <typename RNG_TYPE, std::size_t RequiredComponents>
struct RNGKernelComponentRequirement {
  static_assert(!std::is_same_v<RNG_TYPE, NP::NullKernelRNG<REAL>>,
                "Sampling requires a non-null RNG kernel type.");

  static constexpr std::size_t required_kernel_components = RequiredComponents;

  bool has_required_kernel_components(
      const std::shared_ptr<RNG_TYPE> &rng_kernel) const {
    return rng_kernel &&
           rng_kernel->num_components >= required_kernel_components;
  }

  void validate_required_kernel_components(
      const std::shared_ptr<RNG_TYPE> &rng_kernel) const {
    const std::string available_components =
        rng_kernel ? std::to_string(rng_kernel->num_components) : "null";
    NESOASSERT(has_required_kernel_components(rng_kernel),
               "RNG kernel does not provide enough components. Required: " +
                   std::to_string(required_kernel_components) +
                   ", provided: " + available_components);
  }
};

} // namespace VANTAGE::Reactions
#endif