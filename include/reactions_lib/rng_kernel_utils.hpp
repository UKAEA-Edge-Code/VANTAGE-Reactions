#ifndef REACTIONS_RNG_KERNEL_UTILS_H
#define REACTIONS_RNG_KERNEL_UTILS_H

#include "reactions/neso_particles_namespace_alias.hpp"

namespace rng_kernel_utils {

template <typename RNG_TYPE>
inline void CheckMinimumComponentCount(std::shared_ptr<RNG_TYPE> rng_kernel,
                                       const std::string &case_name,
                                       int required_components) {

  NESOASSERT(rng_kernel->num_components >= required_components,
             "RNG kernel does not provide enough components for " + case_name +
                 ". Required: " + std::to_string(required_components) +
                 ", provided: " + std::to_string(rng_kernel->num_components));
}

} // namespace rng_kernel_utils

#endif