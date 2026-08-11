#ifndef REACTIONS_PROFILING_BASE_H
#define REACTIONS_PROFILING_BASE_H

#include "reactions/neso_particles_namespace_alias.hpp"
#include <optional>
#include <string>
#include <typeinfo>

namespace VANTAGE::Reactions {

constexpr int PROFILING_LEVEL = 1024;

/**
 * Mix-in class to provide profiling to downstream classes.
 */
struct ProfilingBase {
  ProfilingBase() = default;
  virtual ~ProfilingBase() = default;

  /**
   * @returns A name of the class that is being profiled. Override for a better
   * name.
   */
  virtual std::string get_profiling_name();

  /**
   * Start a region to be profiled. The object returned from this call should be
   * passed to `end_profiling_region`.
   *
   * @param subgroup NP::ParticleSubGroup to extract NP::SYCLTarget from.
   * @param key1 Name of region that is being profiled.
   * @returns Region object to pass to `end_profiling_region`.
   */
  [[nodiscard]] std::optional<NESO::Particles::ProfileRegion>
  start_profiling_region(NESO::Particles::ParticleSubGroupSharedPtr &subgroup,
                         const std::string key1);

  /**
   * End a region to be profiled.
   *
   * @param subgroup NP::ParticleSubGroup to extract NP::SYCLTarget from.
   * @param region Region that is being profiled.
   */
  void
  end_profiling_region(NESO::Particles::ParticleSubGroupSharedPtr &subgroup,
                       std::optional<NESO::Particles::ProfileRegion> &region);
};
} // namespace VANTAGE::Reactions

#endif
