#ifndef REACTIONS_COMMON_MARKERS_H
#define REACTIONS_COMMON_MARKERS_H
#include "particle_properties_map.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include "transformation_wrapper.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Marking strategy that selects only those particles in cells containing
 * some minimum number of particles
 *
 */
struct MinimumNPartInCellMarker : MarkingStrategy {

public:
  MinimumNPartInCellMarker() = delete;

  /**
   * @brief Constructor for MinimumNPartInCellMarker.
   *
   * @param min_npart Minimum number of particles in a cell.
   */
  MinimumNPartInCellMarker(INT min_npart);

  NP::ParticleSubGroupSharedPtr
  make_marker_subgroup_v(NP::ParticleSubGroupSharedPtr particle_group);

private:
  INT min_npart;
};

/**
 * @brief Marking strategy that selects only those particles with a panic flag >
 * 0
 *
 */
struct PanickedParticleMarker : MarkingStrategy {

public:
  PanickedParticleMarker() = delete;

  /**
   * @brief Constructor for PanickedParticleMarker.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used to remap the NP::Sym for the Panic property.
   */
  PanickedParticleMarker(
      const std::map<int, std::string> &properties_map = get_default_map());

  NP::ParticleSubGroupSharedPtr
  make_marker_subgroup_v(NP::ParticleSubGroupSharedPtr particle_group);

private:
  NP::Sym<INT> panic_sym;
};

bool panicked(
    NP::ParticleSubGroupSharedPtr particle_group,
    const std::map<int, std::string> &properties_map = get_default_map());

} // namespace VANTAGE::Reactions

#endif
