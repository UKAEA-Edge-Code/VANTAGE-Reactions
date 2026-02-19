#ifndef REACTIONS_COMMON_MARKERS_H
#define REACTIONS_COMMON_MARKERS_H
#include "transformation_wrapper.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;

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
  MinimumNPartInCellMarker(INT min_npart) : min_npart(min_npart) {};

  ParticleSubGroupSharedPtr
  make_marker_subgroup_v(ParticleSubGroupSharedPtr particle_group) {

    auto min_npart = this->min_npart;
    auto marker_subgroup = std::make_shared<ParticleSubGroup>(
        particle_group,
        [=](auto cell_info_npart) {
          return cell_info_npart.get() >= min_npart;
        },
        Access::read(CellInfoNPart{}));
    return marker_subgroup;
  };

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
   * used to remap the Sym for the Panic property.
   */
  PanickedParticleMarker(
      const std::map<int, std::string> &properties_map = get_default_map()) {
    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    this->panic_sym = Sym<INT>(properties_map.at(default_properties.panic));
  };

  ParticleSubGroupSharedPtr
  make_marker_subgroup_v(ParticleSubGroupSharedPtr particle_group) {

    auto marker_subgroup = std::make_shared<ParticleSubGroup>(
        particle_group, [=](auto panic) { return panic[0] > 0; },
        Access::read(this->panic_sym));
    return marker_subgroup;
  };

private:
  Sym<INT> panic_sym;
};

inline bool
panicked(ParticleSubGroupSharedPtr particle_group,
         const std::map<int, std::string> &properties_map = get_default_map()) {

  auto marker = PanickedParticleMarker(properties_map);

  return marker.make_marker_subgroup(particle_group)->get_npart_local();
}

} // namespace VANTAGE::Reactions
#endif
