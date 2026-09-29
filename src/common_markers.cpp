
#include "../include/reactions_lib/common_markers.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

MinimumNPartInCellMarker::MinimumNPartInCellMarker(INT min_npart)
    : min_npart(min_npart) {}

NP::ParticleSubGroupSharedPtr MinimumNPartInCellMarker::make_marker_subgroup_v(
    NP::ParticleSubGroupSharedPtr particle_group) {

  auto min_npart = this->min_npart;
  auto marker_subgroup = std::make_shared<NP::ParticleSubGroup>(
      particle_group,
      [=](auto cell_info_npart) { return cell_info_npart.get() >= min_npart; },
      NP::Access::read(NP::CellInfoNPart{}));
  return marker_subgroup;
}

PanickedParticleMarker::PanickedParticleMarker(
    const std::map<int, std::string> &properties_map) {
  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

  this->panic_sym = NP::Sym<INT>(properties_map.at(default_properties.panic));
}

NP::ParticleSubGroupSharedPtr PanickedParticleMarker::make_marker_subgroup_v(
    NP::ParticleSubGroupSharedPtr particle_group) {

  auto marker_subgroup = std::make_shared<NP::ParticleSubGroup>(
      particle_group, [=](auto panic) { return panic[0] > 0; },
      NP::Access::read(this->panic_sym));
  return marker_subgroup;
}

bool panicked(NP::ParticleSubGroupSharedPtr particle_group,
              const std::map<int, std::string> &properties_map) {

  auto marker = PanickedParticleMarker(properties_map);

  return marker.make_marker_subgroup(particle_group)->get_npart_local();
}

} // namespace VANTAGE::Reactions
