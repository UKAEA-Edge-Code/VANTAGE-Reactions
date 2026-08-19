
#include "../include/reactions_lib/reaction_data/spherical_basis_reflection_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

SphericalBasisReflectionData::SphericalBasisReflectionData(
    std::map<int, std::string> properties_map)
    : ReactionDataBase<SphericalBasisReflectionDataOnDevice, 3,
                       DEFAULT_RNG_KERNEL, 3>(
          Properties<REAL>(required_simple_real_props), properties_map) {

  this->on_device_obj = SphericalBasisReflectionDataOnDevice();
  this->index_on_device_object();
}

void SphericalBasisReflectionData::index_on_device_object() {

  this->on_device_obj->normal_ind = this->required_real_props.find_index(
      this->properties_map.at(props.boundary_intersection_normal));

  this->on_device_obj->vel_ind = this->required_real_props.find_index(
      this->properties_map.at(props.velocity));
}

} // namespace VANTAGE::Reactions
