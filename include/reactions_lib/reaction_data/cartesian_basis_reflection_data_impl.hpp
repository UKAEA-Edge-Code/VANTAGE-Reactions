#ifndef VANTAGE_REACTIONS_CARTESIAN_BASIS_REFLECTION_DATA_IMPL_H
#define VANTAGE_REACTIONS_CARTESIAN_BASIS_REFLECTION_DATA_IMPL_H

#include "cartesian_basis_reflection_data.hpp"

namespace VANTAGE::Reactions {

CartesianBasisReflectionData::CartesianBasisReflectionData(
    std::map<int, std::string> properties_map)
    : ReactionDataBase<CartesianBasisReflectionDataOnDevice, 3,
                       DEFAULT_RNG_KERNEL, 3>(
          Properties<REAL>(required_simple_real_props), properties_map) {

  this->on_device_obj = CartesianBasisReflectionDataOnDevice();
  this->index_on_device_object();
}

void CartesianBasisReflectionData::index_on_device_object() {

  this->on_device_obj->normal_ind = this->required_real_props.find_index(
      this->properties_map.at(props.boundary_intersection_normal));

  this->on_device_obj->vel_ind = this->required_real_props.find_index(
      this->properties_map.at(props.velocity));
}

} // namespace VANTAGE::Reactions

#endif
