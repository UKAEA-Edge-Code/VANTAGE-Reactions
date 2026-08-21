
#include "../include/reactions_lib/reaction_data/fixed_coefficient_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

FixedCoefficientData::FixedCoefficientData(
    REAL rate_coefficient, std::map<int, std::string> properties_map)
    : ReactionDataBase<FixedCoefficientDataOnDevice>(
          Properties<REAL>(required_simple_real_props), properties_map) {

  this->on_device_obj = FixedCoefficientDataOnDevice(rate_coefficient);

  this->index_on_device_object();
}

void FixedCoefficientData::index_on_device_object() {

  this->on_device_obj->weight_ind = this->required_real_props.find_index(
      this->properties_map.at(props.weight));
}

} // namespace VANTAGE::Reactions
