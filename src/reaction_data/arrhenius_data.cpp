#include "../include/reactions_lib/reaction_data/arrhenius_data.hpp"

namespace VANTAGE::Reactions {

ArrheniusData::ArrheniusData(REAL a_coeff, REAL b_coeff,
                             std::map<int, std::string> properties_map)
    : ReactionDataBase<ArrheniusDataOnDevice>(
          Properties<REAL>(required_simple_real_props), properties_map) {

  this->on_device_obj = ArrheniusDataOnDevice(a_coeff, b_coeff);

  this->index_on_device_object();
}

void ArrheniusData::index_on_device_object() {

  this->on_device_obj->weight_ind = this->required_real_props.find_index(
      this->properties_map.at(props.weight));

  this->on_device_obj->temperature_ind = this->required_real_props.find_index(
      this->properties_map.at(props.fluid_temperature));
}

} // namespace VANTAGE::Reactions
