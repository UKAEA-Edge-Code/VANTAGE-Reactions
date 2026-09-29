#include "../include/reactions_lib/particle_properties_map.hpp"

namespace VANTAGE::Reactions {

PropertiesMap::PropertiesMap(std::map<int, std::string> custom_map)
    : private_map(custom_map) {
  // replace default_properties.fluid_flow_speed with the last enum in
  // standard_properties_enum if any changes are made to it.
  for (int i = 0; i < default_properties.fluid_flow_speed; i++) {
    NESOWARN(
        this->private_map.find(i) != this->private_map.end(),
        "The custom properties map provided does not contain all enums from "
        "default_properties in it's list of keys.");
  }
}

std::map<int, std::string> PropertiesMap::get_map() {
  return this->private_map;
}

std::string &PropertiesMap::at(const int &key) {
  return this->private_map.at(key);
};

std::string &PropertiesMap::operator[](const int &key) {
  return this->private_map[key];
};

std::map<int, std::string> get_default_map() {
  return PropertiesMap().get_map();
}

bool map_subset_check(std::map<int, std::string> custom_map) {
  auto default_map = get_default_map();
  auto default_map_size = default_map.size();
  auto custom_map_size = custom_map.size();

  if (custom_map_size < default_map_size) {
    return false;
  }

  for (auto it = default_map.begin(); it != default_map.end(); it++) {
    if (custom_map.find(it->first) == custom_map.end()) {
      return false;
    }
  }

  return true;
}

} // namespace VANTAGE::Reactions
