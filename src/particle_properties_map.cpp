#include "../include/reactions_lib/particle_properties_map.hpp"

namespace VANTAGE::Reactions {

PropertiesMap::PropertiesMap(std::map<int, std::string> custom_map)
    : private_map(custom_map) {
  NESOWARN(map_subset_check(this->private_map),
           "The provided properties map does not include all the keys from the "
           "default map (and therefore is not an extension of that map). There "
           "may be inconsistencies with indexing of properties.");
}

std::map<int, std::string> PropertiesMap::get_map() const {
  return this->private_map;
}

std::string &PropertiesMap::at(const int &key) {
  return this->private_map.at(key);
}

const std::string &PropertiesMap::at(const int &key) const {
  return this->private_map.at(key);
}

std::string &PropertiesMap::operator[](const int &key) {
  return this->private_map[key];
};

void PropertiesMap::erase(const int &key) { this->private_map.erase(key); }

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

bool map_subset_check(const PropertiesMap &custom_map) {
  return map_subset_check(custom_map.get_map());
}

} // namespace VANTAGE::Reactions
