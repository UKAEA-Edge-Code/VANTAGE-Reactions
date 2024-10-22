#pragma once
#include <map>
#include <neso_particles/typedefs.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace Reactions {

/*! A struct containing an enum with labels consisting of the variable names in
 * standard_properties*/

/*! This can be extended by deriving from this struct and defining a public enum
 * member with the first element being the value of the last element in
 * standard_properties_enum+1. For example:
  struct custom_properties_enum : standard_properties_enum {
    public:
      enum {
        custom_prop_1 = standard_properties_enum::fluid_flow_speed+1,
        custom_prop_2,
        custom_prop_3
      };
  };
  Further chaining would work on the same principle.*/
struct standard_properties_enum {
public:
  enum {
    position,
    velocity,
    cell_id,
    id,
    tot_reaction_rate,
    weight,
    internal_state,
    temperature,
    density,
    flow_speed,
    source_energy,
    source_momentum,
    source_density,
    fluid_density,
    fluid_temperature,
    fluid_flow_speed
  };
};

const auto default_properties = standard_properties_enum();

/*! A struct containing a map to reference strings associated with properties in
 * ParticleSpec via integer indices defined in an enumerator from a struct in
 * ParticlePropertiesIndices. */
struct properties_map {

  properties_map() = default;

  /**
   * @brief properties_map constructor
   *
   * @param custom_map User-provided custom map to replace the default private_map.
   */
  properties_map(std::map<int, std::string> custom_map)
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

public:
  std::map<int, std::string> get_map() { return this->private_map; }

  // Just exposes the bounds-checked accessor to the private_map.
  std::string &at(const int &key) { return this->private_map.at(key); };

private:
  std::map<int, std::string> private_map{
      {default_properties.position, "POSITION"},
      {default_properties.velocity, "VELOCITY"},
      {default_properties.cell_id, "CELL_ID"},
      {default_properties.id, "ID"},
      {default_properties.tot_reaction_rate, "TOT_REACTION_RATE"},
      {default_properties.weight, "WEIGHT"},
      {default_properties.internal_state, "INTERNAL_STATE"},
      {default_properties.temperature, "TEMPERATURE"},
      {default_properties.density, "DENSITY"},
      {default_properties.flow_speed, "FLOW_SPEED"},
      {default_properties.source_energy, "SOURCE_ENERGY"},
      {default_properties.source_momentum, "SOURCE_MOMENTUM"},
      {default_properties.source_density, "SOURCE_DENSITY"},
      {default_properties.fluid_density, "FLUID_DENSITY"},
      {default_properties.fluid_temperature, "FLUID_TEMPERATURE"},
      {default_properties.fluid_flow_speed, "FLUID_FLOW_SPEED"}};
};

const auto default_map = properties_map().get_map();
}; // namespace Reactions
