#pragma once
#include "particle_properties_map.hpp"
#include <algorithm>
#include <cstring>
#include <utility>

/**
 * @def standard_properties List of variable names that are relevant to the
 * properties used in Reactions workflows.
 */
// Replace with struct containing ints with facility to extend and a default
// one in case
// #define standard_properties \
//   X(position), X(velocity), X(cell_id), X(id), X(tot_reaction_rate), \
//       X(weight), X(internal_state), X(temperature), X(density), \
//       X(source_energy), X(source_momentum), X(source_density), \
//       X(fluid_density), X(fluid_temperature)

#include <map>
#include <string>
#include <vector>

/**
 * @namespace ParticlePropertiesIndices A namespace containing an enumerator
 * with labels that correspond to the standard_properties macro and a map that
 * maps the values of the enumerator to strings that the properties inside a
 * ParticleSpec need to be formatted in.
 */
namespace ParticlePropertiesIndices {
/*! An enum with labels consisting of the variable names in
 * standard_properties*/

/*! This can be extended by deriving from this struct and defining a public enum
 * member with the first element being the value of the last element in
 * standard_properties_enum+1. For example:
  struct custom_properties_enum : standard_properties_enum {
    public:
      enum {
        custom_prop_1 = standard_properties_enum::fluid_temparture+1,
        custom_prop_2,
        custom_prop_3
      }
  }
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
    source_energy,
    source_momentum,
    source_density,
    fluid_density,
    fluid_temperature
  };
};

const auto default_properties = standard_properties_enum();

// Think about a way for the user to add their own maps and simplify definition
// and construction
/*! A map to reference strings associated with properties in ParticleSpec via
 * integer indices defined in an enumerator in ParticlePropertiesIndices. */
struct properties_map {
  properties_map() = default;

  std::map<int, std::string> get_map() { return this->private_map; }

  void extend_map(int property_key, std::string property_name) {
    this->private_map.emplace(std::make_pair(property_key, property_name));
  }

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
      {default_properties.source_energy, "SOURCE_ENERGY"},
      {default_properties.source_momentum, "SOURCE_MOMENTUM"},
      {default_properties.source_density, "SOURCE_DENSITY"},
      {default_properties.fluid_density, "FLUID_DENSITY"},
      {default_properties.fluid_temperature, "FLUID_TEMPERATURE"}};
};

const auto default_map = properties_map().get_map();
}; // namespace ParticlePropertiesIndices
// #undef standard_properties
// namespace ParticlePropertiesIndices