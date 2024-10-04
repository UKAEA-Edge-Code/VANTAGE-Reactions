#pragma once
#include <map>
#include <string>
#include <utility>

/**
 * @namespace ParticlePropertiesIndices A namespace containing a struct
 * containing an enumerator with labels that correspond to standard property
 * names and a map that maps the values of the enumerator to strings that the
 * properties inside a ParticleSpec need to be formatted in.
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

/*! A map to reference strings associated with properties in ParticleSpec via
 * integer indices defined in an enumerator from a struct in
 * ParticlePropertiesIndices. */
struct properties_map {
  properties_map() = default;

  public:
    std::map<int, std::string> get_map() { return this->private_map; }

    void extend_map(int property_key, std::string property_name) {
      this->private_map.emplace(std::make_pair(property_key, property_name));
    }

    void replace_entry(int old_property_key, int new_property_key, std::string new_property_name) {
      auto entry = this->private_map.extract(old_property_key);
      entry.key() = new_property_key;
      private_map.insert(std::move(entry));
      private_map[new_property_key] = new_property_name;
    }

    void replace_map(std::map<int, std::string> custom_map) {
      this->private_map = custom_map;
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
      {default_properties.flow_speed, "FLOW_SPEED"},
      {default_properties.source_energy, "SOURCE_ENERGY"},
      {default_properties.source_momentum, "SOURCE_MOMENTUM"},
      {default_properties.source_density, "SOURCE_DENSITY"},
      {default_properties.fluid_density, "FLUID_DENSITY"},
      {default_properties.fluid_temperature, "FLUID_TEMPERATURE"},
      {default_properties.fluid_flow_speed, "FLUID_FLOW_SPEED"}};
};

const auto default_map = properties_map().get_map();
}; // namespace ParticlePropertiesIndices
