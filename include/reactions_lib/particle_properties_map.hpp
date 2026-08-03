#ifndef REACTIONS_PARTICLE_PROPERTIES_MAP_H
#define REACTIONS_PARTICLE_PROPERTIES_MAP_H
#include <map>
#include <neso_particles.hpp>
#include <string>
#include <utility>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief Data from this struct is used to access property names in a map from
 * PropertiesMap.
 *
 * This can be extended by deriving from this struct and defining a public enum
 * member with the first element being the value of the last element in
 * StandardPropertiesEnum+1. For example:
 * ```
 *     struct CustomPropertiesEnum : StandardPropertiesEnum {
 *       public:
 *         enum {
 *           custom_prop_1 = StandardPropertiesEnum::fluid_flow_speed+1,
 *           custom_prop_2,
 *           custom_prop_3
 *         };
 *     };
 * ```
 * Further chaining would work on the same principle.
 */
struct StandardPropertiesEnum {
public:
  enum StandardPropertyID {
    reacted_flag,
    grouping_index,
    linear_index,
    panic,
    position,
    velocity,
    cell_id,
    id,
    tot_reaction_rate,
    weight,
    internal_state,
    boundary_intersection_point,
    boundary_intersection_normal,
    boundary_intersection_metadata,
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

const auto default_properties = StandardPropertiesEnum();

/**
 * @brief Used to define mappings between integer indices defined in an
 * enumerator from a StandardPropertiesEnum to Sym names.
 */
struct PropertiesMap {

  PropertiesMap() = default;

  /**
   * @brief Constructor for PropertiesMap.
   *
   * @param custom_map User-provided custom map to replace the default
   * private_map.
   */
  PropertiesMap(std::map<int, std::string> custom_map);

public:
  std::map<int, std::string> get_map() { return this->private_map; }

  // Just exposes the bounds-checked accessor to the private_map.
  std::string &at(const int &key) { return this->private_map.at(key); };

  std::string &operator[](const int &key) { return this->private_map[key]; };

private:
  std::map<int, std::string> private_map{
      {default_properties.reacted_flag, "PARTICLE_REACTED_FLAG"},
      {default_properties.grouping_index, "REACTIONS_GROUPING_INDEX"},
      {default_properties.linear_index, "REACTIONS_LINEAR_INDEX"},
      {default_properties.panic, "REACTIONS_PANIC_FLAG"},
      {default_properties.position, "POSITION"},
      {default_properties.velocity, "VELOCITY"},
      {default_properties.cell_id, "CELL_ID"},
      {default_properties.id, "ID"},
      {default_properties.tot_reaction_rate, "TOT_REACTION_RATE"},
      {default_properties.weight, "WEIGHT"},
      {default_properties.internal_state, "INTERNAL_STATE"},
      {default_properties.boundary_intersection_point,
       BoundaryInteractionSpecification::intersection_point.name},
      {default_properties.boundary_intersection_normal,
       BoundaryInteractionSpecification::intersection_normal.name},
      {default_properties.boundary_intersection_metadata,
       BoundaryInteractionSpecification::intersection_metadata.name},
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

std::map<int, std::string> get_default_map();

/**
 * @brief Function to check whether a custom map is a subset of the default map.
 *
 * @return True if the given custom map is a subset of the default map.
 */
bool map_subset_check(std::map<int, std::string> custom_map);
}; // namespace VANTAGE::Reactions

#endif
