#pragma once
#include <cstring>
#define standard_properties \
  X(position), X(velocity), X(cell_id), X(id), X(tot_reaction_rate), X(weight), X(internal_state), \
  X(temperature), X(density), X(source_energy), X(source_momentum), \
  X(source_density), X(fluid_density), X(fluid_temperature)


#include <map>
#include <string>
#include <vector>

namespace ParticlePropertiesIndices {
#define X(M) M
  enum{standard_properties, num_properties};
#undef X

// Think about a way for the user to add their own maps and simplify definition and construction
const std::map<int, std::string> default_map{
    {position, "POSITION"},
    {velocity, "VELOCITY"},
    {cell_id, "CELL_ID"},
    {id, "ID"},
    {tot_reaction_rate,
      "TOT_REACTION_RATE"},
    {weight,
      "WEIGHT"},
    {internal_state,
      "INTERNAL_STATE"},
    {temperature,
      "TEMPERATURE"},
    {density,
      "DENSITY"},
    {source_energy,
      "SOURCE_ENERGY"},
    {source_momentum,
      "SOURCE_MOMENTUM"},
    {source_density,
      "SOURCE_DENSITY"},
    {fluid_density,
      "FLUID_DENSITY"},
    {fluid_temperature,
      "FLUID_TEMPERATURE"}};
};
#undef standard_properties
// namespace ParticlePropertiesIndices