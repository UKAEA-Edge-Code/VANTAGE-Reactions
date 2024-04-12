#pragma once
#include <map>
#include <string>
#include <vector>

namespace ParticlePropertiesIndices {
enum prop_inds {
  position,
  velocity,
  cell_id,
  id,
  tot_reaction_rate,
  weight,
  internal_state,
  electron_temperature,
  electron_density,
  source_energy,
  source_momentum,
  source_density,
  fluid_density,
  fluid_temperature
};

struct Map {
  std::map<prop_inds, std::vector<std::string>> default_map{
      {prop_inds::position, {"P", "Position", "POSITION", "position"}},
      {prop_inds::velocity, {"V", "Velocity", "VELOCITY", "velocity"}},
      {prop_inds::cell_id, {"CELL_ID", "Cell_ID", "cell_id"}},
      {prop_inds::id, {"ID", "id"}},
      {prop_inds::tot_reaction_rate,
       {"TOT_REACTION_RATE", "Tot_Reaction_Rate", "tot_reaction_rate"}},
      {prop_inds::weight,
       {"WEIGHT", "COMPUTATIONAL_WEIGHT", "weight", "computational_weight",
        "Weight", "Computational_Weight"}},
      {prop_inds::internal_state,
       {"INTERNAL_STATE", "Internal_State", "internal_state"}},
      {prop_inds::electron_temperature,
       {"ELECTRON_TEMPERATURE", "Electron_Temperature",
        "electron_temperature"}},
      {prop_inds::electron_density,
       {"ELECTRON_DENSITY", "Electron_Density", "electron_density"}},
      {prop_inds::source_energy,
       {"SOURCE_ENERGY", "Source_Energy", "source_energy"}},
      {prop_inds::source_momentum,
       {"SOURCE_MOMENTUM", "Source_Momentum", "source_momentum"}},
      {prop_inds::source_density,
       {"SOURCE_DENSITY", "Source_Density", "source_density"}},
      {prop_inds::fluid_density,
       {"FLUID_DENSITY", "Fluid_Density", "fluid_density"}},
      {prop_inds::fluid_temperature,
       {"FLUID_TEMPERATURE", "Fluid_Temperature", "fluid_temperature"}}};
};
} // namespace ParticlePropertiesIndices