#pragma once
#define standard_properties \
  X(position), X(velocity), X(cell_id), X(id), X(tot_reaction_rate), X(weight), X(internal_state), \
  X(electron_temperature), X(electron_density), X(source_energy), X(source_momentum), \
  X(source_density), X(fluid_density), X(fluid_temperature)


#include <map>
#include <string>
#include <vector>

namespace ParticlePropertiesIndices {
#define X(M) M
  enum{standard_properties, num_properties};
#undef X

#define X(M) #M
  static constexpr char const *standard_properties_names[num_properties] = { standard_properties };
#undef X

static const std::map<const char *, std::vector<const char *>> default_map{
    {standard_properties_names[position], {"P", "Position", "POSITION", "position"}},
    {standard_properties_names[velocity], {"V", "Velocity", "VELOCITY", "velocity"}},
    {standard_properties_names[cell_id], {"CELL_ID", "Cell_ID", "cell_id"}},
    {standard_properties_names[id], {"ID", "id"}},
    {standard_properties_names[tot_reaction_rate],
      {"TOT_REACTION_RATE", "Tot_Reaction_Rate", "tot_reaction_rate"}},
    {standard_properties_names[weight],
      {"WEIGHT", "COMPUTATIONAL_WEIGHT", "weight", "computational_weight",
      "Weight", "Computational_Weight"}},
    {standard_properties_names[internal_state],
      {"INTERNAL_STATE", "Internal_State", "internal_state"}},
    {standard_properties_names[electron_temperature],
      {"ELECTRON_TEMPERATURE", "Electron_Temperature",
      "electron_temperature"}},
    {standard_properties_names[electron_density],
      {"ELECTRON_DENSITY", "Electron_Density", "electron_density"}},
    {standard_properties_names[source_energy],
      {"SOURCE_ENERGY", "Source_Energy", "source_energy"}},
    {standard_properties_names[source_momentum],
      {"SOURCE_MOMENTUM", "Source_Momentum", "source_momentum"}},
    {standard_properties_names[source_density],
      {"SOURCE_DENSITY", "Source_Density", "source_density"}},
    {standard_properties_names[fluid_density],
      {"FLUID_DENSITY", "Fluid_Density", "fluid_density"}},
    {standard_properties_names[fluid_temperature],
      {"FLUID_TEMPERATURE", "Fluid_Temperature", "fluid_temperature"}}};
};
#undef standard_properties
// namespace ParticlePropertiesIndices