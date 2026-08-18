
#include "../include/reactions_lib/particle_spec_builder.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

ParticleSpecBuilder::ParticleSpecBuilder(NP::ParticleSpec particle_spec) {
  this->add_particle_spec(particle_spec);
}

ParticleSpecBuilder::ParticleSpecBuilder(
    int ndim, const std::map<int, std::string> &properties_map) {

  NESOWARN(map_subset_check(properties_map),
           "The provided properties_map does not include all the keys from the \
      default_map (and therefore is not an extension of that map). There \
      may be inconsitencies with indexing of properties.");

  this->add_particle_spec(NP::ParticleSpec(
      NP::ParticleProp(
          NP::Sym<REAL>(properties_map.at(default_properties.position)), ndim,
          true),
      NP::ParticleProp(
          NP::Sym<INT>(properties_map.at(default_properties.cell_id)), 1,
          true)));

  auto int_props = Properties<INT>(std::vector<int>{
      default_properties.panic, default_properties.id,
      default_properties.internal_state, default_properties.reacted_flag,
      default_properties.grouping_index, default_properties.linear_index});
  auto real_props_scalar = Properties<REAL>(std::vector<int>{
      default_properties.weight, default_properties.tot_reaction_rate});
  auto real_props_vector =
      Properties<REAL>(std::vector<int>{default_properties.velocity});

  this->add_particle_prop(int_props, 1, false, properties_map);
  this->add_particle_prop(real_props_scalar, 1, false, properties_map);
  this->add_particle_prop(real_props_vector, ndim = ndim, false,
                          properties_map);
}

void ParticleSpecBuilder::add_particle_spec(
    NP::ParticleSpec new_particle_spec) {
  auto existing_properties_real = this->particle_spec.properties_real;
  auto existing_properties_int = this->particle_spec.properties_int;

  std::vector<NP::ParticleProp<REAL>> new_real_props;
  std::vector<NP::ParticleProp<INT>> new_int_props;

  for (auto prop : new_particle_spec.properties_real) {
    if (this->particle_spec.contains(prop)) {
      continue;
    } else {
      new_real_props.push_back(prop);
    }
  }

  for (auto prop : new_particle_spec.properties_int) {
    if (this->particle_spec.contains(prop)) {
      continue;
    } else {
      new_int_props.push_back(prop);
    }
  }

  existing_properties_real.insert(existing_properties_real.end(),
                                  new_real_props.begin(), new_real_props.end());

  existing_properties_int.insert(existing_properties_int.end(),
                                 new_int_props.begin(), new_int_props.end());

  this->particle_spec =
      NP::ParticleSpec(existing_properties_real, existing_properties_int);
}

const NP::ParticleSpec &ParticleSpecBuilder::get_particle_spec() {
  return this->particle_spec;
}

} // namespace VANTAGE::Reactions
