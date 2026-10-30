#include "reactions_lib/reaction_kernel_argument_pack.hpp"
#include <stdexcept>
#include <utility>

namespace VANTAGE::Reactions {

KernelPropertyDescription::KernelPropertyDescription(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    const PropertiesMap &properties_map)
    : required_int_props(std::move(required_int_props), properties_map),
      required_real_props(std::move(required_real_props), properties_map) {}

KernelInputDescription::KernelInputDescription(
    Properties<INT> required_int_props, Properties<REAL> required_real_props,
    Properties<INT> required_int_props_ephemeral,
    Properties<REAL> required_real_props_ephemeral,
    const PropertiesMap &properties_map)
    : KernelPropertyDescription(std::move(required_int_props),
                                std::move(required_real_props), properties_map),
      required_int_props_ephemeral(std::move(required_int_props_ephemeral),
                                   properties_map),
      required_real_props_ephemeral(std::move(required_real_props_ephemeral),
                                    properties_map) {
  this->required_int_props =
      this->required_int_props.merge_with(this->required_int_props_ephemeral);
  this->required_real_props =
      this->required_real_props.merge_with(this->required_real_props_ephemeral);
}

std::shared_ptr<NP::ProductMatrixSpec>
build_kernel_product_schema(KernelPropertyDescription descendants,
                            int ndim_velocity,
                            const PropertiesMap &properties_map) {
  const auto &velocity_name = properties_map.at(default_properties.velocity);
  NP::ParticleSpec particle_spec;
  for (const auto &name : descendants.required_int_props.to_string_vector()) {
    particle_spec.push(NP::ParticleProp<INT>(NP::Sym<INT>(name), 1));
  }
  for (const auto &name : descendants.required_real_props.to_string_vector()) {
    const int components = name == velocity_name ? ndim_velocity : 1;
    if (components <= 0) {
      throw std::invalid_argument("Kernel descendant velocity must have a "
                                  "positive component count.");
    }
    particle_spec.push(NP::ParticleProp<REAL>(NP::Sym<REAL>(name), components));
  }
  return NP::product_matrix_spec(particle_spec);
}

} // namespace VANTAGE::Reactions
