#include "../include/reactions_lib/reaction_kernels_abstract.hpp"

namespace VANTAGE::Reactions {

std::shared_ptr<NP::ProductMatrixSpec> build_kernel_product_schema(
    const ArgumentNameSet<INT> &required_descendant_int_props,
    const ArgumentNameSet<REAL> &required_descendant_real_props,
    int ndim_velocity, const PropertiesMap &properties_map) {
  NP::ParticleSpec particle_spec;
  for (const auto &name : required_descendant_int_props.to_string_vector()) {
    particle_spec.push(NP::ParticleProp<INT>(NP::Sym<INT>(name), 1));
  }
  const auto real_names = required_descendant_real_props.to_string_vector();
  if (!real_names.empty()) {
    const auto prop_map = properties_map.get_map();
    auto vel_it = prop_map.find(default_properties.velocity);
    for (const auto &name : real_names) {
      const bool is_velocity =
          (vel_it != prop_map.end() && name == vel_it->second);
      const int components = is_velocity ? ndim_velocity : 1;
      NESOASSERT(components > 0,
                 "Kernel descendant velocity must have a positive component "
                 "count.");
      particle_spec.push(
          NP::ParticleProp<REAL>(NP::Sym<REAL>(name), components));
    }
  }
  return NP::product_matrix_spec(particle_spec);
}

namespace detail {

void validate_kernel_pre_req_ndims(INT pre_req_ndims) {
  NESOASSERT(pre_req_ndims >= 0,
             "Kernel prerequisite component count must be nonnegative.");
}

} // namespace detail

} // namespace VANTAGE::Reactions
