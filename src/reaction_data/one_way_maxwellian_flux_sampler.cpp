#include "../../include/reactions_lib/reaction_data/one_way_maxwellian_flux_sampler.hpp"

namespace VANTAGE::Reactions {

OneWayMaxwellianFluxSampler::OneWayMaxwellianFluxSampler(
    const REAL &norm_ratio,
    std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel,
    std::map<int, std::string> properties_map)
    : ReactionDataBase<OneWayMaxwellianFluxOnDevice, 3,
                       HostAtomicBlockKernelRNG<REAL>>(
          Properties<INT>(required_simple_int_props),
          Properties<REAL>(required_simple_real_props), properties_map) {
  this->on_device_obj = OneWayMaxwellianFluxOnDevice(norm_ratio);
  this->set_rng_kernel(rng_kernel);
  this->index_on_device_object();
}

void OneWayMaxwellianFluxSampler::index_on_device_object() {
  this->on_device_obj->fluid_flow_speed_ind =
      this->required_real_props.find_index(
          this->properties_map.at(props.fluid_flow_speed));

  this->on_device_obj->fluid_temperature_ind =
      this->required_real_props.find_index(
          this->properties_map.at(props.fluid_temperature));

  this->on_device_obj->basis_e1_ind = this->required_real_props.find_index(
      this->properties_map.at(props.surface_basis_e1));

  this->on_device_obj->basis_e2_ind = this->required_real_props.find_index(
      this->properties_map.at(props.surface_basis_e2));

  this->on_device_obj->basis_pi_ind = this->required_real_props.find_index(
      this->properties_map.at(props.surface_basis_pi));

  this->on_device_obj->panic_ind =
      this->required_int_props.find_index(this->properties_map.at(props.panic));
}

} // namespace VANTAGE::Reactions