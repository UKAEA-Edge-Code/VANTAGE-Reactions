#ifndef REACTIONS_ABSORPTION_KERNELS_H
#define REACTIONS_ABSORPTION_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include <array>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Device type for general absorption kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 */
template <int ndim_velocity>
struct GeneralAbsorptionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<0> {
  GeneralAbsorptionKernelsOnDevice() = default;

  /**
   * @brief General absorption feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which apply is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to be operated on
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for operations inside the kernel.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for operations inside the kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued NDLocalArray containing pre-calculated data
   * (none expected)
   * @param dt The current time step size.
   */
  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Write<INT> &req_int_props,
                       Access::SymVector::Write<REAL> &req_real_props,
                       const std::array<int, 0> &out_states,
                       Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                       double dt) const {

    std::array<REAL, ndim_velocity> k_V_pre;
    REAL delta_vsquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V_pre[vdim] = req_real_props.at(velocity_ind, index, vdim);
      delta_vsquared += k_V_pre[vdim] * k_V_pre[vdim];
    }

    // SOURCE_DENSITY calc
    req_real_props.at(this->source_density_ind, index, 0) += modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
      req_real_props.at(this->source_momentum_ind, index, sm_dim) +=
          this->mass * modified_weight * k_V_pre[sm_dim];
    }

    // Set SOURCE_ENERGY
    req_real_props.at(this->source_energy_ind, index, 0) +=
        modified_weight * this->mass * delta_vsquared * 0.5;

    req_real_props.at(this->weight_ind, index, 0) -= modified_weight;
  }

public:
  INT velocity_ind, source_density_ind, source_momentum_ind, source_energy_ind,
      weight_ind;
  REAL mass;
};

/**
 * @brief Host type for general absorption kernels
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 */
template <int ndim_velocity = 2>
struct GeneralAbsorptionKernels : public ReactionKernelsBase {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 5> required_simple_real_props = {
      props.weight, props.source_density, props.velocity, props.source_momentum,
      props.source_energy};

  /**
   * @brief Constructor for GeneralAbsorptionKernels.
   *
   * @param absorbed_species Species object corresponding to the absorbed
   * particle
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  GeneralAbsorptionKernels(
      const Species &absorbed_species,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(Properties<REAL>(required_simple_real_props), 0,
                            properties_map) {
    this->absorption_kernels_on_device.velocity_ind =

        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);

    this->absorption_kernels_on_device.source_density_ind =
        this->required_real_props.simple_prop_index(props.source_density,
                                                    this->properties_map);

    this->absorption_kernels_on_device.source_momentum_ind =
        this->required_real_props.simple_prop_index(props.source_momentum,
                                                    this->properties_map);

    this->absorption_kernels_on_device.source_energy_ind =
        this->required_real_props.simple_prop_index(props.source_energy,
                                                    this->properties_map);

    this->absorption_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->absorption_kernels_on_device.mass = absorbed_species.get_mass();
  };

private:
  GeneralAbsorptionKernelsOnDevice<ndim_velocity> absorption_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  GeneralAbsorptionKernelsOnDevice<ndim_velocity> get_on_device_obj() {
    return this->absorption_kernels_on_device;
  }
};
}; // namespace VANTAGE::Reactions
#endif
