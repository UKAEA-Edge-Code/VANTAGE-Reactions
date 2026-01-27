#ifndef REACTIONS_LINEAR_SCATTERING_KERNELS_H
#define REACTIONS_LINEAR_SCATTERING_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include <array>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Device type for general linear scattering kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 * @tparam with_sources If true will attempt to write to source properties
 * (defaults to true)
 */
template <int ndim_velocity, bool with_sources>
struct LinearScatteringKernelsOnDevice : public ReactionKernelsBaseOnDevice<1> {
  LinearScatteringKernelsOnDevice() = default;

  /**
   * @brief General scattering kernel - assumes that pre_req_data stores ion
   * velcocities sampled from the ion distribution and sets the product's
   * velocity components to those values
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
   * @param dt The current time step size.
   */
  void scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                         Access::DescendantProducts::Write &descendant_products,
                         Access::SymVector::Write<INT> &req_int_props,
                         Access::SymVector::Write<REAL> &req_real_props,
                         const std::array<int, 1> &out_states,
                         Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                         double dt) const {
    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      descendant_products.at_real(index, 0, descendant_velocity_ind, dimx) =
          pre_req_data.at(index.get_loop_linear_index(), dimx);
    }
  }

  /**
   * @brief Linear scattering weight kernel - simply sets the product's weight
   * to the weight change due to the reaction
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
   * @param dt The current time step size.
   */
  void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                     Access::DescendantProducts::Write &descendant_products,
                     Access::SymVector::Write<INT> &req_int_props,
                     Access::SymVector::Write<REAL> &req_real_props,
                     const std::array<int, 1> &out_states,
                     Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                     double dt) const {
    descendant_products.at_real(index, 0, descendant_weight_ind, 0) =
        modified_weight;
  }

  /**
   * @brief Linear scattering transformation kernel - simply sets the product's
   * ID the target ID
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
   * @param dt The current time step size.
   */
  void
  transformation_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                        Access::DescendantProducts::Write &descendant_products,
                        Access::SymVector::Write<INT> &req_int_props,
                        Access::SymVector::Write<REAL> &req_real_props,
                        const std::array<int, 1> &out_states,
                        Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                        double dt) const {
    descendant_products.at_int(index, 0, descendant_internal_state_ind, 0) =
        out_states[0];
  }

  /**
   * @brief Linear scattering feedback kernel for calculating and applying
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
   * @param dt The current time step size.
   */
  void feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                       Access::DescendantProducts::Write &descendant_products,
                       Access::SymVector::Write<INT> &req_int_props,
                       Access::SymVector::Write<REAL> &req_real_props,
                       const std::array<int, 1> &out_states,
                       Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                       double dt) const {

    if constexpr (with_sources) {
      std::array<REAL, ndim_velocity> k_V_pre, k_V_post;
      REAL delta_vsquared = 0.0;

      for (int vdim = 0; vdim < ndim_velocity; vdim++) {
        k_V_pre[vdim] = req_real_props.at(velocity_ind, index, vdim);
        k_V_post[vdim] = pre_req_data.at(index.get_loop_linear_index(), vdim);
        delta_vsquared +=
            k_V_pre[vdim] * k_V_pre[vdim] - k_V_post[vdim] * k_V_post[vdim];
      }

      // SOURCE_MOMENTUM calc
      for (int sm_dim = 0; sm_dim < ndim_velocity; sm_dim++) {
        req_real_props.at(this->source_momentum_ind, index, sm_dim) +=
            this->mass * modified_weight * (k_V_pre[sm_dim] - k_V_post[sm_dim]);
      }

      // Set SOURCE_ENERGY
      req_real_props.at(this->source_energy_ind, index, 0) +=
          modified_weight * this->mass * delta_vsquared * 0.5;
    }

    req_real_props.at(this->weight_ind, index, 0) -= modified_weight;
  }

public:
  INT velocity_ind, source_momentum_ind, source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL mass;
};

/**
 * @brief Host type for linear scattering kernels - general kernels with
 * post-collision velocities defined by data calculator outputs
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam with_sources If true will track sources (defaults to true)
 */
template <int ndim_velocity = 2, bool with_sources = true>
struct LinearScatteringKernels : public ReactionKernelsBase {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props_with_sources =
      {props.source_momentum, props.source_energy};

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};
  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  /**
   * @brief Constructor for LinearScatteringKernels.
   *
   * @param scattered_species Species object corresponding to the scattered
   * particle
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  LinearScatteringKernels(
      const Species &scattered_species,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(
            (with_sources) ? Properties<REAL>(required_simple_real_props)
                                 .merge_with(Properties<REAL>(
                                     required_simple_real_props_with_sources))
                           : Properties<REAL>(required_simple_real_props),
            ndim_velocity, properties_map) {
    this->linear_scattering_kernels_on_device.velocity_ind =

        this->required_real_props.simple_prop_index(props.velocity,
                                                    this->properties_map);

    if constexpr (with_sources) {
      this->linear_scattering_kernels_on_device.source_momentum_ind =
          this->required_real_props.simple_prop_index(props.source_momentum,
                                                      this->properties_map);

      this->linear_scattering_kernels_on_device.source_energy_ind =
          this->required_real_props.simple_prop_index(props.source_energy,
                                                      this->properties_map);
    }
    this->linear_scattering_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->linear_scattering_kernels_on_device.mass =
        scattered_species.get_mass();

    this->set_required_descendant_int_props(
        Properties<INT>(required_descendant_simple_int_props));

    this->set_required_descendant_real_props(
        Properties<REAL>(required_descendant_simple_real_props));

    this->linear_scattering_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state, this->properties_map);
    this->linear_scattering_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.velocity, this->properties_map);
    this->linear_scattering_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.weight, this->properties_map);

    this->set_descendant_matrix_spec<ndim_velocity, 1>();
  };

private:
  LinearScatteringKernelsOnDevice<ndim_velocity, with_sources>
      linear_scattering_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  LinearScatteringKernelsOnDevice<ndim_velocity, with_sources>
  get_on_device_obj() {
    return this->linear_scattering_kernels_on_device;
  }
};
}; // namespace VANTAGE::Reactions
#endif
