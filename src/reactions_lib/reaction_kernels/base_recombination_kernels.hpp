#ifndef REACTIONS_BASE_RECOMBINATION_KERNELS_H
#define REACTIONS_BASE_RECOMBINATION_KERNELS_H
#include "../particle_properties_map.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels.hpp"
#include <array>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

namespace BASE_RECOMB_KERNEL {
constexpr int num_products_per_parent = 1;

} // namespace BASE_RECOMB_KERNEL

/**
 * @brief A struct that contains SYCL device-compatible kernels for
 * recombination reactions.
 *
 * @tparam ndim_velocity The number of dimensions for the particle
 * velocity property.
 * @tparam ndim_source_momentum The number of dimensions for source
 * momentum property.
 * @tparam has_momentum_req_data The boolean specifying whether a
 * projectile momentum req_data is available.
 */
template <int ndim_velocity, int ndim_source_momentum,
          bool has_momentum_req_data>
struct RecombReactionKernelsOnDevice : public ReactionKernelsBaseOnDevice<1> {
  RecombReactionKernelsOnDevice() = default;

  /**
   * @brief Recombination scattering kernel - assumes that pre_req_data
   * stores the neutral velocities sampled from the existing marker particle
   * distribution and sets the product's velocity to those values
   * (note that the elements of pre_req_data that are relevant in this case
   * are all but the 0th which is reserved for storing data used for
   * calculating the projectile source energy loss).
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which descendant_product_loop is called. Access using either
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
          pre_req_data.at(index.get_loop_linear_index(), 1 + dimx);
    }
  }

  /**
   * @brief Recombination weight kernel - simply sets the product's weight
   * to the weight change due to the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which descendant_product_loop is called. Access using either
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
   * @brief Recombination transformation kernel - simply sets the
   * product's ID to the target ID
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which descendant_product_loop is called. Access using either
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
   * @brief Feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which descendant_product_loop is called. Access using either
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

    std::array<REAL, ndim_velocity> k_V_i;
    REAL visquared = 0.0;
    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V_i[vdim] = pre_req_data.at(index.get_loop_linear_index(), 1 + vdim);
      visquared += k_V_i[vdim] * k_V_i[vdim];
    }

    // Set SOURCE_DENSITY
    req_real_props.at(this->projectile_source_density_ind, index, 0) -=
        modified_weight;

    req_real_props.at(this->target_source_density_ind, index, 0) -=
        modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
      req_real_props.at(this->target_source_momentum_ind, index, sm_dim) -=
          this->target_mass * modified_weight * k_V_i[sm_dim];
    }

    // set SOURCE_ENERGY
    req_real_props.at(this->target_source_energy_ind, index, 0) -=
        this->target_mass * modified_weight * visquared * 0.5;
    req_real_props.at(this->projectile_source_energy_ind, index, 0) -=
        pre_req_data.at(index.get_loop_linear_index(), 0) * dt -
        (this->normalised_potential_energy * modified_weight);
  }

public:
  INT weight_ind;
  INT projectile_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL target_mass, normalised_potential_energy;
};

/**
 * @brief Host type for recombination kernels
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle
 * velocity property (default value of 2)
 * @tparam ndim_source_momentum Optional number of dimensions for source
 * momentum property (default value of ndim_velocity)
 * @tparam Optional boolean specifying whether a projectile momentum req_data
 * is available (default value of false)
 */
template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity,
          bool has_momentum_req_data = false>
struct RecombReactionKernels : public ReactionKernelsBase {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  constexpr static std::array<int, 3> required_species_real_props = {
      props.source_density, props.source_momentum, props.source_energy};

  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  /**
   * @brief Constructor for RecombReactionKerenls.
   *
   * @param target_species Species object representing the recombination
   * target
   * @param projectile_species Species object representing the projectile
   * involved in the recombination (eg. electron).
   * @param normalised_potential_energy Used in calculating the projectile
   * source energy loss
   * @param properties_map A std::map<int, std::string> object to be to be
   * passed to ReactionKernelsBase.
   */
  RecombReactionKernels(
      const Species &target_species, const Species &projectile_species,
      const REAL &normalised_potential_energy,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(
            Properties<REAL>(
                required_simple_real_props,
                std::vector<Species>{target_species, projectile_species},
                required_species_real_props),
            ndim_velocity + 1, properties_map) {
    static_assert((ndim_velocity >= ndim_source_momentum),
                  "Number of dimension for VELOCITY must be greater than or "
                  "equal to number of dimensions for SOURCE_MOMENTUM.");

    this->recomb_reaction_kernels_on_device.normalised_potential_energy =
        normalised_potential_energy;

    this->recomb_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);

    this->recomb_reaction_kernels_on_device.target_source_density_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_density,
                                                     this->properties_map);

    this->recomb_reaction_kernels_on_device.target_source_momentum_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_momentum,
                                                     this->properties_map);

    this->recomb_reaction_kernels_on_device.target_source_energy_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_energy,
                                                     this->properties_map);

    this->recomb_reaction_kernels_on_device.projectile_source_density_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_density,
            this->properties_map);

    this->recomb_reaction_kernels_on_device.projectile_source_momentum_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_momentum,
            this->properties_map);

    this->recomb_reaction_kernels_on_device.projectile_source_energy_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_energy,
            this->properties_map);

    this->recomb_reaction_kernels_on_device.target_mass =
        target_species.get_mass();

    this->set_required_descendant_int_props(
        Properties<INT>(required_descendant_simple_int_props));

    this->set_required_descendant_real_props(
        Properties<REAL>(required_descendant_simple_real_props));

    this->recomb_reaction_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state, this->properties_map);
    this->recomb_reaction_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.velocity, this->properties_map);
    this->recomb_reaction_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(
            props.weight, this->properties_map);

    this->set_descendant_matrix_spec<ndim_velocity, 1>();
  };

private:
  RecombReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                has_momentum_req_data>
      recomb_reaction_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */
  RecombReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                has_momentum_req_data>
  get_on_device_obj() {
    return this->recomb_reaction_kernels_on_device;
  }
};
}; // namespace VANTAGE::Reactions
#endif