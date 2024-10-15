#pragma once
#include <array>
#include <neso_particles.hpp>
#include <neso_particles/containers/product_matrix.hpp>
#include <particle_properties_map.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <vector>

// TODO: docs double-check
// TODO: consistency - should we really use num_products_per_parent and then no
// loops?

using namespace NESO::Particles;

namespace BASE_CX_KERNEL {
constexpr int num_products_per_parent = 1;

const auto props = ParticlePropertiesIndices::default_properties;

const std::vector<int> required_simple_real_props = {props.weight,
                                                     props.velocity};

const std::vector<int> required_species_real_props = {
    props.source_density, props.source_energy,
    props.source_momentum}; // namespace BASE_CX_KERNEL

const std::vector<int> required_descendant_simple_int_props = {
    props.internal_state};
const std::vector<int> required_descendant_simple_real_props = {props.velocity,
                                                                props.weight};
} // namespace BASE_CX_KERNEL

/**
 * struct CXReactionKernelsOnDevice - SYCL device-compatible kernel for
 * charge exchange reactions.
 */
template <int ndim_velocity, int ndim_source_momentum>
struct CXReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<
          BASE_CX_KERNEL::num_products_per_parent> {
  CXReactionKernelsOnDevice() = default;

  /**
   * @brief CX scattering kernel - assumes that pre_req_data stores ion
   * velcocities sampled from the ion distribution and sets the product's
   * velocity components to those values
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
  void scattering_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, BASE_CX_KERNEL::num_products_per_parent>
          &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      descendant_products.at_real(index, 0, descendant_velocity_ind, dimx) =
          pre_req_data.at(index.get_loop_linear_index(), dimx);
    }
  }

  /**
   * @brief CX weight kernel - simply sets the product's weight the weight
   * change due to the reaction
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
  weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                Access::DescendantProducts::Write &descendant_products,
                Access::SymVector::Write<INT> &req_int_props,
                Access::SymVector::Write<REAL> &req_real_props,
                const std::array<int, BASE_CX_KERNEL::num_products_per_parent>
                    &out_states,
                Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                double dt) const {
    descendant_products.at_real(index, 0, descendant_weight_ind, 0) =
        modified_weight;
  }

  /**
   * @brief CX transformation kernel - simply sets the product's ID the target
   * ID
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
  void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, BASE_CX_KERNEL::num_products_per_parent>
          &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
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
  void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_int_props,
                  Access::SymVector::Write<REAL> &req_real_props,
                  const std::array<int, BASE_CX_KERNEL::num_products_per_parent>
                      &out_states,
                  Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                  double dt) const {

    std::array<REAL, ndim_velocity> k_V, k_Vi;
    REAL vsquared = 0.0;
    REAL visquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      k_Vi[vdim] = pre_req_data.at(index.get_loop_linear_index(), vdim);
      vsquared += k_V[vdim] * k_V[vdim];
      visquared += k_Vi[vdim] * k_Vi[vdim];
    }

    // Set SOURCE_DENSITY
    req_real_props.at(this->projectile_source_density_ind, index, 0) +=
        modified_weight;

    req_real_props.at(this->target_source_density_ind, index, 0) -=
        modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
      req_real_props.at(this->target_source_momentum_ind, index, sm_dim) -=
          this->target_mass * modified_weight * k_Vi[sm_dim];
      req_real_props.at(this->projectile_source_momentum_ind, index, sm_dim) +=
          this->projectile_mass * modified_weight * k_V[sm_dim];
    }

    // Set SOURCE_ENERGY
    req_real_props.at(this->target_source_energy_ind, index, 0) -=
        modified_weight * this->target_mass * visquared * 0.5;

    req_real_props.at(this->projectile_source_energy_ind, index, 0) +=
        modified_weight * this->projectile_mass * vsquared * 0.5;

    req_real_props.at(this->weight_ind, index, 0) -= modified_weight;
  }

public:
  INT velocity_ind, projectile_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
  REAL target_mass, projectile_mass;
};

/**
 * @brief A struct defining data and kernel functions needed for a charge
 * exchange reaction.
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam ndim_source_momentum Optional number of dimensions for electron
 * source momentum property (default value of ndim_velocity)
 */
template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity>
struct CXReactionKernels : public ReactionKernelsBase {

  /**
   * @brief Charge exchange reaction kernel host type constructor
   *
   * @param target_species Species object representing the charge exchange
   * target - the ingoing ion and outgoing neutral
   * @param projectile_species Species object representing the projectile
   * species - the outgoing ion and ingoing neutral
   */
  CXReactionKernels(const Species &target_species,
                    const Species &projectile_species,
                    std::map<int, std::string> properties_map_=ParticlePropertiesIndices::default_map)
      : ReactionKernelsBase(
            Properties<REAL>(
                BASE_CX_KERNEL::required_simple_real_props,
                std::vector<Species>{target_species, projectile_species},
                BASE_CX_KERNEL::required_species_real_props),
            ndim_velocity, properties_map_) {

    static_assert((ndim_velocity >= ndim_source_momentum),
                  "Number of dimension for VELOCITY must be greater than or "
                  "equal to number of dimensions for SOURCE_MOMENTUM.");

    this->set_required_descendant_int_props(
        Properties<INT>(BASE_CX_KERNEL::required_descendant_simple_int_props));

    this->set_required_descendant_real_props(Properties<REAL>(
        BASE_CX_KERNEL::required_descendant_simple_real_props));

    auto props = BASE_CX_KERNEL::props;

    this->cx_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity, this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_density_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_density, this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_momentum_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_momentum, this->properties_map);

    this->cx_reaction_kernels_on_device.target_source_energy_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_energy, this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_density_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_density, this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_momentum_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_momentum, this->properties_map);

    this->cx_reaction_kernels_on_device.projectile_source_energy_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_energy, this->properties_map);

    this->cx_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight, this->properties_map);

    this->cx_reaction_kernels_on_device.target_mass = target_species.get_mass();
    this->cx_reaction_kernels_on_device.projectile_mass =
        projectile_species.get_mass();

    this->cx_reaction_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props.simple_prop_index(
            props.internal_state, this->properties_map);
    this->cx_reaction_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props.simple_prop_index(props.velocity, this->properties_map);
    this->cx_reaction_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props.simple_prop_index(props.weight, this->properties_map);

    const auto descendant_internal_state_prop =
        ParticleProp<INT>(Sym<INT>(ParticlePropertiesIndices::default_map.at(
                              props.internal_state)),
                          1);
    const auto descendant_velocity_prop = ParticleProp<REAL>(
        Sym<REAL>(this->properties_map.at(props.velocity)),
        ndim_velocity);
    const auto descendant_weight_prop = ParticleProp<REAL>(
        Sym<REAL>(this->properties_map.at(props.weight)), 1);

    auto descendant_particles_spec = ParticleSpec();
    descendant_particles_spec.push(descendant_internal_state_prop);
    descendant_particles_spec.push(descendant_velocity_prop);
    descendant_particles_spec.push(descendant_weight_prop);

    auto matrix_spec = product_matrix_spec(descendant_particles_spec);

    this->set_descendant_matrix_spec(matrix_spec);
  };

private:
  CXReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum>
      cx_reaction_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  CXReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum>
  get_on_device_obj() {
    return this->cx_reaction_kernels_on_device;
  }
};
