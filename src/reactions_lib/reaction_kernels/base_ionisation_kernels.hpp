#pragma once
#include <array>
#include <neso_particles.hpp>
#include <particle_properties_map.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <vector>

using namespace NESO::Particles;

namespace BASE_IONISATION_KERNEL {
constexpr int num_products_per_parent = 0;

const auto props = ParticlePropertiesIndices::default_properties;

const std::vector<int> required_simple_real_props = {props.weight,
                                                     props.velocity};

const std::vector<int> required_species_real_props = {
    props.source_density, props.source_energy,
    props.source_momentum}; // namespace BASE_IONISATION_KERNEL
} // namespace BASE_IONISATION_KERNEL

/**
 * struct IoniseReactionKernelsOnDevice - SYCL device-compatible kernel for
 * ionisation reactions. Defaults to a 2V model.
 */
template <int ndim_velocity, int ndim_source_momentum,
          bool has_momentum_req_data>
struct IoniseReactionKernelsOnDevice
    : public ReactionKernelsBaseOnDevice<
          BASE_IONISATION_KERNEL::num_products_per_parent> {
  IoniseReactionKernelsOnDevice() = default;

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
   * that may need to operated on
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for operations inside the kernel.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for operations inside the kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued NDLocalArray containing pre-calculated data
   * @param dt The current time step size.
   */
  void feedback_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, BASE_IONISATION_KERNEL::num_products_per_parent>
          &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    std::array<REAL, ndim_velocity> k_V;
    REAL vsquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      vsquared += k_V[vdim] * k_V[vdim];
    }

    // Set SOURCE_DENSITY
    req_real_props.at(this->electron_source_density_ind, index, 0) +=
        modified_weight;

    req_real_props.at(this->target_source_density_ind, index, 0) +=
        modified_weight;

    // SOURCE_MOMENTUM calc
    for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
      req_real_props.at(this->target_source_momentum_ind, index, sm_dim) +=
          this->target_mass * modified_weight * k_V[sm_dim];
    }

    // Treat momentum sources when a projectile momentum req_data is available
    if (has_momentum_req_data) {

      for (int sm_dim = 0; sm_dim < ndim_source_momentum; sm_dim++) {
        req_real_props.at(this->target_source_momentum_ind, index, sm_dim) +=
            pre_req_data.at(index.get_loop_linear_index(), 1) * dt;
        req_real_props.at(this->projectile_source_momentum_ind, index,
                          sm_dim) +=
            pre_req_data.at(index.get_loop_linear_index(), 1) * dt;
      }
    }

    // Set SOURCE_ENERGY
    req_real_props.at(this->target_source_energy_ind, index, 0) +=
        modified_weight * this->target_mass * vsquared * 0.5;

    req_real_props.at(this->projectile_source_energy_ind, index, 0) -=
        pre_req_data.at(index.get_loop_linear_index(), 0) * dt;

    req_real_props.at(this->weight_ind, index, 0) -= modified_weight;
  }

public:
  INT velocity_ind, electron_source_density_ind, projectile_source_energy_ind,
      projectile_source_momentum_ind, target_source_density_ind,
      target_source_momentum_ind, target_source_energy_ind, weight_ind;
  REAL target_mass;
};

/**
 * @brief A struct defining data and kernel functions needed for an ionisation
 * reaction.
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 * @tparam ndim_source_momentum Number of dimensions for electron
 * source momentum property (default value of 2)
 * @tparam has_momentum_req_data Optional boolean specifying whether a
 * projectile momentum req_data is available
 */
template <int ndim_velocity = 2, int ndim_source_momentum = ndim_velocity,
          bool has_momentum_req_data = false>
struct IoniseReactionKernels : public ReactionKernelsBase {

  /**
   * @brief Ionisation reaction kernel host type constructor
   *
   * @param target_species Species object representing the ionisation target
   * (and the corresponding ion field!)
   * @param electron_species Species object representing the electrons
   * @param projectile_species Species object representing the projectile
   * species
   */
  IoniseReactionKernels(const Species &target_species,
                        const Species &electron_species,
                        const Species &projectile_species)
      : ReactionKernelsBase(
            Properties<REAL>(
                BASE_IONISATION_KERNEL::required_simple_real_props,
                std::vector<Species>{target_species, electron_species,
                                     projectile_species},
                BASE_IONISATION_KERNEL::required_species_real_props),
            has_momentum_req_data ? 2 : 1) {
    static_assert(
        (ndim_velocity >= ndim_source_momentum),
        "Number of dimension for VELOCITY must be greater than or "
        "equal to number of dimensions for ELECTRON_SOURCE_MOMENTUM.");

    auto props = BASE_IONISATION_KERNEL::props;

    this->ionise_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity);

    this->ionise_reaction_kernels_on_device.electron_source_density_ind =
        this->required_real_props.species_prop_index(
            electron_species.get_name(), props.source_density);

    this->ionise_reaction_kernels_on_device.target_source_density_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_density);

    this->ionise_reaction_kernels_on_device.target_source_momentum_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_momentum);

    this->ionise_reaction_kernels_on_device.target_source_energy_ind =
        this->required_real_props.species_prop_index(target_species.get_name(),
                                                     props.source_energy);

    this->ionise_reaction_kernels_on_device.projectile_source_momentum_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_momentum);

    this->ionise_reaction_kernels_on_device.projectile_source_energy_ind =
        this->required_real_props.species_prop_index(
            projectile_species.get_name(), props.source_energy);

    this->ionise_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight);

    this->ionise_reaction_kernels_on_device.target_mass =
        target_species.get_mass();
  };

private:
  IoniseReactionKernelsOnDevice<ndim_velocity, ndim_source_momentum,
                                has_momentum_req_data>
      ionise_reaction_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  auto get_on_device_obj() { return this->ionise_reaction_kernels_on_device; }
};
