#pragma once
#include <array>
#include <gtest/gtest.h>
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
    props.density, props.source_density, props.source_energy,
    props.source_momentum}; // namespace BASE_IONISATION_KERNEL
} // namespace BASE_IONISATION_KERNEL

/**
 * @brief A struct that contains data and kernel functions that are to be stored
 * on and used on a SYCL device.
 */
template <int ndim_velocity = 1, int ndim_electron_source_momentum = 1>
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
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  void feedback_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, BASE_IONISATION_KERNEL::num_products_per_parent>
          &out_states,
      Access::NDLocalArray::Read<REAL,2> &pre_req_data, double dt) const {

    std::array<REAL, ndim_velocity> k_V;
    REAL vsquared = 0.0;

    for (int vdim = 0; vdim < ndim_velocity; vdim++) {
      k_V[vdim] = req_real_props.at(velocity_ind, index, vdim);
      vsquared += k_V[vdim] * k_V[vdim];
    }

    REAL inv_k_dt = 1.0 / dt;

    // Set SOURCE_DENSITY
    req_real_props.at(electron_source_density_ind, index, 0) +=
        modified_weight * inv_k_dt;

    // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
    auto k_SD = req_real_props.at(electron_source_density_ind, index, 0);

    // SOURCE_MOMENTUM calc
    for (int esm_dim = 0; esm_dim < ndim_electron_source_momentum; esm_dim++) {
      req_real_props.at(electron_source_momentum_ind, index, esm_dim) +=
          k_SD * k_V[esm_dim];
    }

    // Set SOURCE_ENERGY
    req_real_props.at(electron_source_energy_ind, index, 0) +=
        k_SD * vsquared * 0.5;

    req_real_props.at(weight_ind, index, 0) -= modified_weight;
  }

public:
  int velocity_ind, electron_density_ind, electron_source_density_ind,
      electron_source_momentum_ind, electron_source_energy_ind, weight_ind;
};

/**
 * @brief A struct defining data and kernel functions needed for an ionisation
 * reaction.
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 1)
 * @tparam ndim_electron_source_momentum Number of dimensions for electron
 * source momentum property (default value of 1)
 * @param species_ A vector of Species objects that define the species(plural)
 * that will be acted on by an ionisation reaction.
 */
template <int ndim_velocity = 1, int ndim_electron_source_momentum = 1>
struct IoniseReactionKernels : public ReactionKernelsBase {

  IoniseReactionKernels(const Species species_)
      : ReactionKernelsBase(), species(species_),
        required_real_props(Properties<REAL>(
            BASE_IONISATION_KERNEL::required_simple_real_props,
            std::vector<Species>{this->species},
            BASE_IONISATION_KERNEL::required_species_real_props)) {

    static_assert(
        (ndim_velocity >= ndim_electron_source_momentum),
        "Number of dimension for VELOCITY must be greater than or "
        "equal to number of dimensions for ELECTRON_SOURCE_MOMENTUM.");

    auto species_name = this->species.get_name();

    try {
      if (species_name != "ELECTRON") {
        throw;
      }
    } catch (...) {
      std::cout << "Warning! Species name given is not ELECTRON..."
                << std::endl;
    }

    auto props = BASE_IONISATION_KERNEL::props;

    this->ionise_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.simple_prop_index(props.velocity);

    this->ionise_reaction_kernels_on_device.electron_density_ind =
        this->required_real_props.species_prop_index(species_name,
                                                     props.density);

    this->ionise_reaction_kernels_on_device.electron_source_density_ind =
        this->required_real_props.species_prop_index(species_name,
                                                     props.source_density);

    this->ionise_reaction_kernels_on_device.electron_source_momentum_ind =
        this->required_real_props.species_prop_index(species_name,
                                                     props.source_momentum);

    this->ionise_reaction_kernels_on_device.electron_source_energy_ind =
        this->required_real_props.species_prop_index(species_name,
                                                     props.source_energy);

    this->ionise_reaction_kernels_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight);
  };

private:
  IoniseReactionKernelsOnDevice<ndim_velocity, ndim_electron_source_momentum>
      ionise_reaction_kernels_on_device;

  Species species;

  Properties<REAL> required_real_props;

public:
  /**
   * @brief Getters for the names of all properties and the SYCL
   * device-specific struct.
   */

  std::vector<std::string> get_required_real_props() {
    std::vector<std::string> simple_props;
    try {
      simple_props = this->required_real_props.simple_prop_names();
    } catch (std::logic_error) {
      simple_props = {};
    }
    std::vector<std::string> species_props;
    try {
      species_props = this->required_real_props.species_prop_names();
    } catch (std::logic_error) {
      species_props = {};
    }
    simple_props.insert(simple_props.end(), species_props.begin(),
                        species_props.end());
    return simple_props;
  }

  IoniseReactionKernelsOnDevice<ndim_velocity, ndim_electron_source_momentum>
  get_on_device_obj() {
    return this->ionise_reaction_kernels_on_device;
  }
};
