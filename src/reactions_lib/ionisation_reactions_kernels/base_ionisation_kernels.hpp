#pragma once
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <neso_particles.hpp>
#include <particle_properties_map.hpp>
#include <reaction_kernel_pre_reqs.hpp>
#include <reaction_kernels.hpp>
#include <string_view>
#include <vector>

using namespace NESO::Particles;
using namespace ParticlePropertiesIndices;

namespace BASE_IONISATION_KERNEL {
constexpr int num_products_per_parent = 0;

const std::vector<int> required_simple_real_props = {velocity, weight};

const std::vector<int> required_species_real_props = {
    density, source_density, source_momentum, source_energy};
} // namespace BASE_IONISATION_KERNEL

/**
 * @brief A struct that contains data and kernel functions that are to be stored
 * on and used on a SYCL device.
 */
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
   * @param req_simple_prop_ints Vector of symbols for simple integer-valued
   * properties that need to be used for operations inside the kernel.
   * @param req_simple_prop_reals Vector of symbols for simple real-valued
   * properties that need to be used for operations inside the kernel.
   * @param req_species_prop_ints Vector of symbols for species-dependent
   * integer-valued properties that need to be used for operations inside the
   * kernel.
   * @param req_species_prop_reals Vector of symbols for species-dependent
   * real-valued properties that need to be used for operations inside the
   * kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  void feedback_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_simple_prop_ints,
      Access::SymVector::Write<REAL> &req_simple_prop_reals,
      Access::SymVector::Write<INT> &req_species_prop_ints,
      Access::SymVector::Write<REAL> &req_species_prop_reals,
      const std::array<int, BASE_IONISATION_KERNEL::num_products_per_parent>
          &out_states,
      Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
    auto k_V_0 = req_simple_prop_reals.at(velocity_ind, index, 0);
    auto k_V_1 = req_simple_prop_reals.at(velocity_ind, index, 1);

    const REAL vsquared = (k_V_0 * k_V_0) + (k_V_1 * k_V_1);

    REAL k_n_scale = 1.0; // / test_reaction_data.get_n_to_SI();
    REAL inv_k_dt = 1.0 / dt;

    auto nE = req_species_prop_reals.at(electron_density_ind, index, 0);

    // Set SOURCE_DENSITY
    req_species_prop_reals.at(electron_source_density_ind, index, 0) +=
        nE * modified_weight * k_n_scale * inv_k_dt;

    // Get SOURCE_DENSITY for SOURCE_MOMENTUM calc
    auto k_SD =
        req_species_prop_reals.at(electron_source_density_ind, index, 0);
    req_species_prop_reals.at(electron_source_momentum_ind, index, 0) +=
        k_SD * k_V_0;
    req_species_prop_reals.at(electron_source_momentum_ind, index, 1) +=
        k_SD * k_V_1;

    // Set SOURCE_ENERGY
    req_species_prop_reals.at(electron_source_energy_ind, index, 0) +=
        k_SD * vsquared * 0.5;

    req_simple_prop_reals.at(weight_ind, index, 0) -= modified_weight;
  }

  int velocity_ind, electron_density_ind, electron_source_density_ind,
      electron_source_momentum_ind, electron_source_energy_ind, weight_ind;
};

/**
 * @brief A struct defining data and kernel functions needed for an ionisation
 * reaction.
 *
 * @param species_ A vector of Species objects that define the species(plural)
 * that will be acted on by an ionisation reaction.
 */
struct IoniseReactionKernels : public ReactionKernelsBase {

  IoniseReactionKernels(const std::vector<Species> species_)
      : ReactionKernelsBase(), species(species_),
        required_real_props(RequiredProperties<REAL>(
            BASE_IONISATION_KERNEL::required_simple_real_props, species_,
            BASE_IONISATION_KERNEL::required_species_real_props)) {

    auto species_name = this->species[0].get_name();

    this->ionise_reaction_kernels_on_device.velocity_ind =
        this->required_real_props.required_simple_prop_index(velocity);
    this->ionise_reaction_kernels_on_device.electron_density_ind =
        this->required_real_props.required_species_prop_index(species_name,
                                                              density);
    this->ionise_reaction_kernels_on_device.electron_source_density_ind =
        this->required_real_props.required_species_prop_index(species_name,
                                                              source_density);
    this->ionise_reaction_kernels_on_device.electron_source_momentum_ind =
        this->required_real_props.required_species_prop_index(species_name,
                                                              source_momentum);
    this->ionise_reaction_kernels_on_device.electron_source_energy_ind =
        this->required_real_props.required_species_prop_index(species_name,
                                                              source_energy);
    this->ionise_reaction_kernels_on_device.weight_ind =
        this->required_real_props.required_simple_prop_index(weight);
  };

private:
  IoniseReactionKernelsOnDevice ionise_reaction_kernels_on_device;

  std::vector<Species> species;

  RequiredProperties<REAL> required_real_props;

public:
  /**
   * @brief Getters for number of simple real properties, species-dependent real
   * properties and the names of both sets of properties and the SYCL
   * device-specific struct.
   */
  const int get_num_simple_real_props() {
    return this->required_real_props.get_required_simple_props().size();
  }

  std::vector<std::string> get_required_simple_real_props() {
    return this->required_real_props.required_simple_prop_names();
  }

  const int get_num_species_real_props() {
    return this->required_real_props.get_required_species_props().size();
  }

  std::vector<std::string> get_required_species_real_props() {
    return this->required_real_props.required_species_prop_names();
  }

  IoniseReactionKernelsOnDevice get_on_device_obj() {
    return this->ionise_reaction_kernels_on_device;
  }
};
