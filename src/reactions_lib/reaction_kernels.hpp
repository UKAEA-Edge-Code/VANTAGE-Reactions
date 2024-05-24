#pragma once
#include <neso_particles.hpp>
#include <string>
#include <type_traits>

using namespace NESO::Particles;

/**
 * @brief Base reaction kernels object.
 *
 * @tparam num_products_per_parent The number of products produced per parent
 * by a reaction.
 */

// template list num_products, req_properties (it's own class with getters),
// species (also it's own class)

struct ReactionKernelsBase {
  ReactionKernelsBase() = default;

  virtual const int get_num_simple_int_props() { return 0; }
  virtual const int get_num_simple_real_props() { return 0; }

  virtual const int get_num_species_int_props() { return 0; }
  virtual const int get_num_species_real_props() { return 0; }

  virtual std::vector<std::string> get_required_simple_int_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
  virtual std::vector<std::string> get_required_simple_real_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }

  virtual std::vector<std::string> get_required_species_int_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
  virtual std::vector<std::string> get_required_species_real_props() {
    std::vector<std::string> required_prop_names = {};
    return required_prop_names;
  }
};

template <int num_products_per_parent> struct ReactionKernelsBaseOnDevice {
  ReactionKernelsBaseOnDevice() = default;

  /**
   * @brief Base scattering kernel for calculating and applying
   * reaction-derived velocity modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to operated on
   * @param read_req_ints Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param read_req_reals Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param write_req_ints Symbol indices for integer-valued
   * ParticleDats that need to be modified
   * @param write_req_reals Symbol indices for real-valued
   * ParticleDats that need to be modified
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  virtual void
  scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Write<INT> &req_simple_prop_ints,
                    Access::SymVector::Write<REAL> &req_simple_prop_reals,
                    Access::SymVector::Write<INT> &req_species_prop_ints,
                    Access::SymVector::Write<REAL> &req_species_prop_reals,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::LocalArray::Read<REAL> &pre_req_data,
                    double dt) const {
    return;
  }
  /**
   * @brief Base feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to operated on
   * @param read_req_ints Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param read_req_reals Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param write_req_ints Symbol indices for integer-valued
   * ParticleDats that need to be modified
   * @param write_req_reals Symbol indices for real-valued
   * ParticleDats that need to be modified
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  virtual void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_simple_prop_ints,
                  Access::SymVector::Write<REAL> &req_simple_prop_reals,
                  Access::SymVector::Write<INT> &req_species_prop_ints,
                  Access::SymVector::Write<REAL> &req_species_prop_reals,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::LocalArray::Read<REAL> &pre_req_data,
                  double dt) const {
    return;
  }
  /**
   * @brief Base transformation kernel for calculating and applying
   * reaction-derived ID modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to operated on
   * @param read_req_ints Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param read_req_reals Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param write_req_ints Symbol indices for integer-valued
   * ParticleDats that need to be modified
   * @param write_req_reals Symbol indices for real-valued
   * ParticleDats that need to be modified
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  virtual void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_simple_prop_ints,
      Access::SymVector::Write<REAL> &req_simple_prop_reals,
      Access::SymVector::Write<INT> &req_species_prop_ints,
      Access::SymVector::Write<REAL> &req_species_prop_reals,
      const std::array<int, num_products_per_parent> &out_states,
      Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
    return;
  }
  /**
   * @brief Base weight kernel for calculating and applying
   * reaction-derived weight modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_rate is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param descendant_products Write accessor to descendant products
   * that may need to operated on
   * @param read_req_ints Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param read_req_reals Symbol indices for integer-valued ParticleDats that
   * need to be read for operations inside the kernel
   * @param write_req_ints Symbol indices for integer-valued
   * ParticleDats that need to be modified
   * @param write_req_reals Symbol indices for real-valued
   * ParticleDats that need to be modified
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  virtual void
  weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                Access::DescendantProducts::Write &descendant_products,
                Access::SymVector::Write<INT> &req_simple_prop_ints,
                Access::SymVector::Write<REAL> &req_simple_prop_reals,
                Access::SymVector::Write<INT> &req_species_prop_ints,
                Access::SymVector::Write<REAL> &req_species_prop_reals,
                const std::array<int, num_products_per_parent> &out_states,
                Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
    return;
  }
};
