#pragma once
#include "reaction_kernel_pre_reqs.hpp"
#include <neso_particles.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace NESO::Particles;

/**
 * @brief Base reaction kernels object.
 */

struct ReactionKernelsBase {
  ReactionKernelsBase() = default;

  ReactionKernelsBase(Properties<INT> required_int_props)
      : required_int_props(required_int_props) {}

  ReactionKernelsBase(Properties<REAL> required_real_props)
      : required_real_props(required_real_props) {}

  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props)
      : required_int_props(required_int_props),
        required_real_props(required_real_props) {}
  /**
   * @brief Virtual getters functions that can be overidden by an implementation
   * in a derived struct.
   */
  std::vector<std::string> get_required_int_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names = this->required_int_props.get_prop_names();
    } catch (std::logic_error) {
    }
    return prop_names;
  }

  std::vector<std::string> get_required_real_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names = this->required_real_props.get_prop_names();
    } catch (std::logic_error) {
    }
    return prop_names;
  }

protected:
  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;
};

/**
 * @brief Base reaction kernels object to be used on SYCL devices.
 *
 * @tparam num_products_per_parent The number of products produced per parent
 * by a reaction.
 */
template <int num_products_per_parent> struct ReactionKernelsBaseOnDevice {
  ReactionKernelsBaseOnDevice() = default;

  /**
   * @brief Base scattering kernel for calculating and applying
   * reaction-derived velocity modifications of the particles.
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
  virtual void
  scattering_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                    Access::DescendantProducts::Write &descendant_products,
                    Access::SymVector::Write<INT> &req_int_props,
                    Access::SymVector::Write<REAL> &req_real_props,
                    const std::array<int, num_products_per_parent> &out_states,
                    Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
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
  virtual void
  feedback_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                  Access::DescendantProducts::Write &descendant_products,
                  Access::SymVector::Write<INT> &req_int_props,
                  Access::SymVector::Write<REAL> &req_real_props,
                  const std::array<int, num_products_per_parent> &out_states,
                  Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
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
  virtual void transformation_kernel(
      REAL &modified_weight, Access::LoopIndex::Read &index,
      Access::DescendantProducts::Write &descendant_products,
      Access::SymVector::Write<INT> &req_int_props,
      Access::SymVector::Write<REAL> &req_real_props,
      const std::array<int, num_products_per_parent> &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    return;
  }
  /**
   * @brief Base weight kernel for calculating and applying
   * reaction-derived weight modifications of the particles.
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
  virtual void
  weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                Access::DescendantProducts::Write &descendant_products,
                Access::SymVector::Write<INT> &req_int_props,
                Access::SymVector::Write<REAL> &req_real_props,
                const std::array<int, num_products_per_parent> &out_states,
                Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                double dt) const {
    return;
  }
};
