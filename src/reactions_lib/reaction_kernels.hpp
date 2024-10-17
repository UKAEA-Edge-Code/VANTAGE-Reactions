#pragma once
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <neso_particles.hpp>
#include <neso_particles/containers/product_matrix.hpp>

using namespace NESO::Particles;
namespace Reactions {

/**
 * @brief Base reaction kernels object.
 */
struct ReactionKernelsBase {
  ReactionKernelsBase(std::map<int, std::string> properties_map_ = default_map)
      : properties_map(properties_map_) {}

  ReactionKernelsBase(Properties<INT> required_int_props, INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map_ = default_map)
      : required_int_props(required_int_props), pre_req_ndims(pre_req_ndims),
        properties_map(properties_map_) {}

  ReactionKernelsBase(Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map_ = default_map)
      : required_real_props(required_real_props), pre_req_ndims(pre_req_ndims),
        properties_map(properties_map_) {}

  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map_ = default_map)
      : required_int_props(required_int_props),
        required_real_props(required_real_props), pre_req_ndims(pre_req_ndims),
        properties_map(properties_map_) {}
  /**
   * @brief Virtual getters functions that can be overidden by an implementation
   * in a derived struct.
   */
  std::vector<std::string> get_required_int_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names =
          this->required_int_props.get_prop_names(this->properties_map);
    } catch (std::logic_error) {
    }
    return prop_names;
  }

  std::vector<std::string> get_required_real_props() {
    std::vector<std::string> prop_names;
    try {
      prop_names =
          this->required_real_props.get_prop_names(this->properties_map);
    } catch (std::logic_error) {
    }
    return prop_names;
  }

  const Properties<INT> &get_required_descendant_int_props() {
    return this->required_descendant_int_props;
  }

  void set_required_descendant_int_props(
      const Properties<INT> &required_descendant_int_props) {
    this->required_descendant_int_props = required_descendant_int_props;
  }

  const Properties<REAL> &get_required_descendant_real_props() {
    return this->required_descendant_real_props;
  }

  void set_required_descendant_real_props(
      const Properties<REAL> &required_descendant_real_props) {
    this->required_descendant_real_props = required_descendant_real_props;
  }

  std::shared_ptr<ProductMatrixSpec> get_descendant_matrix_spec() {
    return this->descendant_matrix_spec;
  }

  void set_descendant_matrix_spec(
      std::shared_ptr<ProductMatrixSpec> descendant_matrix_spec) {
    this->descendant_matrix_spec = descendant_matrix_spec;
  }

  const INT &get_pre_ndims() const { return this->pre_req_ndims; }

protected:
  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;

  Properties<INT> required_descendant_int_props;
  Properties<REAL> required_descendant_real_props;

  std::shared_ptr<ProductMatrixSpec> descendant_matrix_spec =
      std::make_shared<ProductMatrixSpec>();

  INT pre_req_ndims;

  std::map<int, std::string> properties_map;
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
}; // namespace Reactions
