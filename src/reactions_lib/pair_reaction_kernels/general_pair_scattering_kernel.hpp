#ifndef REACTIONS_PAIR_SCATTERING_KERNELS_H
#define REACTIONS_PAIR_SCATTERING_KERNELS_H
#include "../pair_reaction_kernels.hpp"
#include <array>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Device type for general pair scattering kernels
 *
 * @tparam ndim_velocity The number of dimensions for the particle velocity
 * property.
 */
template <int ndim_velocity>
struct PairScatteringKernelsOnDevice
    : public PairReactionKernelsBaseOnDevice<2> {

  PairScatteringKernelsOnDevice() = default;

  /**
   * @brief General pair scattering parent kernel. Sets the first products
   * parent to the A particle and the second to the B particle
   *
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param out_states Array defining the IDs of descendant particles
   */
  void parent_kernel(Access::PairLoopIndex::Read &index_a,
                     Access::PairLoopIndex::Read &index_b,
                     Access::DescendantProducts::Write &descendant_products_a,
                     Access::DescendantProducts::Write &descendant_products_b,
                     const std::array<int, 2> &out_states) const {

    descendant_products_a.set_parent(index_a, 0);
    descendant_products_b.set_parent(index_b, 0);
  }
  /**
   * @brief General pair scattering kernel - assumes that pre_req_data stores
   * the 2*ndim_velocity components for the two scattering particles
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * ParticlePairLoop inside which apply is called. Access using
   * pair_index.get_loop_linear_index().
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction kernel.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction kernel.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction kernel.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  void
  scattering_kernel(REAL &modified_weight, Access::PairLoopIndex::Read &index_a,
                    Access::PairLoopIndex::Read &index_b,
                    Access::PairLoopIndex::Read &pair_index,
                    Access::DescendantProducts::Write &descendant_products_a,
                    Access::DescendantProducts::Write &descendant_products_b,
                    Access::SymVector::Write<INT> &req_int_props_a,
                    Access::SymVector::Write<REAL> &req_real_props_a,
                    Access::SymVector::Write<INT> &req_int_props_b,
                    Access::SymVector::Write<REAL> &req_real_props_b,
                    const std::array<int, 2> &out_states,
                    Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                    double dt) const {
    for (int dimx = 0; dimx < ndim_velocity; dimx++) {
      descendant_products_a.at_real(index_a, 0, descendant_velocity_ind, dimx) =
          pre_req_data.at(pair_index.get_loop_linear_index(), dimx);
      descendant_products_b.at_real(index_b, 0, descendant_velocity_ind, dimx) =
          pre_req_data.at(pair_index.get_loop_linear_index(),
                          ndim_velocity + dimx);
    }
  }

  /**
   * @brief General pair scattering weight kernel - simply sets the product's
   * weight to the weight change due to the reaction
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * ParticlePairLoop inside which apply is called. Access using
   * pair_index.get_loop_linear_index().
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction kernel.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction kernel.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction kernel.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   */
  void weight_kernel(REAL &modified_weight,
                     Access::PairLoopIndex::Read &index_a,
                     Access::PairLoopIndex::Read &index_b,
                     Access::PairLoopIndex::Read &pair_index,
                     Access::DescendantProducts::Write &descendant_products_a,
                     Access::DescendantProducts::Write &descendant_products_b,
                     Access::SymVector::Write<INT> &req_int_props_a,
                     Access::SymVector::Write<REAL> &req_real_props_a,
                     Access::SymVector::Write<INT> &req_int_props_b,
                     Access::SymVector::Write<REAL> &req_real_props_b,
                     const std::array<int, 2> &out_states,
                     Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                     double dt) const {
    descendant_products_a.at_real(index_a, 0, descendant_weight_ind, 0) =
        modified_weight;
    descendant_products_b.at_real(index_b, 0, descendant_weight_ind, 0) =
        modified_weight;
  }

  /**
   * @brief General pair scattering transformation kernel
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * ParticlePairLoop inside which apply is called. Access using
   * pair_index.get_loop_linear_index().
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction kernel.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction kernel.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction kernel.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   *
   */
  void transformation_kernel(
      REAL &modified_weight, Access::PairLoopIndex::Read &index_a,
      Access::PairLoopIndex::Read &index_b,
      Access::PairLoopIndex::Read &pair_index,
      Access::DescendantProducts::Write &descendant_products_a,
      Access::DescendantProducts::Write &descendant_products_b,
      Access::SymVector::Write<INT> &req_int_props_a,
      Access::SymVector::Write<REAL> &req_real_props_a,
      Access::SymVector::Write<INT> &req_int_props_b,
      Access::SymVector::Write<REAL> &req_real_props_b,
      const std::array<int, 2> &out_states,
      Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    descendant_products_a.at_int(index_a, 0, descendant_internal_state_ind, 0) =
        out_states[0];
    descendant_products_b.at_int(index_b, 0, descendant_internal_state_ind, 0) =
        out_states[1];
  }

  /**
   * @brief General pair scattering feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * ParticlePairLoop inside which apply is called. Access using
   * pair_index.get_loop_linear_index().
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction kernel.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction kernel.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction kernel.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction kernel.
   * @param out_states Array defining the IDs of descendant particles
   * @param pre_req_data Real-valued local array containing pre-requisite
   * data relating to a derived reaction.
   * @param dt The current time step size.
   *
   */
  void feedback_kernel(REAL &modified_weight,
                       Access::PairLoopIndex::Read &index_a,
                       Access::PairLoopIndex::Read &index_b,
                       Access::PairLoopIndex::Read &pair_index,
                       Access::DescendantProducts::Write &descendant_products_a,
                       Access::DescendantProducts::Write &descendant_products_b,
                       Access::SymVector::Write<INT> &req_int_props_a,
                       Access::SymVector::Write<REAL> &req_real_props_a,
                       Access::SymVector::Write<INT> &req_int_props_b,
                       Access::SymVector::Write<REAL> &req_real_props_b,
                       const std::array<int, 2> &out_states,
                       Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                       double dt) const {

    req_real_props_a.at(this->weight_ind, 0) -= modified_weight;
    req_real_props_b.at(this->weight_ind, 0) -= modified_weight;
  }

public:
  INT weight_ind;
  INT descendant_internal_state_ind, descendant_velocity_ind,
      descendant_weight_ind;
};

/**
 * @brief Host type for general pair scattering kernels - general kernels with
 * post-collision velocities defined by data calculator outputs
 *
 * @tparam ndim_velocity Optional number of dimensions for the particle velocity
 * property (default value of 2)
 */
template <int ndim_velocity = 2>
struct PairScatteringKernels : public PairReactionKernelsBase {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};
  constexpr static std::array<int, 1> required_simple_int_props = {
      props.weight};
  constexpr static std::array<int, 1> required_descendant_simple_int_props = {
      props.internal_state};
  constexpr static std::array<int, 2> required_descendant_simple_real_props = {
      props.velocity, props.weight};
  /**
   * @brief Constructor for PairScatteringKernels.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  PairScatteringKernels(
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionKernelsBase(Properties<REAL>(required_simple_real_props),
                                2 * ndim_velocity, properties_map) {

    this->pair_scattering_kernels_on_device.weight_ind =
        this->required_real_props_a.simple_prop_index(props.weight,
                                                      this->properties_map);

    this->set_required_descendant_int_props_a(
        Properties<INT>(required_descendant_simple_int_props));

    this->set_required_descendant_real_props_a(
        Properties<REAL>(required_descendant_simple_real_props));

    this->set_required_descendant_int_props_b(
        Properties<INT>(required_descendant_simple_int_props));

    this->set_required_descendant_real_props_b(
        Properties<REAL>(required_descendant_simple_real_props));

    this->pair_scattering_kernels_on_device.descendant_internal_state_ind =
        this->required_descendant_int_props_a.simple_prop_index(
            props.internal_state, this->properties_map);
    this->pair_scattering_kernels_on_device.descendant_velocity_ind =
        this->required_descendant_real_props_a.simple_prop_index(
            props.velocity, this->properties_map);
    this->pair_scattering_kernels_on_device.descendant_weight_ind =
        this->required_descendant_real_props_a.simple_prop_index(
            props.weight, this->properties_map);

    this->set_descendant_matrix_spec_a<ndim_velocity, 1>();
    this->set_descendant_matrix_spec_b<ndim_velocity, 1>();
  };

private:
  PairScatteringKernelsOnDevice<ndim_velocity>
      pair_scattering_kernels_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */

  PairScatteringKernelsOnDevice<ndim_velocity> get_on_device_obj() {
    return this->pair_scattering_kernels_on_device;
  }
};
}; // namespace VANTAGE::Reactions
#endif
