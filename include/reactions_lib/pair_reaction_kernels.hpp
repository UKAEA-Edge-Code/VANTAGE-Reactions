#ifndef REACTIONS_PAIR_REACTION_KERNELS_H
#define REACTIONS_PAIR_REACTION_KERNELS_H
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Base pair reaction kernels object.
 */
struct PairReactionKernelsBase {

  /**
   * @brief Constructor for PairReactionKernelsBase.
   *
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction kernel.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction kernel.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction kernel.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions
   * required by a reaction kernel (this in turn matches the number of
   * ReactionData-derived objects that must be passed to the constructor of a
   * PairDataCalculator object when this kernel and the PairDataCalculator
   * object are passed to corresponding reaction object).
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionKernelsBase(
      Properties<INT> req_int_props_a, Properties<REAL> req_real_props_a,
      Properties<INT> req_int_props_b, Properties<REAL> req_real_props_b,
      INT pre_req_ndims = 0,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for PairReactionKernelsBase that by default sets no
   * required props.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionKernelsBase(
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for PairReactionKernelsBase that by default only sets
   * required_int_props and sets them the same for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions
   * required by a reaction kernel (this in turn matches the number of
   * ReactionData-derived objects that must be passed to the constructor of a
   * PairDataCalculator object when this kernel and the PairDataCalculator
   * object are passed to corresponding reaction object).
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionKernelsBase(
      Properties<INT> required_int_props, INT pre_req_ndims = 0,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for PairReactionKernelsBase that by default only sets
   * required_real_props and sets them the same for both particles.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions
   * required by a reaction kernel (this in turn matches the number of
   * ReactionData-derived objects that must be passed to the constructor of a
   * PairDataCalculator object when this kernel and the PairDataCalculator
   * object are passed to corresponding reaction object).
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionKernelsBase(
      Properties<REAL> required_real_props, INT pre_req_ndims = 0,
      std::map<int, std::string> properties_map = get_default_map());

  /**
   * \overload
   * @brief Constructor for PairReactionKernelsBase that sets the same required
   * properties for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction kernel.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions
   * required by a reaction kernel (this in turn matches the number of
   * ReactionData-derived objects that must be passed to the constructor of a
   * PairDataCalculator object when this kernel and the PairDataCalculator
   * object are passed to corresponding reaction object).
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionKernelsBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      INT pre_req_ndims = 0,
      std::map<int, std::string> properties_map = get_default_map());

  virtual ~PairReactionKernelsBase() = default;

  /**
   * @brief Return all required integer property names for the first particle
   *
   */
  std::vector<std::string> get_required_int_props_a();

  /**
   * @brief Return all required integer property names for the second particle
   *
   */
  std::vector<std::string> get_required_int_props_b();
  /**
   * @brief Return all required real property names for the first particle
   *
   */
  std::vector<std::string> get_required_real_props_a();

  /**
   * @brief Return all required real property names for the second particle
   *
   */
  std::vector<std::string> get_required_real_props_b();

  const Properties<INT> &get_required_descendant_int_props_a();

  const Properties<REAL> &get_required_descendant_real_props_a();

  const Properties<INT> &get_required_descendant_int_props_b();

  const Properties<REAL> &get_required_descendant_real_props_b();
  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_a();

  std::shared_ptr<NP::ProductMatrixSpec> get_descendant_matrix_spec_b();
  const INT &get_pre_ndims() const;
  const INT &get_num_products_a() const;
  const INT &get_num_products_b() const;

protected:
  void set_required_descendant_int_props_a(
      const Properties<INT> &required_descendant_int_props);

  void set_required_descendant_real_props_a(
      const Properties<REAL> &required_descendant_real_props);

  void set_required_descendant_int_props_b(
      const Properties<INT> &required_descendant_int_props);

  void set_required_descendant_real_props_b(
      const Properties<REAL> &required_descendant_real_props);

  template <int ndim_velocity = 2, int num_products_per_parent = 0>
  void set_descendant_matrix_spec_a() {
    this->num_products_a = num_products_per_parent;
    if constexpr (num_products_per_parent < 1) {
      return;
    } else {
      NESOWARN(
          ((this->required_descendant_int_props_a.get_props().size() == 0) &&
           (this->required_descendant_real_props_a.get_props().size() == 0)),
          "The number of products per parent is >= 1 but no required "
          "descendant properties are set. This will result in an empty "
          "descendant_matrix_spec.")

      auto descendant_particles_spec = NP::ParticleSpec();

      for (auto prop : this->required_descendant_int_props_a.get_props()) {
        auto descendant_prop = NP::ParticleProp<INT>(
            NP::Sym<INT>(this->properties_map.at(prop)), 1);
        descendant_particles_spec.push(descendant_prop);
      }

      for (auto prop : this->required_descendant_real_props_a.get_props()) {
        if (prop == default_properties.velocity) {
          auto descendant_prop = NP::ParticleProp<REAL>(
              NP::Sym<REAL>(this->properties_map.at(prop)), ndim_velocity);
          descendant_particles_spec.push(descendant_prop);
        } else {
          auto descendant_prop = NP::ParticleProp<REAL>(
              NP::Sym<REAL>(this->properties_map.at(prop)), 1);
          descendant_particles_spec.push(descendant_prop);
        }
      }

      this->descendant_matrix_spec_a =
          NP::product_matrix_spec(descendant_particles_spec);
    }
  }

  template <int ndim_velocity = 2, int num_products_per_parent = 0>
  void set_descendant_matrix_spec_b() {
    this->num_products_b = num_products_per_parent;
    if constexpr (num_products_per_parent < 1) {
      return;
    } else {
      NESOWARN(
          ((this->required_descendant_int_props_b.get_props().size() == 0) &&
           (this->required_descendant_real_props_b.get_props().size() == 0)),
          "The number of products per parent is >= 1 but no required "
          "descendant properties are set. This will result in an empty "
          "descendant_matrix_spec.")

      auto descendant_particles_spec = NP::ParticleSpec();

      for (auto prop : this->required_descendant_int_props_b.get_props()) {
        auto descendant_prop = NP::ParticleProp<INT>(
            NP::Sym<INT>(this->properties_map.at(prop)), 1);
        descendant_particles_spec.push(descendant_prop);
      }

      for (auto prop : this->required_descendant_real_props_b.get_props()) {
        if (prop == default_properties.velocity) {
          auto descendant_prop = NP::ParticleProp<REAL>(
              NP::Sym<REAL>(this->properties_map.at(prop)), ndim_velocity);
          descendant_particles_spec.push(descendant_prop);
        } else {
          auto descendant_prop = NP::ParticleProp<REAL>(
              NP::Sym<REAL>(this->properties_map.at(prop)), 1);
          descendant_particles_spec.push(descendant_prop);
        }
      }

      this->descendant_matrix_spec_b =
          NP::product_matrix_spec(descendant_particles_spec);
    }
  }
  Properties<INT> required_int_props_a;
  Properties<REAL> required_real_props_a;
  Properties<INT> required_int_props_b;
  Properties<REAL> required_real_props_b;

  Properties<INT> required_descendant_int_props_a;
  Properties<REAL> required_descendant_real_props_a;

  Properties<INT> required_descendant_int_props_b;
  Properties<REAL> required_descendant_real_props_b;

  std::shared_ptr<NP::ProductMatrixSpec> descendant_matrix_spec_a =
      std::make_shared<NP::ProductMatrixSpec>();

  std::shared_ptr<NP::ProductMatrixSpec> descendant_matrix_spec_b =
      std::make_shared<NP::ProductMatrixSpec>();

  INT pre_req_ndims;
  INT num_products_a;
  INT num_products_b;

  std::map<int, std::string> properties_map;
};

/**
 * @brief Base pair reaction kernels object to be used on SYCL devices.
 *
 * @tparam num_products_per_parent The number of products produced per parent
 * by a reaction.
 */
template <int num_products_per_parent> struct PairReactionKernelsBaseOnDevice {
  PairReactionKernelsBaseOnDevice() = default;

  /**
   * @brief Parent setting kernel determining which product inherits from which
   * parent. Should call .set_parent on the corresponding descendant products.
   *
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param descendant_products_a Write accessor to descendant products of the
   * first particle
   * @param descendant_products_b Write accessor to descendant products of the
   * second particle
   * @param out_states Array defining the IDs of descendant particles
   */
  void parent_kernel(
      NP::Access::PairLoopIndex::Read &index_a,
      NP::Access::PairLoopIndex::Read &index_b,
      NP::Access::DescendantProducts::Write &descendant_products_a,
      NP::Access::DescendantProducts::Write &descendant_products_b,
      const std::array<int, num_products_per_parent> &out_states) const {
    return;
  }

  /**
   * @brief Base scattering kernel for calculating and applying
   * reaction-derived velocity modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * NP::ParticlePairLoop inside which apply is called. NP::Access using
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
  void scattering_kernel(
      REAL &modified_weight, NP::Access::PairLoopIndex::Read &index_a,
      NP::Access::PairLoopIndex::Read &index_b,
      NP::Access::PairLoopIndex::Read &pair_index,
      NP::Access::DescendantProducts::Write &descendant_products_a,
      NP::Access::DescendantProducts::Write &descendant_products_b,
      NP::Access::SymVector::Write<INT> &req_int_props_a,
      NP::Access::SymVector::Write<REAL> &req_real_props_a,
      NP::Access::SymVector::Write<INT> &req_int_props_b,
      NP::Access::SymVector::Write<REAL> &req_real_props_b,
      const std::array<int, num_products_per_parent> &out_states,
      NP::Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    return;
  }
  /**
   * @brief Base feedback kernel for calculating and applying
   * background field modifications from the reaction.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * NP::ParticlePairLoop inside which apply is called. NP::Access using
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
  void feedback_kernel(
      REAL &modified_weight, NP::Access::PairLoopIndex::Read &index_a,
      NP::Access::PairLoopIndex::Read &index_b,
      NP::Access::PairLoopIndex::Read &pair_index,
      NP::Access::DescendantProducts::Write &descendant_products_a,
      NP::Access::DescendantProducts::Write &descendant_products_b,
      NP::Access::SymVector::Write<INT> &req_int_props_a,
      NP::Access::SymVector::Write<REAL> &req_real_props_a,
      NP::Access::SymVector::Write<INT> &req_int_props_b,
      NP::Access::SymVector::Write<REAL> &req_real_props_b,
      const std::array<int, num_products_per_parent> &out_states,
      NP::Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    return;
  }

  /**
   * @brief Base transformation kernel for calculating and applying
   * reaction-derived ID modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * NP::ParticlePairLoop inside which apply is called. NP::Access using
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
      REAL &modified_weight, NP::Access::PairLoopIndex::Read &index_a,
      NP::Access::PairLoopIndex::Read &index_b,
      NP::Access::PairLoopIndex::Read &pair_index,
      NP::Access::DescendantProducts::Write &descendant_products_a,
      NP::Access::DescendantProducts::Write &descendant_products_b,
      NP::Access::SymVector::Write<INT> &req_int_props_a,
      NP::Access::SymVector::Write<REAL> &req_real_props_a,
      NP::Access::SymVector::Write<INT> &req_int_props_b,
      NP::Access::SymVector::Write<REAL> &req_real_props_b,
      const std::array<int, num_products_per_parent> &out_states,
      NP::Access::NDLocalArray::Read<REAL, 2> &pre_req_data, double dt) const {
    return;
  }
  /**
   * @brief Base weight kernel for calculating and applying
   * reaction-derived weight modifications of the particles.
   *
   * @param modified_weight The weight modification needed for calculating
   * the changes to the background fields.
   * @param index_a Read-only accessor to a loop index for the first particle
   * @param index_b Read-only accessor to a loop index for the second particle
   * @param pair_index Read-only accessor to a pair loop index for a
   * NP::ParticlePairLoop inside which apply is called. NP::Access using
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
  void
  weight_kernel(REAL &modified_weight, NP::Access::PairLoopIndex::Read &index_a,
                NP::Access::PairLoopIndex::Read &index_b,
                NP::Access::PairLoopIndex::Read &pair_index,
                NP::Access::DescendantProducts::Write &descendant_products_a,
                NP::Access::DescendantProducts::Write &descendant_products_b,
                NP::Access::SymVector::Write<INT> &req_int_props_a,
                NP::Access::SymVector::Write<REAL> &req_real_props_a,
                NP::Access::SymVector::Write<INT> &req_int_props_b,
                NP::Access::SymVector::Write<REAL> &req_real_props_b,
                const std::array<int, num_products_per_parent> &out_states,
                NP::Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                double dt) const {
    return;
  }
};
}; // namespace VANTAGE::Reactions
#endif
