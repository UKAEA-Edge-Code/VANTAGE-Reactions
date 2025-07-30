#ifndef REACTION_KERNELS_H
#define REACTION_KERNELS_H
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Base reaction kernels object.
 */
struct ReactionKernelsBase {

  /**
   * @brief Constructor for ReactionKernelsBase.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction kernel.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction kernel.
   * @param required_int_props_ephemeral Properties<INT> object containing
   * information regarding the required INT-based ephemeral properties for the
   * reaction kernel.
   * @param required_real_props_ephemeral Properties<REAL> object containing
   * information regarding the required REAL-based properties for the reaction
   * kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions required by a
   * reaction kernel (this in turn matches the number of ReactionData-derived
   * objects that must be passed to the constructor of a DataCalculator object
   * when this kernel and the DataCalculator object are passed to a
   * LinearReactionBase-derived object constructor).
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * retrieving property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props,
                      Properties<INT> required_int_props_ephemeral,
                      Properties<REAL> required_real_props_ephemeral,
                      INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map = get_default_map())
      : required_int_props(required_int_props),
        required_real_props(required_real_props),
        required_int_props_ephemeral(required_int_props_ephemeral),
        required_real_props_ephemeral(required_real_props_ephemeral),
        pre_req_ndims(pre_req_ndims) {
          NESOWARN(
            map_subset_check(properties_map),
            "The provided properties_map does not include all the keys from the default_map (and therefore is not an extension of that map). \
            There may be inconsitencies with indexing of properties."
          );

          this->properties_map = properties_map;
        }

  /**
   * \overload
   * @brief Constructor for ReactionKernelsBase that by default sets no required props.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * retrieving property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionKernelsBase(std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(Properties<INT>(), Properties<REAL>(),
                            Properties<INT>(), Properties<REAL>(), 0,
                            properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionKernelsBase that by default only sets required_int_props.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions required by a
   * reaction kernel (this in turn matches the number of ReactionData-derived
   * objects that must be passed to the constructor of a DataCalculator object
   * when this kernel and the DataCalculator object are passed to a
   * LinearReactionBase-derived object constructor).
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * retrieving property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionKernelsBase(Properties<INT> required_int_props, INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(required_int_props, Properties<REAL>(),
                            Properties<INT>(), Properties<REAL>(),
                            pre_req_ndims, properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionKernelsBase that by default only sets required_real_props.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions required by a
   * reaction kernel (this in turn matches the number of ReactionData-derived
   * objects that must be passed to the constructor of a DataCalculator object
   * when this kernel and the DataCalculator object are passed to a
   * LinearReactionBase-derived object constructor).
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * retrieving property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionKernelsBase(Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(Properties<INT>(), required_real_props,
                            Properties<INT>(), Properties<REAL>(),
                            pre_req_ndims, properties_map) {}
  
  /**
   * \overload
   * @brief Constructor for ReactionKernelsBase that by default only sets required_int_props and required_real_props.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction kernel.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction kernel.
   * @param pre_req_ndims (Optional) Integer defining the number of dimensions required by a
   * reaction kernel (this in turn matches the number of ReactionData-derived
   * objects that must be passed to the constructor of a DataCalculator object
   * when this kernel and the DataCalculator object are passed to a
   * LinearReactionBase-derived object constructor).
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * retrieving property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionKernelsBase(Properties<INT> required_int_props,
                      Properties<REAL> required_real_props,
                      INT pre_req_ndims = 0,
                      std::map<int, std::string> properties_map = get_default_map())
      : ReactionKernelsBase(required_int_props, required_real_props,
                            Properties<INT>(), Properties<REAL>(),
                            pre_req_ndims, properties_map) {}

  /**
   * @brief Return all required integer property names, including ephemeral
   * properties
   *
   */
  std::vector<std::string> get_required_int_props() {
    auto names = this->required_int_props.get_prop_names(this->properties_map);
    auto ephemeral_names =
        this->required_int_props_ephemeral.get_prop_names(this->properties_map);
    names.insert(names.end(), ephemeral_names.begin(), ephemeral_names.end());
    return names;
  }

  /**
   * @brief Return all required real property names, including ephemeral
   * properties
   *
   */
  std::vector<std::string> get_required_real_props() {
    auto names = this->required_real_props.get_prop_names(this->properties_map);
    auto ephemeral_names = this->required_real_props_ephemeral.get_prop_names(
        this->properties_map);
    names.insert(names.end(), ephemeral_names.begin(), ephemeral_names.end());
    return names;
  }

  /**
   * @brief Return names of required ephemeral integer properties
   *
   */
  std::vector<std::string> get_required_int_props_ephemeral() {
    return this->required_int_props_ephemeral.get_prop_names(
        this->properties_map);
  }
  /**
   * @brief Return names of required ephemeral real properties
   *
   */
  std::vector<std::string> get_required_real_props_ephemeral() {
    return this->required_real_props_ephemeral.get_prop_names(
        this->properties_map);
  }

  const Properties<INT> &get_required_descendant_int_props() {
    return this->required_descendant_int_props;
  }

  const Properties<REAL> &get_required_descendant_real_props() {
    return this->required_descendant_real_props;
  }

  std::shared_ptr<ProductMatrixSpec> get_descendant_matrix_spec() {
    return this->descendant_matrix_spec;
  }

  const INT &get_pre_ndims() const { return this->pre_req_ndims; }

protected:
  void set_required_descendant_int_props(
      const Properties<INT> &required_descendant_int_props) {
    this->required_descendant_int_props = required_descendant_int_props;
  }

  void set_required_descendant_real_props(
      const Properties<REAL> &required_descendant_real_props) {
    this->required_descendant_real_props = required_descendant_real_props;
  }

  template <int ndim_velocity = 2, int num_products_per_parent = 0>
  void set_descendant_matrix_spec() {
    if constexpr (num_products_per_parent < 1) {
      return;
    } else {
      NESOWARN(((this->required_descendant_int_props.get_props().size() == 0) &&
                (this->required_descendant_real_props.get_props().size() == 0)),
               "The number of products per parent is >= 1 but no required "
               "descendant properties are set. This will result in an empty "
               "descendant_matrix_spec.")

      auto descendant_particles_spec = ParticleSpec();

      for (auto prop : this->required_descendant_int_props.get_props()) {
        auto descendant_prop =
            ParticleProp<INT>(Sym<INT>(this->properties_map.at(prop)), 1);
        descendant_particles_spec.push(descendant_prop);
      }

      for (auto prop : this->required_descendant_real_props.get_props()) {
        if (prop == default_properties.velocity) {
          auto descendant_prop = ParticleProp<REAL>(
              Sym<REAL>(this->properties_map.at(prop)), ndim_velocity);
          descendant_particles_spec.push(descendant_prop);
        } else {
          auto descendant_prop =
              ParticleProp<REAL>(Sym<REAL>(this->properties_map.at(prop)), 1);
          descendant_particles_spec.push(descendant_prop);
        }
      }

      this->descendant_matrix_spec =
          product_matrix_spec(descendant_particles_spec);
    }
  }

  Properties<INT> required_int_props;
  Properties<REAL> required_real_props;

  Properties<INT> required_int_props_ephemeral;
  Properties<REAL> required_real_props_ephemeral;

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
  void
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
  void
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
  void transformation_kernel(
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
  void weight_kernel(REAL &modified_weight, Access::LoopIndex::Read &index,
                     Access::DescendantProducts::Write &descendant_products,
                     Access::SymVector::Write<INT> &req_int_props,
                     Access::SymVector::Write<REAL> &req_real_props,
                     const std::array<int, num_products_per_parent> &out_states,
                     Access::NDLocalArray::Read<REAL, 2> &pre_req_data,
                     double dt) const {
    return;
  }
};
}; // namespace VANTAGE::Reactions
#endif