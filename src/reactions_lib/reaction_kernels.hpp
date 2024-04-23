#pragma once
#include <neso_particles.hpp>
#include <particle_properties_map.hpp>
#include <type_traits>

using namespace NESO::Particles;

/**
 * @brief Base reaction kernels object.
 *
 * @tparam num_products_per_parent The number of products produced per parent
 * by a reaction.
 */

template <int num_products_per_parent>

struct ReactionKernelsBase {
  ReactionKernelsBase() = default;

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
                    Access::SymVector::Read<INT> &read_req_ints,
                    Access::SymVector::Read<REAL> &read_req_reals,
                    Access::SymVector::Write<INT> &write_req_ints,
                    Access::SymVector::Write<REAL> &write_req_reals,
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
                  Access::SymVector::Read<INT> &read_req_ints,
                  Access::SymVector::Read<REAL> &read_req_reals,
                  Access::SymVector::Write<INT> &write_req_ints,
                  Access::SymVector::Write<REAL> &write_req_reals,
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
      Access::SymVector::Read<INT> &read_req_ints,
      Access::SymVector::Read<REAL> &read_req_reals,
      Access::SymVector::Write<INT> &write_req_ints,
      Access::SymVector::Write<REAL> &write_req_reals,
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
                Access::SymVector::Read<INT> &read_req_ints,
                Access::SymVector::Read<REAL> &read_req_reals,
                Access::SymVector::Write<INT> &write_req_ints,
                Access::SymVector::Write<REAL> &write_req_reals,
                const std::array<int, num_products_per_parent> &out_states,
                Access::LocalArray::Read<REAL> &pre_req_data, double dt) const {
    return;
  }

  template <typename PROP_TYPE>
  std::vector<Sym<PROP_TYPE>> build_sym_vector(ParticleSpec particle_spec, const char** required_properties, const int num_props) {

    std::vector<Sym<PROP_TYPE>> syms = {};

    for (int iprop = 0; iprop < num_props; iprop++) {
      auto req_prop = required_properties[iprop];
      std::vector<const char *> possible_names;
      try {
        possible_names = ParticlePropertiesIndices::default_map.at(req_prop);
      } catch (std::out_of_range) {
        std::cout << "No instances of " << req_prop
                  << " found in keys of default_map..." << std::endl;
      }
      for (auto &possible_name : possible_names) {
        if constexpr (std::is_same_v<PROP_TYPE, INT>) {
        for (auto &int_prop : particle_spec.properties_int) {
          if (strcmp(int_prop.name.c_str(), possible_name) == 0) {
            syms.push_back(Sym<INT>(int_prop.name));
          }
        }}
        else if constexpr (std::is_same_v<PROP_TYPE, REAL>) {
        for (auto &real_prop : particle_spec.properties_real) {
          if (strcmp(real_prop.name.c_str(), possible_name) == 0) {
            syms.push_back(Sym<REAL>(real_prop.name));
          }
        }}
      }
    }

    return syms;
  }

  virtual const int get_num_props() { return 0; }

  virtual const char **get_required_properties() {
    static const char *required_prop_names[] = {};
    return required_prop_names;
  }
};