#pragma once
#include "../reaction_data.hpp"
#include "reactions_lib/reaction_kernels/specular_reflection_kernels.hpp"
#include <array>
#include <memory>
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>

using namespace NESO::Particles;
namespace Reactions {

/**
 * @brief Device reaction data returning an array based on lookup table for
 * integer-valued key
 *
 * @tparam N The size of the REAL-valued array stored in the lookup table
 * @tparam ephemeral_dat True if the Sym storing the key value is an ephemeral
 * dat
 * @param key_comp The key dat component index to use as the lookup key
 * @param default_data The default array to be returned in case the lookup key
 * cannot be found
 */
template <size_t N, bool ephemeral_dat>
struct ArrayLookupDataOnDevice : public ReactionDataBaseOnDevice<N> {
  ArrayLookupDataOnDevice(const int &key_comp,
                          const std::array<REAL, N> &default_data)
      : key_comp(key_comp), default_data(default_data){};

  /**
   * @brief Function to calculate the reaction rate for a fixed rate reaction
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   */
  std::array<REAL, N>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<N>::RNG_KERNEL_TYPE::KernelType
                &kernel) const {

    int key_val;

    if constexpr (ephemeral_dat) {

      key_val = req_int_props.at_ephemeral(0, index, this->key_comp);
    } else {

      key_val = req_int_props.at(0, index, this->key_comp);
    }

    const std::array<REAL, N> *val_ptr = nullptr;

    const bool exists = this->lut_root->get(key_val, &val_ptr);

    if (not exists) {
      val_ptr = &this->default_data;
    }

    return *val_ptr;
  }

private:
  std::array<REAL, N> default_data;
  INT key_comp;

public:
  BlockedBinaryNode<int, std::array<REAL, N>, 8> *lut_root;
};

/**
 * @brief Hos reaction data returning an array based on lookup table for
 * integer-valued key
 *
 * @tparam N The size of the REAL-valued array stored in the lookup table
 * @tparam ephemeral_dat True if the Sym storing the key value is an ephemeral
 * dat
 * @param key_comp The key dat component index to use as the lookup key
 * @param default_data The default array to be returned in case the lookup key
 * cannot be found
 */
template <size_t N, bool ephemeral_dat = false>
struct ArrayLookupData : public ReactionDataBase<N> {

  ArrayLookupData(const Sym<INT> &key_sym, int key_sym_comp,
                  const std::map<int, std::array<REAL, N>> &lookup_table,
                  const std::array<REAL, N> &default_values,
                  SYCLTargetSharedPtr sycl_target)
      : ReactionDataBase<N>(), key_sym(key_sym),
        array_lookup_data_on_device(ArrayLookupDataOnDevice<N, ephemeral_dat>(
            key_sym_comp, default_values)) {

    this->lut =
        std::make_shared<BlockedBinaryTree<int, std::array<REAL, N>, 8>>(
            sycl_target);

    for (const auto &[key, value] : lookup_table)
      this->lut->add(key, value);

    this->array_lookup_data_on_device.lut_root = lut->root;
  }

private:
  Sym<INT> key_sym;
  ArrayLookupDataOnDevice<N, ephemeral_dat> array_lookup_data_on_device;
  std::shared_ptr<BlockedBinaryTree<int, std::array<REAL, N>, 8>> lut;

public:
  /**
   * @brief Getter for the SYCL device-specific struct.
   */
  ArrayLookupDataOnDevice<N, ephemeral_dat> get_on_device_obj() {
    return this->array_lookup_data_on_device;
  }

  std::vector<std::string> get_required_int_props() {
    return std::vector<std::string>{this->key_sym.name};
  }

  std::vector<std::string> get_required_int_props_ephemeral() {
    if constexpr (ephemeral_dat) {
      return std::vector<std::string>{this->key_sym.name};
    } else {
      return this->required_int_props_ephemeral.get_prop_names(
          this->properties_map);
    }
  }
};
}; // namespace Reactions
