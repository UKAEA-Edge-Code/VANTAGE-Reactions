#ifndef REACTIONS_SAMPLER_DATA_H
#define REACTIONS_SAMPLER_DATA_H
#include "../particle_properties_map.hpp"
#include "../utils.hpp"
#include <iostream>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: ReactionData sampling a single random number from a random
 * kernel
 *
 * @tparam RNG_KERNEL The RNG kernel type
 */
template <typename RNG_KERNEL>
struct SamplerDataOnDevice : public ReactionDataBaseOnDevice<1, RNG_KERNEL> {

  SamplerDataOnDevice() = default;

  /**
   * @brief Sample one number from the rng_kernel
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param rng_kernel The random number generator kernel to sample from
   *
   * @return Sampled random number
   */
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename RNG_KERNEL::KernelType &rng_kernel) const {
    bool is_kernel_valid = true;
    auto rand = rng_kernel.at(index, 0, &is_kernel_valid);
    if (!is_kernel_valid) {
      req_int_props.at(this->panic_ind, index, 0) += 1;
    }
    return std::array<REAL, 1>{rand};
  }

public:
  int panic_ind;
};

/**
 * @brief On host reaction data class for sampling one number from an rng_kernel
 * (to be used in pipelines)
 *
 * @tparam RNG_KERNEL The RNG kernel type
 */
template <typename RNG_KERNEL>
struct SamplerData
    : public ReactionDataBase<SamplerDataOnDevice<RNG_KERNEL>, 1, RNG_KERNEL> {

  constexpr static auto props = default_properties;

  constexpr static auto required_simple_int_props =
      std::array<int, 1>{props.panic};

  /**
   * @brief Constructor for SamplerData.
   *
   * @param rng_kernel Shared pointer to kernel to be used
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  SamplerData(std::shared_ptr<RNG_KERNEL> rng_kernel,
              std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<SamplerDataOnDevice<RNG_KERNEL>, 1, RNG_KERNEL>(
            Properties<INT>(required_simple_int_props), properties_map) {
    this->on_device_obj = SamplerDataOnDevice<RNG_KERNEL>();

    this->set_rng_kernel(rng_kernel);
    this->index_on_device_object();
  }

  /**
   * @brief Index the panic flag on the on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->panic_ind = this->required_int_props.find_index(
        this->properties_map.at(props.panic));
  };
};
}; // namespace VANTAGE::Reactions
#endif
