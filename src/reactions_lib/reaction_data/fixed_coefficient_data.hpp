#pragma once
#include "../reaction_data.hpp"
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace Reactions {

struct FixedCoefficientDataOnDevice : public ReactionDataBaseOnDevice<> {
  FixedCoefficientDataOnDevice(REAL rate) : rate(rate){};

  /**
   * @brief Function to calculate the reaction rate for a fixed reaction
   * coefficient reaction
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
  std::array<REAL, 1>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    auto weight = req_real_props.at(this->weight_ind, index, 0);

    return std::array<REAL, 1>{weight * this->rate};
  }

public:
  int weight_ind;
  REAL rate;
};

/**
 * @brief A struct defining the data needed for a fixed rate coefficient
 * reaction. The reaction rate is calculated as
 * rate_coefficient*particle_weight.
 *
 * @param rate_coeff A real-valued rate coefficient (rate proportianl to this
 * and the particle weight)
 * @param properties_map_ A std::map<int, std::string> object to be passed to
 * ReactionDataBase
 */
struct FixedCoefficientData : public ReactionDataBase<> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  FixedCoefficientData(REAL rate_coefficient,
                       std::map<int, std::string> properties_map_ = default_map)
      : ReactionDataBase(Properties<REAL>(required_simple_real_props),
                         properties_map_),
        fixed_coefficient_data_on_device(
            FixedCoefficientDataOnDevice(rate_coefficient)) {

    this->fixed_coefficient_data_on_device.weight_ind =
        this->required_real_props.simple_prop_index(props.weight,
                                                    this->properties_map);
  }

private:
  FixedCoefficientDataOnDevice fixed_coefficient_data_on_device;

public:
  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */

  FixedCoefficientDataOnDevice get_on_device_obj() {
    return this->fixed_coefficient_data_on_device;
  }
};
}; // namespace Reactions
