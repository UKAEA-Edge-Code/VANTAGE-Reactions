#ifndef REACTIONS_FIXED_COEFFICIENT_DATA_H
#define REACTIONS_FIXED_COEFFICIENT_DATA_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation for a fixed rate coefficient
 * reaction. The reaction rate is calculated as
 * rate_coefficient*particle_weight.
 */
struct FixedCoefficientDataOnDevice : public ReactionDataBaseOnDevice<> {

  FixedCoefficientDataOnDevice() = default;
  /**
   * @brief Constructor for FixedCoefficientDataOnDevice.
   *
   * @param rate REAL-valued rate to be used in reaction rate calculation.
   */
  FixedCoefficientDataOnDevice(REAL rate) : rate(rate) {};

  /**
   * @brief Function to calculate the reaction rate for a fixed rate
   * coefficient reaction
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size 1 containing the calculated reaction
   * rate.
   */
  std::array<REAL, 1>
  calc_data(const SingleReactionDataAccessors &accessors,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    auto weight =
        accessors.req_real_props.at(this->weight_ind, accessors.index, 0);

    return std::array<REAL, 1>{weight * this->rate};
  }

public:
  int weight_ind;
  REAL rate;
};

/**
 * @brief Reaction rate data calculation for a fixed rate coefficient reaction.
 * The reaction rate is calculated as rate_coefficient*particle_weight.
 */
struct FixedCoefficientData
    : public ReactionDataBase<FixedCoefficientDataOnDevice> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 1> required_simple_real_props = {
      props.weight};

  /**
   * @brief Constructor for FixedCoefficientData.
   *
   * @param rate_coeff A real-valued rate coefficient (rate proportional to this
   * and the particle weight)
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names
   */
  FixedCoefficientData(
      REAL rate_coefficient,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<FixedCoefficientDataOnDevice>(
            Properties<REAL>(required_simple_real_props), properties_map) {

    this->on_device_obj = FixedCoefficientDataOnDevice(rate_coefficient);

    this->index_on_device_object();
  }

  /**
   * @brief Index the particle weight on the on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->weight_ind =
        this->argument_pack.required_real_props.find_index(
            this->properties_map.at(props.weight));
  };
};
}; // namespace VANTAGE::Reactions
#endif
