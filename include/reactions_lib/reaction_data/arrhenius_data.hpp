#ifndef REACTIONS_ARRHENIUS_DATA_H
#define REACTIONS_ARRHENIUS_DATA_H
#include "../reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

#include <vector>

namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction rate data calculation for an Arrhenius rate
 * coefficient reaction. The reaction rate is calculated as a*temparture**b *
 * particle_weight.
 */
struct ArrheniusDataOnDevice : public ReactionDataBaseOnDevice<> {

  ArrheniusDataOnDevice() = default;
  /**
   * @brief Constructor for ArrheniusDataOnDevice.
   *
   * @param a_coeff NP::REAL-valued multiplicative factor for the Arrhenius
   * rate.
   * @param b_coeff NP::REAL-valued power for the Arrhenius rate.
   */
  ArrheniusDataOnDevice(NP::REAL a_coeff, NP::REAL b_coeff)
      : a_coeff(a_coeff), b_coeff(b_coeff) {};

  /**
   * @brief Function to calculate the reaction rate for an Arrhenius rate
   * coefficient reaction
   *
   * @param index Read-only accessor to a loop index for a NP::ParticleLoop
   * inside which calc_data is called. NP::Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A NP::REAL-valued array of size 1 containing the calculated
   * reaction rate.
   */
  std::array<NP::REAL, 1>
  calc_data(const NP::Access::LoopIndex::Read &index,
            const NP::Access::SymVector::Write<NP::INT> &req_int_props,
            const NP::Access::SymVector::Read<NP::REAL> &req_real_props,
            typename ReactionDataBaseOnDevice::RNG_KERNEL_TYPE::KernelType
                &kernel) const {
    auto weight = req_real_props.at(this->weight_ind, index, 0);
    auto temperature = req_real_props.at(this->temperature_ind, index, 0);

    return std::array<NP::REAL, 1>{weight * this->a_coeff *
                                   sycl::pow(temperature, b_coeff)};
  }

public:
  int weight_ind, temperature_ind;
  NP::REAL a_coeff, b_coeff;
};

/**
 * @brief Reaction rate data calculation for an Arrhenius rate coefficient
 * reaction. The reaction rate is calculated as a_coeff * temperature ** b_coeff
 * * particle_weight.
 */
struct ArrheniusData : public ReactionDataBase<ArrheniusDataOnDevice> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.fluid_temperature};

  /**
   * @brief Constructor for ArrheniusData.
   *
   * @param a_coeff NP::REAL-valued multiplicative factor for the Arrhenius
   * rate.
   * @param b_coeff NP::REAL-valued power for the Arrhenius rate.
   * and the particle weight)
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names
   */
  ArrheniusData(NP::REAL a_coeff, NP::REAL b_coeff,
                std::map<int, std::string> properties_map = get_default_map());

  /**
   * @brief Index the particle weight and fluid temperature on the on-device
   * object
   */
  void index_on_device_object();
};
}; // namespace VANTAGE::Reactions
#endif
