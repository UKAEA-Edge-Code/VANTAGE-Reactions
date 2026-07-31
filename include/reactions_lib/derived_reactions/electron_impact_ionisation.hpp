#ifndef REACTIONS_ELECTRON_IMPACT_IONISATION_H
#define REACTIONS_ELECTRON_IMPACT_IONISATION_H
#include "../data_calculator.hpp"
#include "../reaction_base.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels/base_ionisation_kernels.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief A struct defining a reaction representing electron impact
 * ionisation, allowing for separate rate and energy rate calculation
 *
 * @tparam RateData ReactionData template parameter used for the rate
 * calculation
 * @tparam EnergyRateData ReactionData template parameter used for the energy
 * rate calculation,
 * @tparam ndim Optional template parameter defining the ndim_velocity template
 * parameter to use with IoniseReactionKernels
 */
template <typename RateData, typename EnergyRateData, int ndim = 2>
struct ElectronImpactIonisation
    : public LinearReactionBase<0, RateData, IoniseReactionKernels<ndim>,
                                DataCalculator<EnergyRateData>> {

  /**
   * @brief Constructor for ElectronImpactIonisation.
   *
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param rate_data ReactionData object used to calculate the ionisation rate
   * @param energy_rate_data ReactionData object used to calculate the electron
   * energy loss rate
   * @param target_species Species object representing the ionisation target
   * (and the corresponding ion fluid)
   * @param electron_species Species object corresponding to the electrons
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names.
   */
  ElectronImpactIonisation(
      SYCLTargetSharedPtr sycl_target, RateData rate_data,
      EnergyRateData energy_rate_data, Species target_species,
      Species electron_species,
      const std::map<int, std::string> &properties_map = get_default_map())
      : LinearReactionBase<0, RateData, IoniseReactionKernels<ndim>,
                           DataCalculator<EnergyRateData>>(
            sycl_target, target_species.get_id(), std::array<int, 0>{},
            rate_data,
            IoniseReactionKernels<ndim>(target_species, electron_species,
                                        electron_species, properties_map),
            DataCalculator<EnergyRateData>(energy_rate_data), properties_map) {}
};
}; // namespace VANTAGE::Reactions
#endif