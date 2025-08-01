#ifndef REACTIONS_RECOMBINATION_REACTION_H
#define REACTIONS_RECOMBINATION_REACTION_H
#include "../reaction_base.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels/base_recombination_kernels.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief A struct defining a reaction representing recombination. 
 * Takes in a marker species, which represents the ions,
 * and produces products based on their weights,
 * without reducing them. The user is responsible for setting the weight of
 * the marker species in a way that reproduces the sources they want.
 *
 * @tparam RateData ReactionData template parameter used for the rate
 * calculation
 * @tparam DataCalcType DataCalculator template parameter used for calculating,
 * electron source energy loss and the velocities for generated neutrals
 *
 */

template <typename RateData, typename DataCalcType>
struct Recombination
    : public LinearReactionBase<1, RateData, RecombReactionKernels<>,
                                DataCalcType> {

  /**
   * @brief Constructor for Recombination.
   *
   * @param sycl_target SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param rate_data ReactionData object used to calculate the recombination
   * rate
   * @param data_calc_obj DataCalculator that will calculate electron source
   * energy loss and the velocities of the generated neutrals, in that order (so
   * dimensionality 3 or 4, depending on velocity space dim)
   * @param marker_species Species object representing the recombination target
   * - will only be used as source locations and their weight will impact the
   * rate, but it won't be changed
   * @param electron_species Species object corresponding to the electrons
   * @param neutral_species Species object representing the neutrals that will
   * be generated
   * @param normalised_potential_energy Used in calculating the electron source
   * energy loss, the rate of which is given by the first data_calc_obj element
   * + the potential energy x rate
   * @param properties_map (Optional) A std::map<int, std::string> object to be used when
   * remapping property names.
   */
  Recombination(SYCLTargetSharedPtr sycl_target, RateData rate_data,
                DataCalcType data_calc_obj, Species marker_species,
                Species electron_species, Species neutral_species,
                const REAL &normalised_potential_energy,
                const std::map<int, std::string> &properties_map = get_default_map())
      : LinearReactionBase<1, RateData, RecombReactionKernels<>, DataCalcType>(
            sycl_target, marker_species.get_id(),
            std::array<int, 1>{static_cast<int>(neutral_species.get_id())},
            rate_data,
            RecombReactionKernels<>(marker_species, electron_species,
                                    normalised_potential_energy),
            data_calc_obj, properties_map) {}
};
}; // namespace VANTAGE::Reactions
#endif
