#pragma once
#include "../reaction_base.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels/base_recombination_kernels.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace Reactions {

/**
 * struct Recombination - Reaction representing electron-impact recombination
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
   * @brief Electron impact recombination reaction construction
   *
   * @param sycl_target_ SYCL target pointer used to interface with
   * NESO-Particles routines
   * @param total_reaction_rate Sym<REAL> associated with the total reaction
   * rate ParticleDat
   * @param weight_sym Sym<REAL> associated with the weight ParticleDat
   * @param rate_data ReactionData object used to calculate the recombination 
   * rate
   * @param data_calc_obj DataCalculator that will calculate electron source
   * energy loss and the velocities of the generated neutrals
   * @param marker_species Species object representing the recombination target
   * @param electron_species Species object corresponding to the electrons
   * @param neutral_species Species object representing the neutrals that will
   * be generated
   * @param particle_spec ParticleSpec associated with the particle group this
   * reaction should act on
   * @param normalised_potential_energy Used in calculating the electron source
   * energy loss
   * @param properties_map Optional property map to be used with the
   * recombination kernels. Defaults to the default_map object
   */
  Recombination(SYCLTargetSharedPtr sycl_target_, Sym<REAL> total_reaction_rate,
                Sym<REAL> weight_sym, RateData rate_data,
                DataCalcType data_calc_obj, Species marker_species,
                Species electron_species, Species neutral_species,
                const ParticleSpec &particle_spec,
                const REAL &normalised_potential_energy,
                std::map<int, std::string> properties_map = default_map)
      : LinearReactionBase<1, RateData, RecombReactionKernels<>, DataCalcType>(
            sycl_target_, total_reaction_rate, weight_sym,
            marker_species.get_id(),
            std::array<int, 1>{static_cast<int>(neutral_species.get_id())},
            rate_data,
            RecombReactionKernels<>(marker_species, electron_species,
                                    normalised_potential_energy),
            particle_spec, data_calc_obj) {}
};
}; // namespace Reactions