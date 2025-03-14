#pragma once
#include "../reaction_base.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels/base_recombination_kernels.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace Reactions {
template <typename RateData, typename DataCalcType>
struct Recombination
    : public LinearReactionBase<1, RateData, RecombReactionKernels<>,
                                DataCalcType> {

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