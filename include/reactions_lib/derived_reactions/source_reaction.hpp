#ifndef REACTIONS_SOURCE_REACTION_H
#define REACTIONS_SOURCE_REACTION_H
#include "../data_calculator.hpp"
#include "../reaction_base.hpp"
#include "../reaction_data/fixed_rate_data.hpp"
#include "../reaction_kernel_pre_reqs.hpp"
#include "../reaction_kernels/general_linear_scattering_kernels.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Generates particles from markers without reducing marker weights.
 *
 * Uses a fixed rate of 1.0 and the standard linear-reaction timestep and
 * participating-weight limits. Product velocities are supplied by reaction
 * data. Optional momentum and energy feedback uses the marker and product
 * velocities and the product mass.
 *
 * @tparam VelocityData ReactionData type returning the product velocity.
 * @tparam ndim Number of velocity components (defaults to 2).
 * @tparam with_sources Whether to accumulate momentum and energy sources
 * (defaults to false).
 */
template <typename VelocityData, size_t ndim = 2, bool with_sources = false>
struct SourceReaction
    : public LinearReactionBase<
          1, FixedRateData, LinearScatteringKernels<ndim, with_sources, false>,
          DataCalculator<VelocityData>> {
  /**
   * @brief Constructs a particle source with fixed rate 1.0.
   *
   * @param sycl_target Compute target shared by the marker and product groups.
   * @param velocity_data ReactionData returning ndim product velocity
   * components. Samplers and composed data are supported.
   * @param marker_species Species identifying the parent markers. Their
   * weights are not modified.
   * @param product_species Species identifying the generated particles and
   * supplying the mass used for optional source feedback.
   * @param properties_map Optional property-name mapping, propagated to both
   * the reaction and its kernels. Source properties must be mapped as needed
   * when with_sources is true.
   */
  SourceReaction(
      NP::SYCLTargetSharedPtr sycl_target, VelocityData velocity_data,
      Species marker_species, Species product_species,
      const std::map<int, std::string> &properties_map = get_default_map())
      : LinearReactionBase<1, FixedRateData,
                           LinearScatteringKernels<ndim, with_sources, false>,
                           DataCalculator<VelocityData>>(
            sycl_target, marker_species.get_id(),
            std::array<int, 1>{static_cast<int>(product_species.get_id())},
            FixedRateData(1.0),
            LinearScatteringKernels<ndim, with_sources, false>(product_species,
                                                               properties_map),
            DataCalculator<VelocityData>(velocity_data), properties_map) {}
};

} // namespace VANTAGE::Reactions
#endif
