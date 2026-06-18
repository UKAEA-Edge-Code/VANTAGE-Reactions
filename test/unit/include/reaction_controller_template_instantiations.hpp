#ifndef REACTION_CONTROLLER_TEMPLATE_INSTANTIATIONS_HPP
#define REACTION_CONTROLLER_TEMPLATE_INSTANTIATIONS_HPP

/**
 * @file
 * @brief extern template declarations for the heavy deterministic templates
 *        used by test/unit/test_reaction_controller.cpp.
 *
 * The matching explicit instantiations live in
 * test/unit/reaction_controller_template_instantiations.cpp. By suppressing
 * implicit instantiation in the test translation unit we move the bulk of the
 * template codegen (and its peak memory footprint) into a dedicated TU.
 *
 * Templates that are already covered by
 * include/example_template_instantiations.hpp (MergeTransformationStrategy<2>,
 * CellwiseAccumulator<REAL> and ElectronImpactIonisation<FixedRateData,
 * FixedRateData, 2>) are pulled in via that header rather than re-declared
 * here.
 */

#include "example_template_instantiations.hpp"
#include "mock_reactions.hpp"

// Type alias for the reaction data used in semi_dsmc_test:
//   FixedCoefficientData * extract<1>("WEIGHT")
// The runtime coefficient (1.0 vs 3.0) is not part of the type, so both
// reactions in that test share this single deterministic type. Naming it lets
// us extern-template the corresponding LinearReactionBase specialisation.
using SquaredWeightData =
    decltype(std::declval<FixedCoefficientData>() * extract<1>("WEIGHT"));

// TestReaction and TestReactionKernels are declared in the global namespace
// (mock_reactions.hpp only does `using namespace VANTAGE::Reactions;`, it does
// not open that namespace), so their extern template declarations must live in
// the global namespace.
extern template class TestReaction<0>;
extern template class TestReaction<1>;
extern template class TestReaction<2>;

namespace VANTAGE::Reactions {

// LinearReactionBase used directly in surface_mode_test (and, via TestReaction,
// in the other tests). The fourth template parameter defaults to
// DataCalculator<>. TestReactionKernels is a global-namespace template arg,
// which is valid here.
extern template class LinearReactionBase<
    1, FixedCoefficientData, TestReactionKernels<1>, DataCalculator<>>;

// AMJUEL-backed ionisation reaction used in ionisation_reaction_amjuel.
extern template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData,
                                               2>;

// LinearReactionBase used in semi_dsmc_test (both reactions share the same
// SquaredWeightData type).
extern template class LinearReactionBase<
    1, SquaredWeightData, TestReactionKernels<1>, DataCalculator<>>;

} // namespace VANTAGE::Reactions

#endif // REACTION_CONTROLLER_TEMPLATE_INSTANTIATIONS_HPP
