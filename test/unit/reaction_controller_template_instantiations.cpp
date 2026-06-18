#include "include/reaction_controller_template_instantiations.hpp"

// TestReaction is declared in the global namespace (see
// mock_reactions.hpp), so its explicit instantiations must live there too.
template class TestReaction<0>;
template class TestReaction<1>;
template class TestReaction<2>;

namespace VANTAGE::Reactions {

template class LinearReactionBase<1, FixedCoefficientData,
                                  TestReactionKernels<1>, DataCalculator<>>;

template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;

template class LinearReactionBase<1, SquaredWeightData, TestReactionKernels<1>,
                                  DataCalculator<>>;

} // namespace VANTAGE::Reactions
