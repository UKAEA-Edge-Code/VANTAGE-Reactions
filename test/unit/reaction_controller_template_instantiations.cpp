#include "include/reaction_controller_template_instantiations.hpp"

// TestReaction is declared in the global namespace (see
// mock_reactions.hpp), so its explicit instantiations must live there too.
template class TestReaction<0>;
template class TestReaction<1>;
template class TestReaction<2>;

namespace VANTAGE::Reactions {

template class LinearReactionBase<1, FixedCoefficientData,
                                  TestReactionKernels<1>, DataCalculator<>>;

template class LinearReactionBase<1, SquaredWeightData, TestReactionKernels<1>,
                                  DataCalculator<>>;

} // namespace VANTAGE::Reactions

// ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2> is already
// shipped by the compiled library (src/reactions_lib/instantiations/
// instantiations.cpp) with a matching extern template in extern_templates.hpp
// that suppresses implicit instantiation in consumers. Only emit it from the
// test TU in header-only mode, where the library emits nothing; otherwise the
// linker sees duplicate symbols and we re-pay the codegen this file exists to
// avoid. The mock-type instantiations above are test-only and stay ungated.
#ifdef VANTAGE_REACTIONS_HEADER_ONLY
namespace VANTAGE::Reactions {
template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;
} // namespace VANTAGE::Reactions
#endif
