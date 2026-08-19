// Explicit template instantiations for unit tests shipped by the compiled
// VANTAGE-Reactions library. The matching `extern template` declarations live
// in test/unit/include/test_extern_templates.hpp
//
// The set is a pay-once baseline for the documented configurations; it is
// not a closed enumeration of every Data x Kernels x DataCalc combination.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Derived reactions
// ---------------------------------------------------------------------------
template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;

} // namespace VANTAGE::Reactions
