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
// Reaction data types
// ---------------------------------------------------------------------------
template class AMJUELFitCrossSection<2, 0, 0>;
template class AMJUELFitCrossSection<2, 2, 0>;
template class AMJUELFitCrossSection<2, 2, 2>;
template class AMJUELFitCrossSection<3, 3, 3>;
template class CellwiseReactionDataAccumulator<KinEnergyData2D>;
template class AMJUEL1DData<3>;
template class AMJUEL1DData<9>;
template class AMJUEL2DData<2, 2>;
template class AMJUEL2DDataH3<2, 2, 2>;
template class FixedArrayData<3>;
template class ArrayLookupData<1, false>;
template class ArrayLookupData<1, true>;
template class ExtractorData<1>;
template class ExtractorData<2>;
template class ExtractorData<3>;

} // namespace VANTAGE::Reactions
