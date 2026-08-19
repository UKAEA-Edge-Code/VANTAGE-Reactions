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
// Linear reactions: LinearScatteringKernels + ScatteringDataCalculator aliases
// ---------------------------------------------------------------------------
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<2, true>,
                                  ScatteringDataCalculator2D>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<2, false>,
                                  ScatteringDataCalculator2D>;

} // namespace VANTAGE::Reactions
