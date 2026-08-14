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
// Linear reactions: SpecularReflection + FilteredMaxwellianSampler DataCalcs
// ---------------------------------------------------------------------------
template class LinearReactionBase<0, FixedRateData,
                                  SpecularReflectionKernels<2>>;
template class LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                                  DataCalculator<FilteredMaxwellianSampler<2>>>;
template class LinearReactionBase<1, FixedRateData, CXReactionKernels<3>,
                                  DataCalculator<FilteredMaxwellianSampler<3>>>;

} // namespace VANTAGE::Reactions
