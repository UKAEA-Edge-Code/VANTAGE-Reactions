// Explicit template instantiations shipped by the compiled VANTAGE-Reactions
// library. The matching `extern template` declarations live in
// include/reactions_lib/extern_templates.hpp, included from reactions.hpp.
//
// The set is a pay-once baseline for the documented configurations; it is
// not a closed enumeration of every Data x Kernels x DataCalc combination.

#include "../../include/reactions/reactions.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
template class FilteredMaxwellianSampler<3, ConstantRateCrossSection>;
template class SpecularReflectionData<2>;
template class SpecularReflectionData<3>;

} // namespace VANTAGE::Reactions
