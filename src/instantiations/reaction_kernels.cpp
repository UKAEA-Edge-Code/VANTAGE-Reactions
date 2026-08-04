// Explicit template instantiations shipped by the compiled VANTAGE-Reactions
// library. The matching `extern template` declarations live in
// include/reactions_lib/extern_templates.hpp, included from reactions.hpp.
//
// The set is a pay-once baseline for the documented configurations; it is
// not a closed enumeration of every Data x Kernels x DataCalc combination.

#include "../../include/reactions/reactions.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
template class CXReactionKernels<2>;
template class CXReactionKernels<3>;
template class IoniseReactionKernels<2>;
template class IoniseReactionKernels<3>;
template class RecombReactionKernels<2>;
template class RecombReactionKernels<3>;
template class LinearScatteringKernels<2, true>;
template class LinearScatteringKernels<2, false>;
template class LinearScatteringKernels<3, true>;
template class LinearScatteringKernels<3, false>;
template class GeneralAbsorptionKernels<2>;
template class GeneralAbsorptionKernels<3>;
template class SpecularReflectionKernels<2>;
template class SpecularReflectionKernels<3>;

} // namespace VANTAGE::Reactions
