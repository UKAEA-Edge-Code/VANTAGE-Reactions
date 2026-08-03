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

// ---------------------------------------------------------------------------
// Common transformation strategies
// ---------------------------------------------------------------------------
template class CellwiseAccumulator<REAL>;
template class CellwiseAccumulator<INT>;
template class WeightedCellwiseAccumulator<REAL>;
template class WeightedCellwiseAccumulator<INT>;
template class ParticleDatZeroer<REAL>;
template class ParticleDatZeroer<INT>;
template class MergeTransformationStrategy<2>;
template class MergeTransformationStrategy<3>;
template class CellwiseDistributor<REAL>;
template class CellwiseDistributor<INT>;

// ---------------------------------------------------------------------------
// Downsampling strategies
// ---------------------------------------------------------------------------
template class VranicMergingKernels<2>;
template class VranicMergingKernels<3>;
template class DownsamplingStrategy<SimpleThinningKernels>;
template class DownsamplingStrategy<VranicMergingKernels<2>>;
template class DownsamplingStrategy<VranicMergingKernels<3>>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
template class FilteredMaxwellianSampler<3, ConstantRateCrossSection>;
template class SpecularReflectionData<2>;
template class SpecularReflectionData<3>;

} // namespace VANTAGE::Reactions
