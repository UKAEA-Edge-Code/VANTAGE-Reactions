#ifndef VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP
#define VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP

// This header is included at the end of reactions.hpp. It declares `extern
// template` for the instantiations the library ships, so consumers do not
// re-instantiate what the library already provides.
//
// The matching explicit instantiations live in
// src/instantiations/instantiations.cpp.

#include <utility>

namespace VANTAGE::Reactions {

// The declarations below are grouped to mirror
// src/instantiations/instantiations.cpp

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
extern template class CXReactionKernels<2>;
extern template class CXReactionKernels<3>;
extern template class IoniseReactionKernels<2>;
extern template class IoniseReactionKernels<3>;
extern template class RecombReactionKernels<2>;
extern template class RecombReactionKernels<3>;
extern template class LinearScatteringKernels<2, true>;
extern template class LinearScatteringKernels<2, false>;
extern template class LinearScatteringKernels<3, true>;
extern template class LinearScatteringKernels<3, false>;
extern template class GeneralAbsorptionKernels<2>;
extern template class GeneralAbsorptionKernels<3>;
extern template class SpecularReflectionKernels<2>;
extern template class SpecularReflectionKernels<3>;

// ---------------------------------------------------------------------------
// Common transformation strategies
// ---------------------------------------------------------------------------
extern template class CellwiseAccumulator<REAL>;
extern template class CellwiseAccumulator<INT>;
extern template class WeightedCellwiseAccumulator<REAL>;
extern template class WeightedCellwiseAccumulator<INT>;
extern template class ParticleDatZeroer<REAL>;
extern template class ParticleDatZeroer<INT>;
extern template class MergeTransformationStrategy<2>;
extern template class MergeTransformationStrategy<3>;
extern template class CellwiseDistributor<REAL>;
extern template class CellwiseDistributor<INT>;

// ---------------------------------------------------------------------------
// Downsampling strategies
// ---------------------------------------------------------------------------
extern template class VranicMergingKernels<2>;
extern template class VranicMergingKernels<3>;
extern template class DownsamplingStrategy<SimpleThinningKernels>;
extern template class DownsamplingStrategy<VranicMergingKernels<2>>;
extern template class DownsamplingStrategy<VranicMergingKernels<3>>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
extern template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
extern template class FilteredMaxwellianSampler<3, ConstantRateCrossSection>;
extern template class SpecularReflectionData<2>;
extern template class SpecularReflectionData<3>;

} // namespace VANTAGE::Reactions

#endif // VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP
