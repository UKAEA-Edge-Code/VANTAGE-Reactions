#ifndef VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP
#define VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP

// This header is included at the end of reactions.hpp. It does two things:
//
//  1. Always defines the type aliases used to name the supported runtime
//     instantiations (VelocityReflectionPipeline, ScatteringDataCalculator,
//     KinEnergyData). The aliases are in unevaluated decltype contexts, so they
//     do not themselves trigger template instantiation.
//
//  2. In compiled mode (the default) it declares `extern template` for the
//     instantiations the library ships, so consumers do not re-instantiate
//     what the library already provides. In header-only mode
//     (-DVANTAGE_REACTIONS_HEADER_ONLY=ON) these declarations are absent and
//     every consumer instantiates from headers, exactly as before.
//
// The matching explicit instantiations live in
// src/reactions_lib/instantiations/instantiations.cpp.

#include <utility>

namespace VANTAGE::Reactions {

// Type aliases for the complex types used by the supported instantiations.
// These expressions are only used in unevaluated decltype contexts so they do
// not create objects or trigger template instantiations here.
using VelocityExtractor = decltype(extract<2>("VELOCITY"));
using WeightExtractor = decltype(extract<1>("WEIGHT"));

using VelocityReflectionPipeline =
    decltype(pipe(VelocityExtractor(NESO::Particles::Sym<REAL>("VELOCITY")),
                  SpecularReflectionData<2>()));

using ScatteringDataCalculator =
    decltype(DataCalculator<VelocityReflectionPipeline>(
        std::declval<VelocityReflectionPipeline>()));

using VelocityExtractor3D = decltype(extract<3>("VELOCITY"));

using VelocityReflectionPipeline3D =
    decltype(pipe(VelocityExtractor3D(NESO::Particles::Sym<REAL>("VELOCITY")),
                  SpecularReflectionData<3>()));

using ScatteringDataCalculator3D =
    decltype(DataCalculator<VelocityReflectionPipeline3D>(
        std::declval<VelocityReflectionPipeline3D>()));

using KinEnergyData = decltype(std::declval<WeightExtractor>() *
                               std::declval<VelocityExtractor>() *
                               std::declval<VelocityExtractor>());

#ifndef VANTAGE_REACTIONS_HEADER_ONLY

// ---------------------------------------------------------------------------
// Reactions used in the examples
// ---------------------------------------------------------------------------
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<2>,
    DataCalculator<FixedRateData, FixedRateData>>;
extern template class LinearReactionBase<
    0, FixedRateData, IoniseReactionKernels<2>, DataCalculator<FixedRateData>>;
extern template class LinearReactionBase<
    1, FixedRateData, RecombReactionKernels<2, 2>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<2, true>,
                                         ScatteringDataCalculator>;
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<3>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculator3D>;

extern template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
extern template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData,
                                               2>;
extern template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
extern template class CXReactionKernels<2>;
extern template class IoniseReactionKernels<2>;
extern template class RecombReactionKernels<2, 2>;
extern template class LinearScatteringKernels<2, true>;
extern template class CXReactionKernels<3>;
extern template class LinearScatteringKernels<3, true>;

// ---------------------------------------------------------------------------
// DataCalculator specialisations
// ---------------------------------------------------------------------------
extern template class DataCalculator<FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData>;
extern template class DataCalculator<VelocityReflectionPipeline>;

// ---------------------------------------------------------------------------
// Common transformation strategies
// ---------------------------------------------------------------------------
extern template class CellwiseAccumulator<REAL>;
extern template class WeightedCellwiseAccumulator<REAL>;
extern template class ParticleDatZeroer<REAL>;
extern template class MergeTransformationStrategy<2>;

// ---------------------------------------------------------------------------
// Downsampling strategies
// ---------------------------------------------------------------------------
extern template class VranicMergingKernels<2>;
extern template class DownsamplingStrategy<VranicMergingKernels<2>>;
extern template class DownsamplingStrategy<SimpleThinningKernels>;

// ---------------------------------------------------------------------------
// Other heavy reaction data types
// ---------------------------------------------------------------------------
extern template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
extern template class CellwiseReactionDataAccumulator<KinEnergyData>;

#endif // VANTAGE_REACTIONS_HEADER_ONLY

} // namespace VANTAGE::Reactions

#endif // VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP
