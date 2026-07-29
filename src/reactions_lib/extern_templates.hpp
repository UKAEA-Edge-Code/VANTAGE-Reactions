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

using SphericalReflectionPipeline = decltype(pipe(
    std::declval<FixedArrayData<3>>(), SphericalBasisReflectionData()));

using CartesianReflectionPipeline = decltype(pipe(
    std::declval<FixedArrayData<3>>(), CartesianBasisReflectionData()));

using ScatteringDataCalculatorSpherical =
    decltype(DataCalculator<SphericalReflectionPipeline>(
        std::declval<SphericalReflectionPipeline>()));

using ScatteringDataCalculatorCartesian =
    decltype(DataCalculator<CartesianReflectionPipeline>(
        std::declval<CartesianReflectionPipeline>()));

using KinEnergyData = decltype(std::declval<WeightExtractor>() *
                               std::declval<VelocityExtractor>() *
                               std::declval<VelocityExtractor>());

#ifndef VANTAGE_REACTIONS_HEADER_ONLY

// The declarations below are grouped to mirror
// src/reactions_lib/instantiations/instantiations.cpp and the consumer-facing
// list in docs/sphinx/source/overview/supported_instantiations.rst.

// ---------------------------------------------------------------------------
// Linear reactions
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
// Default 4th arg (DataCalc = DataCalculator<>)
extern template class LinearReactionBase<0, FixedRateData,
                                         GeneralAbsorptionKernels<2>>;
extern template class LinearReactionBase<0, FixedRateData,
                                         SpecularReflectionKernels<2>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<2, false>,
                                         ScatteringDataCalculator>;
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<2>,
    DataCalculator<FilteredMaxwellianSampler<2>>>;
extern template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<3>,
    DataCalculator<FilteredMaxwellianSampler<3>>>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculatorSpherical>;
extern template class LinearReactionBase<1, FixedRateData,
                                         LinearScatteringKernels<3, true>,
                                         ScatteringDataCalculatorCartesian>;

// ---------------------------------------------------------------------------
// Derived reactions
// ---------------------------------------------------------------------------
extern template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
extern template class ElectronImpactIonisation<FixedRateData, FixedRateData, 3>;
extern template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData,
                                               2>;
extern template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;
extern template class Recombination<
    FixedRateData,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData, FixedRateData>,
    3>;

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
// DataCalculator specialisations
// ---------------------------------------------------------------------------
extern template class DataCalculator<FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData>;
extern template class DataCalculator<VelocityReflectionPipeline>;
extern template class DataCalculator<FilteredMaxwellianSampler<2>>;
extern template class DataCalculator<FilteredMaxwellianSampler<3>>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData, FixedRateData>;
extern template class DataCalculator<SphericalReflectionPipeline>;
extern template class DataCalculator<CartesianReflectionPipeline>;
extern template class DataCalculator<VelocityReflectionPipeline3D>;

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
extern template class FilteredMaxwellianSampler<2,
                                                AMJUELFitCrossSection<2, 0, 0>>;
extern template class FilteredMaxwellianSampler<2,
                                                AMJUELFitCrossSection<2, 2, 0>>;
extern template class FilteredMaxwellianSampler<2,
                                                AMJUELFitCrossSection<2, 2, 2>>;
extern template class FilteredMaxwellianSampler<2,
                                                AMJUELFitCrossSection<3, 3, 3>>;
extern template class FilteredMaxwellianSampler<3,
                                                AMJUELFitCrossSection<2, 0, 0>>;
extern template class FilteredMaxwellianSampler<3,
                                                AMJUELFitCrossSection<2, 2, 0>>;
extern template class FilteredMaxwellianSampler<3,
                                                AMJUELFitCrossSection<2, 2, 2>>;
extern template class FilteredMaxwellianSampler<3,
                                                AMJUELFitCrossSection<3, 3, 3>>;
extern template class CellwiseReactionDataAccumulator<KinEnergyData>;
extern template class AMJUEL1DData<3>;
extern template class AMJUEL1DData<9>;
extern template class AMJUEL2DData<2, 2>;
extern template class AMJUEL2DDataH3<2, 2, 2>;
extern template class FixedArrayData<3>;
extern template class ArrayLookupData<1, false>;
extern template class ArrayLookupData<1, true>;
extern template class AMJUELFitCrossSection<2, 0, 0>;
extern template class AMJUELFitCrossSection<2, 2, 0>;
extern template class AMJUELFitCrossSection<2, 2, 2>;
extern template class AMJUELFitCrossSection<3, 3, 3>;
extern template class ExtractorData<1>;
extern template class ExtractorData<2>;
extern template class ExtractorData<3>;
extern template class SpecularReflectionData<2>;
extern template class SpecularReflectionData<3>;

// ---------------------------------------------------------------------------
// Interpolation / grid family (int-parametrized only)
// ---------------------------------------------------------------------------
extern template class CartesianGridData<1>;
extern template class CartesianGridData<2>;
extern template class CartesianGridData<3>;
extern template class CartesianGridData<4>;
extern template class CartesianGridData<5>;
extern template class TrimEvalData<5>;
extern template class TrimEvalData<7>;

#endif // VANTAGE_REACTIONS_HEADER_ONLY

} // namespace VANTAGE::Reactions

#endif // VANTAGE_REACTIONS_EXTERN_TEMPLATES_HPP
