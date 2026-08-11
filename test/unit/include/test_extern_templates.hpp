#ifndef VANTAGE_REACTIONS_TYPE_ALIASES_HPP
#define VANTAGE_REACTIONS_TYPE_ALIASES_HPP

#include <reactions/reactions.hpp>

namespace VANTAGE::Reactions {

// Type aliases for the complex types used by the supported instantiations.
// These expressions are only used in unevaluated decltype contexts so they do
// not create objects or trigger template instantiations here.
using VelocityExtractor2D = decltype(extract<2>("VELOCITY"));
using WeightExtractor = decltype(extract<1>("WEIGHT"));

using SquaredWeightData =
    decltype(std::declval<FixedCoefficientData>() * extract<1>("WEIGHT"));

using VelocityReflectionPipeline2D = decltype(pipe(
    VelocityExtractor2D(NESO::Particles::Sym<NP::REAL>("VELOCITY")),
    SpecularReflectionData<2>()));

using ScatteringDataCalculator2D =
    decltype(DataCalculator<VelocityReflectionPipeline2D>(
        std::declval<VelocityReflectionPipeline2D>()));

using VelocityExtractor3D = decltype(extract<3>("VELOCITY"));

using VelocityReflectionPipeline3D = decltype(pipe(
    VelocityExtractor3D(NESO::Particles::Sym<NP::REAL>("VELOCITY")),
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

using KinEnergyData2D = decltype(std::declval<WeightExtractor>() *
                                 std::declval<VelocityExtractor2D>() *
                                 std::declval<VelocityExtractor2D>());

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
                                         ScatteringDataCalculator2D>;
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
                                         ScatteringDataCalculator2D>;
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
// DataCalculator specialisations
// ---------------------------------------------------------------------------
extern template class DataCalculator<FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData>;
extern template class DataCalculator<VelocityReflectionPipeline2D>;
extern template class DataCalculator<FilteredMaxwellianSampler<2>>;
extern template class DataCalculator<FilteredMaxwellianSampler<3>>;
extern template class DataCalculator<FixedRateData, FixedRateData,
                                     FixedRateData, FixedRateData>;
extern template class DataCalculator<SphericalReflectionPipeline>;
extern template class DataCalculator<CartesianReflectionPipeline>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
extern template class AMJUELFitCrossSection<2, 0, 0>;
extern template class AMJUELFitCrossSection<2, 2, 0>;
extern template class AMJUELFitCrossSection<2, 2, 2>;
extern template class AMJUELFitCrossSection<3, 3, 3>;
extern template class CellwiseReactionDataAccumulator<KinEnergyData2D>;
extern template class AMJUEL1DData<3>;
extern template class AMJUEL1DData<9>;
extern template class AMJUEL2DData<2, 2>;
extern template class AMJUEL2DDataH3<2, 2, 2>;
extern template class FixedArrayData<3>;
extern template class ArrayLookupData<1, false>;
extern template class ArrayLookupData<1, true>;
extern template class ExtractorData<1>;
extern template class ExtractorData<2>;

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

} // namespace VANTAGE::Reactions
#endif