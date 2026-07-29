// Explicit template instantiations shipped by the compiled VANTAGE-Reactions
// library. Compiled only when VANTAGE_REACTIONS_HEADER_ONLY is OFF (see
// src/CMakeLists.txt). The matching `extern template` declarations live in
// src/reactions_lib/extern_templates.hpp, included from reactions.hpp.

#include "reactions/reactions.hpp"
#include "reactions_lib/extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Reactions used in the examples
// ---------------------------------------------------------------------------
template class LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                                  DataCalculator<FixedRateData, FixedRateData>>;
template class LinearReactionBase<0, FixedRateData, IoniseReactionKernels<2>,
                                  DataCalculator<FixedRateData>>;
template class LinearReactionBase<
    1, FixedRateData, RecombReactionKernels<2, 2>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<2, true>,
                                  ScatteringDataCalculator>;
template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<3>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<3, true>,
                                  ScatteringDataCalculator3D>;

template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;
template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
template class CXReactionKernels<2>;
template class IoniseReactionKernels<2>;
template class RecombReactionKernels<2>;
template class LinearScatteringKernels<2, true>;
template class CXReactionKernels<3>;
template class LinearScatteringKernels<3, true>;

// ---------------------------------------------------------------------------
// DataCalculator specialisations
// ---------------------------------------------------------------------------
template class DataCalculator<FixedRateData>;
template class DataCalculator<FixedRateData, FixedRateData>;
template class DataCalculator<FixedRateData, FixedRateData, FixedRateData>;
template class DataCalculator<VelocityReflectionPipeline>;

// ---------------------------------------------------------------------------
// Common transformation strategies
// ---------------------------------------------------------------------------
template class CellwiseAccumulator<REAL>;
template class WeightedCellwiseAccumulator<REAL>;
template class ParticleDatZeroer<REAL>;
template class MergeTransformationStrategy<2>;

// ---------------------------------------------------------------------------
// Downsampling strategies
// ---------------------------------------------------------------------------
template class VranicMergingKernels<2>;
template class DownsamplingStrategy<VranicMergingKernels<2>>;
template class DownsamplingStrategy<SimpleThinningKernels>;

// ---------------------------------------------------------------------------
// Other heavy reaction data types
// ---------------------------------------------------------------------------
template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
template class CellwiseReactionDataAccumulator<KinEnergyData>;

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
template class GeneralAbsorptionKernels<2>;
template class SpecularReflectionKernels<2>;
template class LinearScatteringKernels<2, false>;
template class GeneralAbsorptionKernels<3>;
template class SpecularReflectionKernels<3>;
template class LinearScatteringKernels<3, false>;
template class IoniseReactionKernels<3>;
template class RecombReactionKernels<3, 3>;

// ---------------------------------------------------------------------------
// DataCalculator specialisations
// ---------------------------------------------------------------------------
template class DataCalculator<FilteredMaxwellianSampler<2>>;
template class DataCalculator<FilteredMaxwellianSampler<3>>;
template class DataCalculator<FixedRateData, FixedRateData, FixedRateData,
                              FixedRateData>;
template class DataCalculator<SphericalReflectionPipeline>;
template class DataCalculator<CartesianReflectionPipeline>;
template class DataCalculator<VelocityReflectionPipeline3D>;

// ---------------------------------------------------------------------------
// Reactions
// ---------------------------------------------------------------------------
// Default 4th arg (DataCalc = DataCalculator<>)
template class LinearReactionBase<0, FixedRateData,
                                  GeneralAbsorptionKernels<2>>;
template class LinearReactionBase<0, FixedRateData,
                                  SpecularReflectionKernels<2>>;

template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<2, false>,
                                  ScatteringDataCalculator>;
template class LinearReactionBase<1, FixedRateData, CXReactionKernels<2>,
                                  DataCalculator<FilteredMaxwellianSampler<2>>>;
template class LinearReactionBase<1, FixedRateData, CXReactionKernels<3>,
                                  DataCalculator<FilteredMaxwellianSampler<3>>>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<3, true>,
                                  ScatteringDataCalculatorSpherical>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<3, true>,
                                  ScatteringDataCalculatorCartesian>;

template class Recombination<
    FixedRateData,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData, FixedRateData>,
    3>;

template class ElectronImpactIonisation<FixedRateData, FixedRateData, 3>;

// ---------------------------------------------------------------------------
// Transformation strategies
// ---------------------------------------------------------------------------
template class CellwiseAccumulator<INT>;
template class WeightedCellwiseAccumulator<INT>;
template class ParticleDatZeroer<INT>;
template class CellwiseDistributor<REAL>;
template class CellwiseDistributor<INT>;
template class MergeTransformationStrategy<3>;

// ---------------------------------------------------------------------------
// Downsampling strategies
// ---------------------------------------------------------------------------
template class VranicMergingKernels<3>;
template class DownsamplingStrategy<VranicMergingKernels<3>>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
template class AMJUEL1DData<3>;
template class AMJUEL1DData<9>;
template class AMJUEL2DData<2, 2>;
template class AMJUEL2DDataH3<2, 2>;
template class FixedArrayData<3>;
template class ArrayLookupData<1>;
template class ArrayLookupData<1, true>;
template class AMJUELFitCrossSection<2, 0, 0>;
template class AMJUELFitCrossSection<2, 2, 0>;
template class AMJUELFitCrossSection<2, 2, 2>;
template class AMJUELFitCrossSection<3, 3, 3>;
template class FilteredMaxwellianSampler<3>;
template class ExtractorData<1>;
template class ExtractorData<2>;
template class ExtractorData<3>;
template class SpecularReflectionData<2>;
template class SpecularReflectionData<3>;

// ---------------------------------------------------------------------------
// Interpolation / grid family (int-parametrized only)
// ---------------------------------------------------------------------------
template class CartesianGridData<1>;
template class CartesianGridData<2>;
template class CartesianGridData<3>;
template class CartesianGridData<4>;
template class CartesianGridData<5>;
template class TrimEvalData<5>;
template class TrimEvalData<7>;

} // namespace VANTAGE::Reactions
