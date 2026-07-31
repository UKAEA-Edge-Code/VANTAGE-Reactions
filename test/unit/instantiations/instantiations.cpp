// Explicit template instantiations for unit tests shipped by the compiled
// VANTAGE-Reactions library. The matching `extern template` declarations live
// in test/unit/include/test_extern_templates.hpp
//
// The set is a pay-once baseline for the documented configurations; it is
// not a closed enumeration of every Data x Kernels x DataCalc combination.

#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Linear reactions
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
                                  ScatteringDataCalculator2D>;
template class LinearReactionBase<
    1, FixedRateData, CXReactionKernels<3>,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData>>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<3, true>,
                                  ScatteringDataCalculator3D>;
// Default 4th arg (DataCalc = DataCalculator<>)
template class LinearReactionBase<0, FixedRateData,
                                  GeneralAbsorptionKernels<2>>;
template class LinearReactionBase<0, FixedRateData,
                                  SpecularReflectionKernels<2>>;
template class LinearReactionBase<1, FixedRateData,
                                  LinearScatteringKernels<2, false>,
                                  ScatteringDataCalculator2D>;
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

// ---------------------------------------------------------------------------
// Derived reactions
// ---------------------------------------------------------------------------
template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
template class ElectronImpactIonisation<FixedRateData, FixedRateData, 3>;
template class ElectronImpactIonisation<AMJUEL1DData<9>, FixedRateData, 2>;
template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;
template class Recombination<
    FixedRateData,
    DataCalculator<FixedRateData, FixedRateData, FixedRateData, FixedRateData>,
    3>;

// ---------------------------------------------------------------------------
// DataCalculator specialisations
// ---------------------------------------------------------------------------
template class DataCalculator<FixedRateData>;
template class DataCalculator<FixedRateData, FixedRateData>;
template class DataCalculator<FixedRateData, FixedRateData, FixedRateData>;
template class DataCalculator<VelocityReflectionPipeline2D>;
template class DataCalculator<FilteredMaxwellianSampler<2>>;
template class DataCalculator<FilteredMaxwellianSampler<3>>;
template class DataCalculator<FixedRateData, FixedRateData, FixedRateData,
                              FixedRateData>;
template class DataCalculator<SphericalReflectionPipeline>;
template class DataCalculator<CartesianReflectionPipeline>;
template class DataCalculator<VelocityReflectionPipeline3D>;

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
template class AMJUELFitCrossSection<2, 0, 0>;
template class AMJUELFitCrossSection<2, 2, 0>;
template class AMJUELFitCrossSection<2, 2, 2>;
template class AMJUELFitCrossSection<3, 3, 3>;
template class FilteredMaxwellianSampler<2, AMJUELFitCrossSection<2, 0, 0>>;
template class FilteredMaxwellianSampler<2, AMJUELFitCrossSection<2, 2, 0>>;
template class FilteredMaxwellianSampler<2, AMJUELFitCrossSection<2, 2, 2>>;
template class FilteredMaxwellianSampler<2, AMJUELFitCrossSection<3, 3, 3>>;
template class FilteredMaxwellianSampler<3, AMJUELFitCrossSection<2, 0, 0>>;
template class FilteredMaxwellianSampler<3, AMJUELFitCrossSection<2, 2, 0>>;
template class FilteredMaxwellianSampler<3, AMJUELFitCrossSection<2, 2, 2>>;
template class FilteredMaxwellianSampler<3, AMJUELFitCrossSection<3, 3, 3>>;
template class CellwiseReactionDataAccumulator<KinEnergyData2D>;
template class AMJUEL1DData<3>;
template class AMJUEL1DData<9>;
template class AMJUEL2DData<2, 2>;
template class AMJUEL2DDataH3<2, 2, 2>;
template class FixedArrayData<3>;
template class ArrayLookupData<1, false>;
template class ArrayLookupData<1, true>;
template class ExtractorData<1>;
template class ExtractorData<2>;
template class ExtractorData<3>;

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