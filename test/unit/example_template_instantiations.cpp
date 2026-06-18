#include "include/example_template_instantiations.hpp"

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

template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
template class Recombination<
    FixedRateData, DataCalculator<FixedRateData, FixedRateData, FixedRateData>,
    2>;

// ---------------------------------------------------------------------------
// Kernel classes
// ---------------------------------------------------------------------------
template class CXReactionKernels<2>;
template class IoniseReactionKernels<2>;
template class RecombReactionKernels<2, 2>;
template class LinearScatteringKernels<2, true>;

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

} // namespace VANTAGE::Reactions
