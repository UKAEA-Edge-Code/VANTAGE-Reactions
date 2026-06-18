#ifndef EXAMPLE_TEMPLATE_INSTANTIATIONS_HPP
#define EXAMPLE_TEMPLATE_INSTANTIATIONS_HPP

#include <utility>

#include "reactions/reactions.hpp"

namespace VANTAGE::Reactions {

// Type aliases for complex types used in the examples.
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

using KinEnergyData = decltype(std::declval<WeightExtractor>() *
                               std::declval<VelocityExtractor>() *
                               std::declval<VelocityExtractor>());

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

extern template class ElectronImpactIonisation<FixedRateData, FixedRateData, 2>;
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

} // namespace VANTAGE::Reactions

#endif // EXAMPLE_TEMPLATE_INSTANTIATIONS_HPP
