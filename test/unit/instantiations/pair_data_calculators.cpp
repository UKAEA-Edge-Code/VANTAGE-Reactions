// Explicit template instantiations for the pair data calculators and the pair
// pipelines used to build them, as exercised by test_pair_data_calculator.cpp
// and the SWPM tests. The matching `extern template` declarations live in
// test/unit/include/test_extern_templates.hpp.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Pair pipeline / transform plumbing
// ---------------------------------------------------------------------------
template class ConcatenatorData<CSPairData<2, ConstantRateCrossSection>,
                                CSPairData<2, ConstantCrossSection>>;
template class UnaryArrayTransformData<ScalerArrayTransform<1>,
                                       PairReactionDataArgumentPack>;
template class PipelineData<
    CSPairData<2, ConstantRateCrossSection>,
    UnaryArrayTransformData<ScalerArrayTransform<1>,
                            PairReactionDataArgumentPack>>;
template class BinaryArrayTransformData<
    BinaryElementwiseOperatorTransform<1, 1, decltype(sycl::plus())>,
    CSPairData<2, ConstantRateCrossSection>,
    CSPairData<2, ConstantCrossSection>>;

// ---------------------------------------------------------------------------
// PairDataCalculator specialisations
// ---------------------------------------------------------------------------
template class PairDataCalculator<CSPairData<2, ConstantRateCrossSection>>;
template class PairDataCalculator<CSPairData<2, ConstantRateCrossSection>,
                                  CSPairData<2, ConstantCrossSection>>;
template class PairDataCalculator<
    ConcatenatorData<CSPairData<2, ConstantRateCrossSection>,
                     CSPairData<2, ConstantCrossSection>>>;
template class PairDataCalculator<
    PipelineData<CSPairData<2, ConstantRateCrossSection>,
                 UnaryArrayTransformData<ScalerArrayTransform<1>,
                                         PairReactionDataArgumentPack>>>;
template class PairDataCalculator<BinaryArrayTransformData<
    BinaryElementwiseOperatorTransform<1, 1, decltype(sycl::plus())>,
    CSPairData<2, ConstantRateCrossSection>,
    CSPairData<2, ConstantCrossSection>>>;
template class PairDataCalculator<HSScatteringData<2>>;
template class PairDataCalculator<SSScatteringData<3>>;

} // namespace VANTAGE::Reactions
