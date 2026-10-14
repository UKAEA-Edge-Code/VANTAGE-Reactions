// Explicit template instantiations for the SWPMReaction specialisations used
// by test_swpm_reaction.cpp and test_swpm_reaction_controller.cpp. The matching
// `extern template` declarations live in
// test/unit/include/test_extern_templates.hpp.
//
// SWPMReaction is not instantiated whole-class: doing so would also instantiate
// the default-DataCalc constructor
// SWPMReaction(..., DataCalc() = PairDataCalculator<>()), which requires
// ReactionData to be default constructible (HSScatteringData/SSScatteringData
// are not). The members reached by the tests are instantiated individually;
// the DataCalc-independent members are instantiated on the SWPMReactionImpl
// base class.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// SWPMReaction<2, CSPairData<2, ConstantCrossSection>,
// PairScatteringKernels<2>,
//              PairDataCalculator<HSScatteringData<2>>>
// ---------------------------------------------------------------------------
template SWPMReaction<2, CSPairData<2, ConstantCrossSection>,
                      PairScatteringKernels<2>,
                      PairDataCalculator<HSScatteringData<2>>>::
    SWPMReaction(NP::SYCLTargetSharedPtr, std::array<int, 2>,
                 std::array<int, 2>, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>,
                 PairDataCalculator<HSScatteringData<2>>,
                 const std::map<int, std::string> &);

template void SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                               PairScatteringKernels<2>>::
    calculate_rates(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT);

template void
SWPMReaction<2, CSPairData<2, ConstantCrossSection>, PairScatteringKernels<2>,
             PairDataCalculator<HSScatteringData<2>>>::
    apply(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT, double, NP::ParticleGroupSharedPtr);

template NP::LocalArraySharedPtr<REAL> &
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_device_rate_buffer();

template REAL
    SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                     PairScatteringKernels<2>>::get_sigma_v_bound(REAL);

template std::vector<int>
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_in_states();

template std::vector<int>
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::get_out_states();

template void
SWPMReactionImpl<2, CSPairData<2, ConstantCrossSection>,
                 PairScatteringKernels<2>>::set_max_buffer_size(size_t);

// ---------------------------------------------------------------------------
// SWPMReaction<2, CSPairData<3, IPLCrossSection>, PairScatteringKernels<3>,
//              PairDataCalculator<SSScatteringData<3>>>
// ---------------------------------------------------------------------------
template SWPMReaction<2, CSPairData<3, IPLCrossSection>,
                      PairScatteringKernels<3>,
                      PairDataCalculator<SSScatteringData<3>>>::
    SWPMReaction(NP::SYCLTargetSharedPtr, std::array<int, 2>,
                 std::array<int, 2>, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>,
                 PairDataCalculator<SSScatteringData<3>>,
                 const std::map<int, std::string> &);

template void
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>, PairScatteringKernels<3>>::
    calculate_rates(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT);

template void
SWPMReaction<2, CSPairData<3, IPLCrossSection>, PairScatteringKernels<3>,
             PairDataCalculator<SSScatteringData<3>>>::
    apply(
        NP::CellwisePairListAbsolute<NP::ParticleGroup, NP::CellwisePairList> &,
        INT, INT, double, NP::ParticleGroupSharedPtr);

template NP::LocalArraySharedPtr<REAL> &
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_device_rate_buffer();

template REAL
    SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                     PairScatteringKernels<3>>::get_sigma_v_bound(REAL);

template std::vector<int>
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_in_states();

template std::vector<int>
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::get_out_states();

template void
SWPMReactionImpl<2, CSPairData<3, IPLCrossSection>,
                 PairScatteringKernels<3>>::set_max_buffer_size(size_t);

} // namespace VANTAGE::Reactions
