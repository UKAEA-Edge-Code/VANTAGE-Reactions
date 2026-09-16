// Explicit template instantiations for the SWPM specification and controller
// used by test_swpm_reaction.cpp and test_swpm_reaction_controller.cpp. The
// matching `extern template` declarations live in
// test/unit/include/test_extern_templates.hpp.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

template class AbstractSWPMSpecification<NP::HostAtomicBlockKernelRNG<REAL>>;
template class SWPMReactionController<SWPMDSMCSpecification,
                                      NP::HostRNGGenerationFunction<REAL>>;

} // namespace VANTAGE::Reactions
