// Explicit template instantiations for the SWPM/pair reaction data used by the
// unit tests. The matching `extern template` declarations live in
// test/unit/include/test_extern_templates.hpp.
//
// CSPairData is not instantiated whole-class: its default constructor delegates
// to CSPairData(ConstantRateCrossSection(0.0)), which is ill-formed unless the
// cross-section type is ConstantRateCrossSection. Only the members the tests
// use are instantiated.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// CSPairData
// ---------------------------------------------------------------------------
template CSPairData<2, ConstantRateCrossSection>::CSPairData(
    ConstantRateCrossSection, std::map<int, std::string>);
template REAL
    CSPairData<2, ConstantRateCrossSection>::get_cs_max_rate_val(REAL);
template CSPairData<2, ConstantCrossSection>::CSPairData(
    ConstantCrossSection, std::map<int, std::string>);
template REAL CSPairData<2, ConstantCrossSection>::get_cs_max_rate_val(REAL);
template CSPairData<3, IPLCrossSection>::CSPairData(IPLCrossSection,
                                                    std::map<int, std::string>);
template REAL CSPairData<3, IPLCrossSection>::get_cs_max_rate_val(REAL);

// ---------------------------------------------------------------------------
// Hard-sphere / soft-sphere scattering data
// ---------------------------------------------------------------------------
template class HSScatteringData<2>;
template class SSScatteringData<3>;

} // namespace VANTAGE::Reactions
