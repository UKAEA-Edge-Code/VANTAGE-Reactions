// Explicit template instantiations for unit tests shipped by the compiled
// VANTAGE-Reactions library. The matching `extern template` declarations live
// in test/unit/include/test_extern_templates.hpp
//
// The set is a pay-once baseline for the documented configurations; it is
// not a closed enumeration of every Data x Kernels x DataCalc combination.

#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

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
