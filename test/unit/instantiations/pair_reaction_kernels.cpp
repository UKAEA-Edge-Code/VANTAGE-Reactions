// Explicit template instantiations for the SWPM/pair reaction kernels used by
// the unit tests. The matching `extern template` declarations live in
// test/unit/include/test_extern_templates.hpp.

#include "../include/test_common.hpp"
#include "../include/test_extern_templates.hpp"

namespace VANTAGE::Reactions {

template class PairScatteringKernels<2>;
template class PairScatteringKernels<3>;

} // namespace VANTAGE::Reactions
