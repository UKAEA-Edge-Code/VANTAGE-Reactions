#include "../include/reactions_lib/downsampling_kernels/vranic_merging_kernels.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Vranic merging kernels
// ---------------------------------------------------------------------------
template class VranicMergingKernels<2>;
template class VranicMergingKernels<3>;

} // namespace VANTAGE::Reactions
