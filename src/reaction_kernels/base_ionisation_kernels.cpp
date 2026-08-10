#include "../include/reactions_lib/reaction_kernels/base_ionisation_kernels.hpp"

namespace VANTAGE::Reactions {
// ---------------------------------------------------------------------------
// Ionisation Kernel class
// ---------------------------------------------------------------------------
template class IoniseReactionKernels<2>;
template class IoniseReactionKernels<3>;

} // namespace VANTAGE::Reactions