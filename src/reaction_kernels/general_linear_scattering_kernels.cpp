#include "../include/reactions_lib/reaction_kernels/general_linear_scattering_kernels.hpp"

namespace VANTAGE::Reactions {

template class LinearScatteringKernels<2, true>;
template class LinearScatteringKernels<2, false>;
template class LinearScatteringKernels<3, true>;
template class LinearScatteringKernels<3, false>;

} // namespace VANTAGE::Reactions