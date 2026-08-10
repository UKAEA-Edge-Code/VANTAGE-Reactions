#include "../include/reactions_lib/reaction_kernels/specular_reflection_kernels.hpp"

namespace VANTAGE::Reactions {
// ---------------------------------------------------------------------------
// Specular Reflection Kernel class
// ---------------------------------------------------------------------------
template class SpecularReflectionKernels<2>;
template class SpecularReflectionKernels<3>;

} // namespace VANTAGE::Reactions