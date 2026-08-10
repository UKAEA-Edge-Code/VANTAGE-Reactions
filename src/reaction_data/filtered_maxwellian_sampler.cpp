#include "../../include/reactions_lib/reaction_data/filtered_maxwellian_sampler.hpp"

namespace VANTAGE::Reactions {

// ---------------------------------------------------------------------------
// Reaction data types
// ---------------------------------------------------------------------------
template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
template class FilteredMaxwellianSampler<3, ConstantRateCrossSection>;

} // namespace VANTAGE::Reactions