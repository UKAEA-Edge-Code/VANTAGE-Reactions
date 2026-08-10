#include "../../include/reactions_lib/reaction_data/filtered_maxwellian_sampler.hpp"

namespace VANTAGE::Reactions {

template class FilteredMaxwellianSampler<2, ConstantRateCrossSection>;
template class FilteredMaxwellianSampler<3, ConstantRateCrossSection>;

} // namespace VANTAGE::Reactions