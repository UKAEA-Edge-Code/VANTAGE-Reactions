#include "../include/reactions_lib/reaction_data/fixed_rate_data.hpp"

namespace VANTAGE::Reactions {

FixedRateData::FixedRateData(const REAL &rate) {
  this->on_device_obj = FixedRateDataOnDevice(rate);
}

void FixedRateData::index_on_device_object() {}

} // namespace VANTAGE::Reactions
