#ifndef VANTAGE_REACTIONS_FIXED_RATE_DATA_IMPL_H
#define VANTAGE_REACTIONS_FIXED_RATE_DATA_IMPL_H

#include "fixed_rate_data.hpp"

namespace VANTAGE::Reactions {

FixedRateData::FixedRateData(const REAL &rate) {
  this->on_device_obj = FixedRateDataOnDevice(rate);
}

void FixedRateData::index_on_device_object() {}

} // namespace VANTAGE::Reactions

#endif
