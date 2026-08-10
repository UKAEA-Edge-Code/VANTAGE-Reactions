#include "../include/reactions_lib/reaction_data/grid_descriptors.hpp"

namespace VANTAGE::Reactions {

namespace grid_utils {

void append(REAL *ptr, size_t &offset, const REAL *data, size_t n) {
  std::copy(data, data + n, ptr + offset);
  offset += n;
}

} // namespace grid_utils

} // namespace VANTAGE::Reactions
