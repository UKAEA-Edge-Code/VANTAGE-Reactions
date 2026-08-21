#include "../include/reactions_lib/interp_utils.hpp"

namespace VANTAGE::Reactions::interp_utils {

// Only function in interp_utils.hpp that is called host-side, so can be defined
// here.
std::vector<size_t> construct_initial_hypercube(const size_t &ndim) {
  size_t total_num = 1 << ndim;
  std::vector<size_t> points(total_num);

  for (size_t i = 0; i < total_num; i++) {
    points[i] = (i ^ (i >> 1));
  }

  return points;
}

} // namespace VANTAGE::Reactions::interp_utils
