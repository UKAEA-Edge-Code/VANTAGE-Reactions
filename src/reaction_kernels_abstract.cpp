#include "reactions_lib/reaction_kernels_abstract.hpp"
#include <stdexcept>

namespace VANTAGE::Reactions::detail {

void validate_kernel_product_counts(INT extent, INT count_a, INT count_b) {
  // Subtraction avoids overflow even for invalid, very large channel counts.
  if (extent < 0 || count_a < 0 || count_b < 0 || count_a > extent ||
      count_b != extent - count_a) {
    throw std::invalid_argument(
        "Kernel A/B product counts must be nonnegative and sum to the device "
        "output-state extent.");
  }
}

void validate_kernel_pre_req_ndims(INT pre_req_ndims) {
  if (pre_req_ndims < 0) {
    throw std::invalid_argument(
        "Kernel prerequisite component count must be nonnegative.");
  }
}

} // namespace VANTAGE::Reactions::detail
