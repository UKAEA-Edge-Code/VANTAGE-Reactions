#ifndef REACTIONS_COMMON_ARRAY_TRANSFORMS_H
#define REACTIONS_COMMON_ARRAY_TRANSFORMS_H
#include "unary_array_transform_data.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Unary array transform taking each element of the input array and
 * calculating the value of its polynomial with given coefficients
 *
 * @tparam DIM The expected input/output size
 * @tparam POLY_ORDER The order of the applied polynomial
 */
template <size_t DIM, size_t POLY_ORDER>
struct PolynomialArrayTransform : AbstractUnaryArrayTransform<DIM, DIM> {

  PolynomialArrayTransform() = default;
  /**
   * @brief Constructor of PolynomialArrayTransform
   *
   * @param coeffs The array of polynomial coefficients, given in ascending
   * order from 0
   */
  PolynomialArrayTransform(const std::array<REAL, POLY_ORDER + 1> &coeffs)
      : coeffs(coeffs) {};

  std::array<REAL, DIM> apply(const std::array<REAL, DIM> &input) const {

    std::array<REAL, DIM> result;
    REAL buffer;

    for (int i = 0; i < DIM; i++) {

      buffer = 1;
      result[i] = 0;
      for (int j = 0; j < POLY_ORDER + 1; j++) {
        result[i] += this->coeffs[j] * buffer;
        buffer *= input[i];
      }
    }

    return result;
  };

private:
  std::array<REAL, POLY_ORDER + 1> coeffs;
};

} // namespace VANTAGE::Reactions
#endif
