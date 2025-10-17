#ifndef REACTIONS_COMMON_ARRAY_TRANSFORMS_H
#define REACTIONS_COMMON_ARRAY_TRANSFORMS_H
#include "binary_array_transform_data.hpp"
#include "reaction_data.hpp"
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

/**
 * @brief Binary element-wise addition transform
 *
 * @tparam DIM The size of the transformed arrays
 */
template <size_t DIM>
struct BinaryArrayAddTransform : AbstractBinaryArrayTransform<DIM, DIM, DIM> {

  std::array<REAL, DIM> apply(const std::array<REAL, DIM> &input_1,
                              const std::array<REAL, DIM> &input_2) const {

    std::array<REAL, DIM> result;

    for (int i = 0; i < DIM; i++) {

      result[i] = input_1[i] + input_2[i];
    }
    return result;
  };
};

template <typename ON_DEVICE_TYPE1, size_t dim, typename RNG_TYPE1,
          typename ON_DEVICE_TYPE2, typename RNG_TYPE2>
auto operator+(
    const ReactionDataBase<ON_DEVICE_TYPE1, dim, RNG_TYPE1, 0> &lhs,
    const ReactionDataBase<ON_DEVICE_TYPE2, dim, RNG_TYPE2, 0> &rhs) {

  return BinaryArrayTransformData(BinaryArrayAddTransform<dim>(), lhs, rhs);
};
} // namespace VANTAGE::Reactions
#endif
