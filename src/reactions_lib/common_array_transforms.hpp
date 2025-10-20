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
 * @brief Binary element-wise transform supporting broadcasting from size 1
 * arrays
 *
 * @tparam DIM1 The size of the first transformed array
 * @tparam DIM2 The size of the second transformed array
 * @tparam OP Binary operator to be used
 */
template <size_t DIM1, size_t DIM2, typename OP>
struct BinaryArrayOperatorTransform
    : AbstractBinaryArrayTransform<DIM1, DIM2, std::max(DIM1, DIM2)> {

  BinaryArrayOperatorTransform<DIM1, DIM2, OP>(const OP &op) : op(op) {

    if constexpr (DIM1 != DIM2) {

      static_assert(
          DIM1 == 1 || DIM2 == 1,
          "BinaryArrayOperatorTransform supports different array sizes "
          "only if one of them is size 1");
    };
  };

  std::array<REAL, std::max(DIM1, DIM2)>
  apply(const std::array<REAL, DIM1> &input_1,
        const std::array<REAL, DIM2> &input_2) const {

    if constexpr (DIM1 == DIM2) {

      const size_t DIM = DIM1;
      std::array<REAL, DIM1> result;

      for (int i = 0; i < DIM; i++) {

        result[i] = this->op(input_1[i], input_2[i]);
      }
      return result;
    } else if constexpr (DIM1 == 1) {

      std::array<REAL, DIM2> result;

      for (int i = 0; i < DIM2; i++) {

        result[i] = this->op(input_1[0], input_2[i]);
      }
      return result;
    }

    else {

      std::array<REAL, DIM1> result;

      for (int i = 0; i < DIM1; i++) {

        result[i] = this->op(input_1[i], input_2[0]);
      }
      return result;
    }
  };

private:
  OP op;
};

template <typename ON_DEVICE_TYPE1, size_t dim1, typename RNG_TYPE1,
          typename ON_DEVICE_TYPE2, size_t dim2, typename RNG_TYPE2>
inline auto
operator+(const ReactionDataBase<ON_DEVICE_TYPE1, dim1, RNG_TYPE1, 0> &lhs,
          const ReactionDataBase<ON_DEVICE_TYPE2, dim2, RNG_TYPE2, 0> &rhs) {

  return BinaryArrayTransformData(
      BinaryArrayOperatorTransform<dim1, dim2, decltype(sycl::plus())>(
          sycl::plus()),
      lhs, rhs);
};

template <typename ON_DEVICE_TYPE1, size_t dim1, typename RNG_TYPE1,
          typename ON_DEVICE_TYPE2, size_t dim2, typename RNG_TYPE2>
inline auto
operator*(const ReactionDataBase<ON_DEVICE_TYPE1, dim1, RNG_TYPE1, 0> &lhs,
          const ReactionDataBase<ON_DEVICE_TYPE2, dim2, RNG_TYPE2, 0> &rhs) {

  return BinaryArrayTransformData(
      BinaryArrayOperatorTransform<dim1, dim2, decltype(sycl::multiplies())>(
          sycl::multiplies()),
      lhs, rhs);
};
} // namespace VANTAGE::Reactions
#endif
