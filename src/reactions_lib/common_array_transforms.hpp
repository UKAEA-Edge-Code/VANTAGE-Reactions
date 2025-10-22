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
 * @brief Unary array transform multiplying each element of the array with a
 * scalar
 *
 * @tparam DIM The expected input/output size
 */
template <size_t DIM>
struct ScalerArrayTransform : AbstractUnaryArrayTransform<DIM, DIM> {

  ScalerArrayTransform() = default;
  /**
   * @brief Constructor of PolynomialArrayTransform
   *
   * @param mult The multiplicative constant to be used
   */
  ScalerArrayTransform(const REAL &mult) : mult(mult) {};

  std::array<REAL, DIM> apply(const std::array<REAL, DIM> &input) const {

    std::array<REAL, DIM> result;

    for (int i = 0; i < DIM; i++) {

      result[i] = this->mult * input[i];
    }

    return result;
  };

private:
  REAL mult;
};

/**
 * @brief Unary array transform projecting the input on a fixed input vector
 *
 * @tparam DIM The expected input/output size
 */
template <size_t DIM>
struct UnaryProjectArrayTransform : AbstractUnaryArrayTransform<DIM, DIM> {

  UnaryProjectArrayTransform() = default;
  /**
   * @brief Constructor of UnaryProjectArrayTransform
   *
   * @param dir The array representing the projection direction vector
   */
  UnaryProjectArrayTransform(const std::array<REAL, DIM> &dir) : dir(dir) {};

  std::array<REAL, DIM> apply(const std::array<REAL, DIM> &input) const {

    return utils::project_vector(input, this->dir);
  };

private:
  std::array<REAL, DIM> dir;
};

/**
 * @brief Binary array transform projecting the first input onto the second one
 *
 * @tparam DIM The size of the transformed array
 */
template <size_t DIM>
struct BinaryProjectArrayTransform
    : AbstractBinaryArrayTransform<DIM, DIM, DIM> {

  BinaryProjectArrayTransform() = default;

  std::array<REAL, DIM> apply(const std::array<REAL, DIM> &input_1,
                              const std::array<REAL, DIM> &input_2) const {

    return utils::project_vector(input_1, input_2);
  };
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

  BinaryArrayOperatorTransform() = default;
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

/**
 * @brief Binary array transform returning the dot-product of two arrays
 *
 * @tparam DIM The size of the transformed array
 */
template <size_t DIM>
struct BinaryDotArrayTransform : AbstractBinaryArrayTransform<DIM, DIM, 1> {

  BinaryDotArrayTransform() = default;

  std::array<REAL, 1> apply(const std::array<REAL, DIM> &input_1,
                            const std::array<REAL, DIM> &input_2) const {
    std::array<REAL, 1> result{0};

    for (auto i = 0; i < DIM; i++) {

      result[0] += input_1[i] * input_2[i];
    }
    return result;
  };
};
template <
    typename T, typename U,
    std::enable_if_t<
        std::is_base_of<ReactionDataBase<typename T::ON_DEVICE_OBJ_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                        T>::value,
        bool> = true>
inline auto operator+(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryArrayOperatorTransform<T::DIM, U::DIM, decltype(sycl::plus())>(
          sycl::plus()),
      lhs, rhs);
};

template <
    typename T, typename U,
    std::enable_if_t<
        std::is_base_of<ReactionDataBase<typename T::ON_DEVICE_OBJ_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                        T>::value,
        bool> = true>
inline auto operator*(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryArrayOperatorTransform<T::DIM, U::DIM,
                                   decltype(sycl::multiplies())>(
          sycl::multiplies()),
      lhs, rhs);
};

template <
    typename T, typename U,
    std::enable_if_t<
        std::is_base_of<ReactionDataBase<typename T::ON_DEVICE_OBJ_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                        T>::value,
        bool> = true>

inline auto dot_product(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(BinaryDotArrayTransform<T::DIM>(), lhs, rhs);
};
template <size_t DIM> inline auto scale_by(const REAL &mult) {

  return UnaryArrayTransformData(ScalerArrayTransform<DIM>(mult));
}

} // namespace VANTAGE::Reactions
#endif
