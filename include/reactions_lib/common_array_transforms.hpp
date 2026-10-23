#ifndef REACTIONS_COMMON_ARRAY_TRANSFORMS_H
#define REACTIONS_COMMON_ARRAY_TRANSFORMS_H

#include "binary_array_transform_data.hpp"
#include "reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include "unary_array_transform_data.hpp"

namespace VANTAGE::Reactions {

/**
 * @brief Unary array transform taking each element of the input array and
 * calculating the value of its polynomial with given coefficients
 *
 * @tparam dim The expected input/output size
 * @tparam poly_order The order of the applied polynomial
 */
template <size_t dim, size_t poly_order>
struct PolynomialArrayTransform : AbstractUnaryArrayTransform<dim, dim> {

  PolynomialArrayTransform() = default;
  /**
   * @brief Constructor of PolynomialArrayTransform
   *
   * @param coeffs The array of polynomial coefficients, given in ascending
   * order from 0
   */
  PolynomialArrayTransform(const std::array<REAL, poly_order + 1> &coeffs)
      : coeffs(coeffs) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input) const {

    std::array<REAL, dim> result;
    REAL buffer;

    for (int i = 0; i < dim; i++) {

      buffer = 1;
      result[i] = 0;
      for (int j = 0; j < poly_order + 1; j++) {
        result[i] += this->coeffs[j] * buffer;
        buffer *= input[i];
      }
    }

    return result;
  };

private:
  std::array<REAL, poly_order + 1> coeffs;
};

/**
 * @brief Unary array transform multiplying each element of the array with a
 * scalar
 *
 * @tparam dim The expected input/output size
 */
template <size_t dim>
struct ScalerArrayTransform : AbstractUnaryArrayTransform<dim, dim> {

  ScalerArrayTransform() = default;
  /**
   * @brief Constructor of PolynomialArrayTransform
   *
   * @param mult The multiplicative constant to be used
   */
  ScalerArrayTransform(const REAL &mult) : mult(mult) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input) const {

    std::array<REAL, dim> result;

    for (int i = 0; i < dim; i++) {

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
 * @tparam dim The expected input/output size
 */
template <size_t dim>
struct UnaryProjectArrayTransform : AbstractUnaryArrayTransform<dim, dim> {

  UnaryProjectArrayTransform() = default;
  /**
   * @brief Constructor of UnaryProjectArrayTransform
   *
   * @param dir The array representing the projection direction vector
   */
  UnaryProjectArrayTransform(const std::array<REAL, dim> &dir) : dir(dir) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input) const {

    return utils::project_vector(input, this->dir);
  };

private:
  std::array<REAL, dim> dir;
};

/**
 * @brief Unary array transform projecting the input onto the plane normal to a
 * fixed vector, i.e. (I-P)*x where P is the projection operator and x the
 * input.
 *
 * @tparam dim The expected input/output size
 */
template <size_t dim>
struct UnaryProjectNormalArrayTransform
    : AbstractUnaryArrayTransform<dim, dim> {

  UnaryProjectNormalArrayTransform() = default;
  /**
   * @brief Constructor of UnaryProjectArrayTransform
   *
   * @param dir The array representing the projection direction vector
   */
  UnaryProjectNormalArrayTransform(const std::array<REAL, dim> &dir)
      : dir(dir) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input) const {

    std::array<REAL, dim> proj = utils::project_vector(input, this->dir);

    for (auto i = 0; i < dim; i++) {

      proj[i] = input[i] - proj[i];
    }

    return proj;
  };

private:
  std::array<REAL, dim> dir;
};

/**
 * @brief Binary array transform projecting the first input onto the second one
 *
 * @tparam dim The size of the transformed array
 */
template <size_t dim>
struct BinaryProjectArrayTransform
    : AbstractBinaryArrayTransform<dim, dim, dim> {

  BinaryProjectArrayTransform() = default;

  /**
   * @brief Function to apply the transform.
   *
   * @param input_1 Input REAL-valued array of size dim to project onto input_2.
   * @param input_2 Input REAL-valued array of size dim on which input_1 is
   * projected.
   *
   * @return REAL-valued array of size dim, that's the result of the projection.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input_1,
                              const std::array<REAL, dim> &input_2) const {

    return utils::project_vector(input_1, input_2);
  };
};

/**
 * @brief Binary array transform projecting the first input onto the plane
 * normal to the second one
 *
 * @tparam dim The size of the transformed array
 */
template <size_t dim>
struct BinaryProjectNormalArrayTransform
    : AbstractBinaryArrayTransform<dim, dim, dim> {

  BinaryProjectNormalArrayTransform() = default;

  /**
   * @brief Function to apply the transform.
   *
   * @param input_1 Input REAL-valued array of size dim to project onto the
   * plane normal to input_2.
   * @param input_2 Input REAL-valued array of size dim whose plane normal
   * input_1 is projected onto.
   *
   * @return REAL-valued array of size dim, that's the result of the projection.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input_1,
                              const std::array<REAL, dim> &input_2) const {

    std::array<REAL, dim> proj = utils::project_vector(input_1, input_2);

    for (auto i = 0; i < dim; i++) {

      proj[i] = input_1[i] - proj[i];
    }

    return proj;
  };
};

/**
 * @brief Unary full array transform
 *
 * @tparam dim_in The size of the transformed array
 * @tparam dim_out The size of the transformed array
 * @tparam OP_T Unary operator to be used (a struct with operator(const
 * std::array<REAL,dim_in>& input) defined)
 */
template <size_t dim_in, size_t dim_out, typename OP_T>
struct UnaryArrayOperatorTransform
    : AbstractUnaryArrayTransform<dim_in, dim_out> {

  UnaryArrayOperatorTransform() = default;

  /**
   * @brief Constructor for UnaryArrayOperatorTransform.
   *
   * @param op Unary operator.
   */
  UnaryArrayOperatorTransform(const OP_T &op) : op(op) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim_out> apply(const std::array<REAL, dim_in> &input) const {

    return this->op(input);
  };

private:
  OP_T op;
};
/**
 * @brief Unary element-wise transform
 *
 * @tparam dim The size of the transformed array
 * @tparam OP_T Unary operator to be used (a struct with operator(const REAL&
 * input) defined)
 */
template <size_t dim, typename OP_T>
struct UnaryElementwiseOperatorTransform
    : AbstractUnaryArrayTransform<dim, dim> {

  UnaryElementwiseOperatorTransform() = default;

  /**
   * @brief Constructor for UnaryElementwiseOperatorTransform.
   *
   * @param op Unary operator.
   */
  UnaryElementwiseOperatorTransform(const OP_T &op) : op(op) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input REAL-valued array of size dim to which the transform is
   * applied.
   *
   * @return REAL-valued array of size dim that is the transformed array.
   */
  std::array<REAL, dim> apply(const std::array<REAL, dim> &input) const {

    std::array<REAL, dim> result;

    for (int i = 0; i < dim; i++) {

      result[i] = this->op(input[i]);
    }
    return result;
  };

private:
  OP_T op;
};

/**
 * @brief Binary full array transform supporting broadcasting from size 1
 * arrays
 *
 * @tparam dim1 The size of the first transformed array
 * @tparam dim2 The size of the second transformed array
 * @tparam dim_out The size of the output array
 * @tparam OP_T Binary operator to be used (acting on std::arrays)
 */
template <size_t dim1, size_t dim2, size_t dim_out, typename OP_T>
struct BinaryArrayOperatorTransform
    : AbstractBinaryArrayTransform<dim1, dim2, dim_out> {

  BinaryArrayOperatorTransform() = default;

  /**
   * @brief Constructor for BinaryArrayOperatorTransform.
   *
   * @param op Binary operator.
   */
  BinaryArrayOperatorTransform(const OP_T &op) : op(op) {};

  /**
   * @brief Function to apply the transform.
   *
   * @param input_1 REAL-valued array of size dim1 that's passed to the binary
   * operator.
   * @param input_2 REAL-valued array of size dim2 that's passed to the binary
   * operator.
   *
   * @return REAL-valued array of size dim_out that is the transformed array.
   */
  std::array<REAL, dim_out> apply(const std::array<REAL, dim1> &input_1,
                                  const std::array<REAL, dim2> &input_2) const {

    return this->op(input_1, input_2);
  };

private:
  OP_T op;
};

/**
 * @brief Binary element-wise transform supporting broadcasting from size 1
 * arrays
 *
 * @tparam dim1 The size of the first transformed array
 * @tparam dim2 The size of the second transformed array
 * @tparam OP_T Binary operator to be used
 */
template <size_t dim1, size_t dim2, typename OP_T>
struct BinaryElementwiseOperatorTransform
    : AbstractBinaryArrayTransform<dim1, dim2, std::max(dim1, dim2)> {

  BinaryElementwiseOperatorTransform() = default;

  /**
   * @brief Constructor for BinaryElementwiseOperatorTransform.
   *
   * @param op Binary operator.
   */
  BinaryElementwiseOperatorTransform(const OP_T &op) : op(op) {

    if constexpr (dim1 != dim2) {

      static_assert(
          dim1 == 1 || dim2 == 1,
          "BinaryElementwiseOperatorTransform supports different array sizes "
          "only if one of them is size 1");
    };
  };

  /**
   * @brief Function to apply the transform.
   *
   * @param input_1 REAL-valued array of size dim1 that's passed elementwise to
   * the binary operator.
   * @param input_2 REAL-valued array of size dim2 that's passed elementwise to
   * the binary operator.
   *
   * @return REAL-valued array of size max(dim1, dim2) that is the transformed
   * array.
   */
  std::array<REAL, std::max(dim1, dim2)>
  apply(const std::array<REAL, dim1> &input_1,
        const std::array<REAL, dim2> &input_2) const {

    if constexpr (dim1 == dim2) {

      const size_t dim = dim1;
      std::array<REAL, dim1> result;

      for (int i = 0; i < dim; i++) {

        result[i] = this->op(input_1[i], input_2[i]);
      }
      return result;
    } else if constexpr (dim1 == 1) {

      std::array<REAL, dim2> result;

      for (int i = 0; i < dim2; i++) {

        result[i] = this->op(input_1[0], input_2[i]);
      }
      return result;
    }

    else {

      std::array<REAL, dim1> result;

      for (int i = 0; i < dim1; i++) {

        result[i] = this->op(input_1[i], input_2[0]);
      }
      return result;
    }
  };

private:
  OP_T op;
};
/**
 * @brief Binary array transform returning the dot-product of two arrays
 *
 * @tparam dim The size of the transformed array
 */
template <size_t dim>
struct BinaryDotArrayTransform : AbstractBinaryArrayTransform<dim, dim, 1> {

  BinaryDotArrayTransform() = default;

  /**
   * @brief Function to apply the transform.
   *
   * @param input_1 REAL-valued array of size dim.
   * @param input_2 REAL-valued array of size dim.
   *
   * @return REAL-valued array of size 1 that is the result of the dot product.
   */
  std::array<REAL, 1> apply(const std::array<REAL, dim> &input_1,
                            const std::array<REAL, dim> &input_2) const {
    std::array<REAL, 1> result{0};

    for (auto i = 0; i < dim; i++) {

      result[0] += input_1[i] * input_2[i];
    }
    return result;
  };
};

/**
 * @brief Overload of the "+" operator for adding the outputs of
 * two ReactionData objects using BinaryArrayTransformData. The addition is
 * elementwise.
 *
 * Example usage: auto result = obj1 + obj2;
 * Where obj1 and obj2 both inherit from ReactionDataBase.
 * The result can be treated as CompositeData-derived object.
 */
template <
    typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto operator+(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryElementwiseOperatorTransform<T::DIM, U::DIM,
                                         decltype(sycl::plus())>(sycl::plus()),
      lhs, rhs);
};

/**
 * @brief Overload of the "*" operator for multiplying the outputs of
 * two ReactionData objects using BinaryArrayTransformData. The multiplication
 * is elementwise.
 *
 * Example usage: auto result = obj1 * obj2;
 * Where obj1 and obj2 both inherit from ReactionDataBase.
 * The result can be treated as CompositeData-derived object.
 */
template <
    typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto operator*(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryElementwiseOperatorTransform<T::DIM, U::DIM,
                                         decltype(sycl::multiplies())>(
          sycl::multiplies()),
      lhs, rhs);
};

/**
 * @brief Overload of the "-" operator for subtracting the outputs of
 * two ReactionData objects using BinaryArrayTransformData. The subtraction is
 * elementwise.
 *
 * Example usage: auto result = obj1 - obj2;
 * Where obj1 and obj2 both inherit from ReactionDataBase.
 * The result can be treated as CompositeData-derived object.
 */
template <
    typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto operator-(const T &lhs, const U &rhs) {

  // TODO: change to NP::Kernel::minus when available
  return BinaryArrayTransformData(
      BinaryElementwiseOperatorTransform<T::DIM, U::DIM,
                                         decltype(std::minus())>(std::minus()),
      lhs, rhs);
};

/**
 * @brief Overload of the "/" operator for dividing the outputs of
 * two ReactionData objects using BinaryArrayTransformData. The division is
 * elementwise.
 *
 * Example usage: auto result = obj1 / obj2;
 * Where obj1 and obj2 both inherit from ReactionDataBase.
 * The result can be treated as CompositeData-derived object.
 */
template <
    typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto operator/(const T &lhs, const U &rhs) {

  // TODO: change to NP::Kernel::divides when available
  return BinaryArrayTransformData(
      BinaryElementwiseOperatorTransform<T::DIM, U::DIM,
                                         decltype(std::divides())>(
          std::divides()),
      lhs, rhs);
};

/**
 * @brief Helper function to apply a dot product to the outputs of two
 * ReactionData objects using BinaryArrayTransformData. The dot product itself
 * is applied via BinaryDotArrayTransform.
 *
 * Example usage: auto result = dot_product(obj1, obj2);
 * Where obj1 and obj2 both inherit from ReactionDataBase.
 * The result can be treated as CompositeData-derived object.
 *
 * @param lhs ReactionData object (with the same output dimensions as rhs).
 * @param rhs ReactionData object (with the same output dimensions as lhs).
 *
 * @return BinaryArrayTransformData object whose calc_data will output the
 * result of the dot product.
 */
template <
    typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>

inline auto dot_product(const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(BinaryDotArrayTransform<T::DIM>(), lhs, rhs);
};

/**
 * @brief Helper function to generate a UnaryArrayTransformData that can be used
 * to scale the output of a ReactionData object by a scalar.
 *
 * Example usage:
 *     auto velocity_data = extract<3>("VELOCITY");
 *     auto doubler = scale_by<3>(2.0);
 *     auto pipeline = pipe(velocity_data, doubler)
 *
 * The pipeline can be treated as CompositeData-derived object.
 *
 * @param mult REAL-valued scalar to multiply the elements of the output of a
 * ReactionData object.
 *
 * @return UnaryArrayTransformData object whose calc_data will output an
 * elementwise multiplication of the output of the calc_data of a ReactionData
 * object.
 */
template <size_t dim, typename ARGUMENT_PACK_T = SingleReactionDataArgumentPack>
inline auto scale_by(const REAL &mult) {

  return UnaryArrayTransformData<ScalerArrayTransform<dim>, ARGUMENT_PACK_T>(
      ScalerArrayTransform<dim>(mult));
}

/**
 * @brief Helper function to generate a UnaryArrayTransformData that can be used
 * to pass the output of a ReactionData object to a lambda function.
 *
 * Example usage:
 *     auto velocity_data = extract<2>("VELOCITY");
 *     auto vnorm_lambda = [](const std::array<REAL, 2> &arr) {
 *       return std::array<REAL, 1>{
 *           std::sqrt(arr[0] * arr[0] + arr[1] * arr[1])
 *       };
 *     };
 *     auto vnorm_lambda_wrapper =
 *         utils::LambdaWrapper<decltype(vnorm_lambda), 2>(vnorm_lambda);
 *     auto unary_transform_data =
 *         uatData<2, decltype(vnorm_lambda_wrapper)>(vnorm_lambda_wrapper);
 *     auto pipeline = pipe(velocity_data, unary_transform_data)
 *
 * The pipeline can be treated as CompositeData-derived object.
 *
 * @param lambda A lambda wrapper object that defines the transformation to
 * apply to a ReactionData object.
 *
 * @return UnaryArrayTransformData object whose calc_data will output a
 * transformation of the output of the calc_data of a ReactionData object
 * according to the lamdba argument.
 */
template <size_t dim_in, typename LAMBDA_T,
          typename ARGUMENT_PACK_T = SingleReactionDataArgumentPack>
inline auto uatData(const LAMBDA_T &lambda) {

  return UnaryArrayTransformData<
      UnaryArrayOperatorTransform<dim_in, LAMBDA_T::OUTPUT_DIM, LAMBDA_T>,
      ARGUMENT_PACK_T>(
      UnaryArrayOperatorTransform<dim_in, LAMBDA_T::OUTPUT_DIM, LAMBDA_T>(
          lambda));
}

/**
 * @brief Helper function to generate a UnaryArrayTransformData that can be used
 * to pass the output of a ReactionData object to a lambda function elementwise.
 *
 * Example usage:
 *     auto velocity_data = extract<2>("VELOCITY");
 *     auto vsquare_lambda = [](const REAL &v_comp) {
 *       return v_comp * v_comp;
 *     };
 *     auto vsquare_lambda_wrapper = utils::LambdaWrapper(vsquare_lambda);
 *     auto unary_transform_data =
 *         uetData<2,decltype(vsquare_lambda_wrapper)>(vsquare_lambda_wrapper);
 *     auto pipeline = pipe(velocity_data, unary_transform_data)
 *
 * The pipeline can be treated as CompositeData-derived object.
 *
 * @param lambda A lambda wrapper object that defines the elementwise
 * transformation to apply to a ReactionData object.
 *
 * @return UnaryArrayTransformData object whose calc_data will output an
 * elementwise transformation of the output of the calc_data of a ReactionData
 * object according to the lamdba argument.
 */
template <size_t dim_in, typename LAMBDA_T,
          typename ARGUMENT_PACK_T = SingleReactionDataArgumentPack>
inline auto uetData(const LAMBDA_T &lambda) {

  return UnaryArrayTransformData<
      UnaryElementwiseOperatorTransform<dim_in, LAMBDA_T>, ARGUMENT_PACK_T>(
      UnaryElementwiseOperatorTransform<dim_in, LAMBDA_T>(lambda));
}

/**
 * @brief Helper function to generate a BinaryArrayTransformData that can be
 * used to pass the output of two ReactionData objects to a lambda function.
 *
 * @param lambda A lambda wrapper object that defines the transformation to
 * apply to the ReactionData objects.
 * @param lhs ReactionData object
 * @param rhs ReactionData object
 *
 * @return BinaryArrayTransformData object whose calc_data will output the
 * binary transformation that will be applied to the outputs of the calc_data of
 * two ReactionData objects according to the lambda argument.
 */
template <
    typename LAMBDA_T, typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto batData(const LAMBDA_T &lambda, const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryArrayOperatorTransform<T::DIM, U::DIM, LAMBDA_T::OUTPUT_DIM,
                                   LAMBDA_T>(lambda),
      lhs, rhs);
}

/**
 * @brief Helper function to generate a BinaryArrayTransformData that can be
 * used to pass the output of two ReactionData objects to a lambda function to
 * be operated on elementwise.
 *
 * @param lambda A lambda wrapper object that defines the elementwise
 * transformation to apply to the ReactionData objects.
 * @param lhs ReactionData object
 * @param rhs ReactionData object
 *
 * @return BinaryArrayTransformData object whose calc_data will output the
 * binary transformation that will be elementwise applied to the outputs of the
 * calc_data of two ReactionData objects according to the lambda argument.
 */
template <
    typename LAMBDA_T, typename T, typename U,
    std::enable_if_t<std::is_base_of<AbstractReactionData<
                                         typename T::ON_DEVICE_OBJ_TYPE,
                                         typename T::ARGUMENT_PACK_TYPE, T::DIM,
                                         typename T::RNG_KERNEL_TYPE, 0>,
                                     T>::value,
                     bool> = true>
inline auto betData(const LAMBDA_T &lambda, const T &lhs, const U &rhs) {

  return BinaryArrayTransformData(
      BinaryElementwiseOperatorTransform<T::DIM, U::DIM, LAMBDA_T>(lambda), lhs,
      rhs);
}
} // namespace VANTAGE::Reactions
#endif
