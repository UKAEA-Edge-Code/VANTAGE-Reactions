#ifndef REACTIONS_PAIR_REACTION_DATA_H
#define REACTIONS_PAIR_REACTION_DATA_H
#include "reaction_data.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <type_traits>
#include <utility>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * @brief Base pair reaction data object.
 *
 * @tparam ON_DEVICE_TYPE Type of the on-device object
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_TYPE Sets the type of RNG that is used for sampling (Optional).
 * @tparam input_dim The dimension of the input array (Optional, defaults to 0,
 * not defining the corresponding calc_data)
 */
template <typename ON_DEVICE_TYPE, size_t dim = 1,
          typename RNG_TYPE = DEFAULT_RNG_KERNEL, size_t input_dim = 0>
struct PairReactionDataBase {

  using RNG_KERNEL_TYPE = RNG_TYPE;
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  /**
   * @brief Validate that the on-device type defines calc_data with the
   * parameter signature and return type expected by this host base class.
   */
  static constexpr bool validate_on_device_type() {
    if constexpr (ON_DEVICE_TYPE::INPUT_DIM > 0) {
      using input_t = const std::array<typename ON_DEVICE_TYPE::INPUT_TYPE,
                                       ON_DEVICE_TYPE::INPUT_DIM> &;

      static_assert(
          is_calc_data_callable_v<ON_DEVICE_TYPE, input_t,
                                  const Access::PairLoopIndex::Read &,
                                  const Access::SymVector::Write<INT> &,
                                  const Access::SymVector::Read<REAL> &,
                                  const Access::SymVector::Write<INT> &,
                                  const Access::SymVector::Read<REAL> &,
                                  typename RNG_TYPE::KernelType &>,
          "ON_DEVICE_TYPE::calc_data parameter signature mismatch");

      static_assert(check_calc_data_return_type<
                        ON_DEVICE_TYPE,
                        std::array<typename ON_DEVICE_TYPE::VALUE_TYPE,
                                   ON_DEVICE_TYPE::DIM>,
                        input_t, const Access::PairLoopIndex::Read &,
                        const Access::SymVector::Write<INT> &,
                        const Access::SymVector::Read<REAL> &,
                        const Access::SymVector::Write<INT> &,
                        const Access::SymVector::Read<REAL> &,
                        typename RNG_TYPE::KernelType &>(),
                    "ON_DEVICE_TYPE::calc_data return type mismatch");
    }
    return true;
  }

  /**
   * @brief Constructor for PairReactionDataBase.
   *
   * @param required_int_props_a Properties<INT> object containing information
   * regarding the required INT-based properties of the first particle for the
   * reaction data.
   * @param required_real_props_a Properties<REAL> object containing information
   * regarding the required REAL-based properties of the first particle for the
   * reaction data.
   * @param required_int_props_b Properties<INT> object containing information
   * regarding the required INT-based properties of the second particle for the
   * reaction data.
   * @param required_real_props_b Properties<REAL> object containing information
   * regarding the required REAL-based properties of the second particle for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionDataBase(
      Properties<INT> required_int_props_a,
      Properties<REAL> required_real_props_a,
      Properties<INT> required_int_props_b,
      Properties<REAL> required_real_props_b,
      std::map<int, std::string> properties_map = get_default_map())
      : required_int_props_a(
            ArgumentNameSet(required_int_props_a, properties_map)),
        required_real_props_a(
            ArgumentNameSet(required_real_props_a, properties_map)),
        required_int_props_b(
            ArgumentNameSet(required_int_props_b, properties_map)),
        required_real_props_b(
            ArgumentNameSet(required_real_props_b, properties_map)),
        properties_map(properties_map) {

    static_assert(validate_on_device_type());
    this->rng_kernel = std::make_shared<RNG_TYPE>();
  }

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets no required
   * properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionDataBase(
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(Properties<INT>(), Properties<REAL>(),
                             Properties<INT>(), Properties<REAL>(),
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets only required int
   * properties that are the same for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionDataBase(
      Properties<INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(required_int_props, Properties<REAL>(),
                             required_int_props, Properties<REAL>(),
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets only required real
   * properties that are the same for both particles.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionDataBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(Properties<INT>(), required_real_props,
                             Properties<INT>(), required_real_props,
                             properties_map) {}

  /**
   * \overload
   * @brief Constructor for PairReactionDataBase that sets required int and
   * real properties, same for both particles.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  PairReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : PairReactionDataBase(required_int_props, required_real_props,
                             required_int_props, required_real_props,
                             properties_map) {}
  /**
   * @brief Return all required integer properties for the first particle
   *
   */
  ArgumentNameSet<INT> get_required_int_props_a() {
    return this->required_int_props_a;
  }

  /**
   * @brief Setter for required integer properties of the first particle
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_int_props_a(const ArgumentNameSet<INT> &props) {
    this->required_int_props_a = props;
    this->index_on_device_object();
  }

  /**
   * @brief Return all required integer properties for the second particle
   *
   */
  ArgumentNameSet<INT> get_required_int_props_b() {
    return this->required_int_props_b;
  }

  /**
   * @brief Setter for required integer properties of the second particle
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_int_props_b(const ArgumentNameSet<INT> &props) {
    this->required_int_props_b = props;
    this->index_on_device_object();
  }

  /**
   * @brief Return all required integer properties of the first particle as a
   * vector of Syms
   *
   */
  std::vector<Sym<INT>> get_required_int_sym_vector_a() {
    return this->required_int_props_a.to_sym_vector();
  }

  /**
   * @brief Return all required integer properties of the second particle as a
   * vector of Syms
   *
   */
  std::vector<Sym<INT>> get_required_int_sym_vector_b() {
    return this->required_int_props_b.to_sym_vector();
  }

  /**
   * @brief Return all required real properties of the first particle
   *
   */
  ArgumentNameSet<REAL> get_required_real_props_a() {
    return this->required_real_props_a;
  }

  /**
   * @brief Return all required real properties of the first particle as a
   * vector of Syms
   *
   */
  std::vector<Sym<REAL>> get_required_real_sym_vector_a() {
    return this->required_real_props_a.to_sym_vector();
  }

  /**
   * @brief Setter for required real properties of the first particle
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_real_props_a(const ArgumentNameSet<REAL> &props) {
    this->required_real_props_a = props;
    this->index_on_device_object();
  }

  /**
   * @brief Return all required real properties of the second particle
   *
   */
  ArgumentNameSet<REAL> get_required_real_props_b() {
    return this->required_real_props_b;
  }

  /**
   * @brief Return all required real properties of the second particle as a
   * vector of Syms
   *
   */
  std::vector<Sym<REAL>> get_required_real_sym_vector_b() {
    return this->required_real_props_b.to_sym_vector();
  }

  /**
   * @brief Setter for required real properties of the second particle
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_real_props_b(const ArgumentNameSet<REAL> &props) {
    this->required_real_props_b = props;
    this->index_on_device_object();
  }

  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

  virtual ~PairReactionDataBase() = default;

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object() {};

  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */
  const ON_DEVICE_TYPE &get_on_device_obj() {

    NESOASSERT(this->on_device_obj.has_value(),
               "on_device_obj in PairReactionDataBase not initialised");
    return this->on_device_obj.value();
  }

protected:
  std::optional<ON_DEVICE_TYPE> on_device_obj;
  ArgumentNameSet<INT> required_int_props_a;
  ArgumentNameSet<REAL> required_real_props_a;
  ArgumentNameSet<INT> required_int_props_b;
  ArgumentNameSet<REAL> required_real_props_b;
  std::shared_ptr<RNG_TYPE> rng_kernel;
  std::map<int, std::string> properties_map;
};

/**
 * @brief Base pair reaction data object to be used on SYCL devices.
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_TYPE Sets the type of RNG that is used for sampling (Optional).
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines) (Optional, default 0)
 * @tparam VAL_TYPE Return type of this objects calc_data routine (Optional,
 * default REAL)
 * @tparam IN_TYPE Input type of array required by this object (if input_dim >0)
 */
template <size_t dim = 1, typename RNG_TYPE = DEFAULT_RNG_KERNEL,
          size_t input_dim = 0, typename VAL_TYPE = REAL,
          typename IN_TYPE = REAL>
struct PairReactionDataBaseOnDevice {
  using RNG_KERNEL_TYPE = RNG_TYPE;
  using VALUE_TYPE = VAL_TYPE;
  using INPUT_TYPE = IN_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  PairReactionDataBaseOnDevice() = default;

  /**
   * @brief Function to calculate the reaction data.
   *
   * @param index Read-only accessor to a pair loop index for a ParticlePairLoop
   * inside which calc_data is called. Access using
   * index.get_loop_linear_index().
   * @param req_int_props_a Vector of symbols for integer-valued properties of
   * the first particle that need to be used for the reaction data calculation.
   * @param req_real_props_a Vector of symbols for real-valued properties of the
   * first particle that need to be used for the reaction data calculation.
   * @param req_int_props_b Vector of symbols for integer-valued properties of
   * the second particle that need to be used for the reaction data calculation.
   * @param req_real_props_b Vector of symbols for real-valued properties of the
   * second particle that need to be used for the reaction data calculation.
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation
   *
   * @return A REAL-valued array of size dim containing the calculated reaction
   * rate.
   */
  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D == 0) && D == INPUT_DIM, int> = 0>
  std::array<VAL_TYPE, dim>
  calc_data(const Access::PairLoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props_a,
            const Access::SymVector::Read<REAL> &req_real_props_a,
            const Access::SymVector::Write<INT> &req_int_props_b,
            const Access::SymVector::Read<REAL> &req_real_props_b,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }

  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D > 0) && D == INPUT_DIM, int> = 0>
  std::array<VAL_TYPE, dim>
  calc_data(const std::array<IN_TYPE, INPUT_DIM> &input,
            const Access::PairLoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props_a,
            const Access::SymVector::Read<REAL> &req_real_props_a,
            const Access::SymVector::Write<INT> &req_int_props_b,
            const Access::SymVector::Read<REAL> &req_real_props_b,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }
  static constexpr size_t get_dim() { return dim; }
};

}; // namespace VANTAGE::Reactions
#endif
