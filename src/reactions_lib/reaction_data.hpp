#ifndef REACTIONS_REACTION_DATA_H
#define REACTIONS_REACTION_DATA_H
#include "reaction_kernel_pre_reqs.hpp"
#include <memory>
#include <neso_particles.hpp>
#include <stdexcept>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief An abstract base class for cross-section objects.
 * All classes derived from this class should be device copyable in order to be
 * used within ReactionData classes.
 */
struct AbstractCrossSection {

  /**
   * @brief Get the value of the cross section for a given relative velocity
   * value of projectile and target
   *
   * @param relative_vel Magnitude of relative velocity of target and projectile
   * @return REAL-valued cross-section at requested relative vel magnitude
   */
  REAL get_value_at(const REAL &relative_vel) const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
  };

  /**
   * @brief Get the maximum value of sigma*v_r where sigma is this cross-section
   * evaluated at v_r and v_r is the relative speed of the projectile and target
   *
   * @return REAL-valued maximum rate
   */
  REAL get_max_rate_val() const {
    // This function should never actually be called. If it is called and we do
    // not have a return value then the calling function will receive an
    // undefined value. By setting a value we at least know what the returned
    // value is and can pick one that is detectable.
    return std::numeric_limits<REAL>::lowest();
  };

  /**
   * @brief Accept-reject function for when this cross-section is used in
   * rejection methods. Accepts if the uniform random number on (0,1) is less
   * than the ratio of sigma*v evaluated at a given relative speed to the
   * maximum value of sigma*v.
   *
   * @param relative_vel Magnitude of relative velocity of the projectile and
   * target
   * @param uniform_rand Uniformly sampled random number on (0,1)
   * @param value_at Value of cross section for a given relative velocity value
   * of projectile and target (NOTE this is currently a workaround due to the
   * limitation on calling get_value_at(...) inside this function.)
   * @param max_rate_val Maximum value of sigma*v_r (NOTE this is currently a
   * workaround due to the limitation on calling get_max_rate_val(...) inside
   * this function.)
   * @return true if relative_vel value is accepted, false otherwise
   */
  bool accept_reject(REAL relative_vel, REAL uniform_rand, REAL value_at,
                     REAL max_rate_val) const {
    return uniform_rand < (value_at * relative_vel / max_rate_val);
  }
};

using DEFAULT_RNG_KERNEL = HostPerParticleBlockRNG<REAL>;
/**
 * @brief Base reaction data object.
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
struct ReactionDataBase {

  using RNG_KERNEL_TYPE = RNG_TYPE;
  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  /**
   * @brief Constructor for ReactionDataBase.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param required_int_props_ephemeral Properties<INT> object containing
   * information regarding the required INT-based ephemeral properties for the
   * reaction data.
   * @param required_real_props_ephemeral Properties<REAL> object containing
   * information regarding the required REAL-based ephemeral properties for the
   * reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      std::map<int, std::string> properties_map = get_default_map())
      : required_int_props(
            ArgumentNameSet(required_int_props, properties_map)
                .merge_with(ArgumentNameSet(required_int_props_ephemeral,
                                            properties_map))),
        required_real_props(
            ArgumentNameSet(required_real_props, properties_map)
                .merge_with(ArgumentNameSet(required_real_props_ephemeral,
                                            properties_map))),
        properties_map(properties_map) {

    this->rng_kernel = std::make_shared<RNG_TYPE>();
  }

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets not required properties.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<INT>(), Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int
   * properties.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, Properties<REAL>(),
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required real
   * properties.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(Properties<INT>(), required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}

  /**
   * \overload
   * @brief Constructor for ReactionDataBase that sets only required int and
   * real properties.
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  ReactionDataBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase(required_int_props, required_real_props,
                         Properties<INT>(), Properties<REAL>(),
                         properties_map) {}
  /**
   * @brief Return all required integer properties, including ephemeral
   *
   */
  ArgumentNameSet<INT> get_required_int_props() {
    return this->required_int_props;
  }

  /**
   * @brief Setter for required integer properties
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_int_props(const ArgumentNameSet<INT> &props) {
    this->required_int_props = props;
    this->index_on_device_object();
  }

  /**
   * @brief Return all required integer properties as a vector of Syms
   *
   */
  std::vector<Sym<INT>> get_required_int_sym_vector() {
    return this->required_int_props.to_sym_vector();
  }

  /**
   * @brief Return all required real properteis, including ephemeral
   * properties
   *
   */
  ArgumentNameSet<REAL> get_required_real_props() {
    return this->required_real_props;
  }

  /**
   * @brief Return all required real properties as a vector of Syms
   *
   */
  std::vector<Sym<REAL>> get_required_real_sym_vector() {
    return this->required_real_props.to_sym_vector();
  }

  /**
   * @brief Setter for required real properties
   *
   * @param props ArgumentNameSet to use
   */
  void set_required_real_props(const ArgumentNameSet<REAL> &props) {
    this->required_real_props = props;
    this->index_on_device_object();
  }

  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  static constexpr size_t get_dim() { return dim; }

  virtual ~ReactionDataBase<ON_DEVICE_TYPE, dim, RNG_TYPE, input_dim>() =
      default;

  /**
   * @brief To be implemented by each derived class in order to handle required
   * property indexing on the on-device object
   */
  virtual void index_on_device_object() {};

  /**
   * @brief Getter for the SYCL device-specific
   * struct.
   */
  ON_DEVICE_TYPE get_on_device_obj() {

    NESOASSERT(this->on_device_obj.has_value(),
               "on_device_obj in ReactionDataBase not initialised");
    return this->on_device_obj.value();
  }

protected:
  std::optional<ON_DEVICE_TYPE> on_device_obj;
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;
  std::shared_ptr<RNG_TYPE> rng_kernel;
  std::map<int, std::string> properties_map;
};

/**
 * @brief Base reaction data object to be used on SYCL devices.
 *
 * @tparam dim Used to set the size of the array that calc_data returns
 * (Optional).
 * @tparam RNG_TYPE Sets the type of RNG that is used for sampling (Optional).
 * @tparam input_dim The dimension of the optional input array (for use in
 * pipelines)
 */
template <size_t dim = 1, typename RNG_TYPE = DEFAULT_RNG_KERNEL,
          size_t input_dim = 0>
struct ReactionDataBaseOnDevice {
  using RNG_KERNEL_TYPE = RNG_TYPE;
  static const size_t DIM = dim;
  static const size_t INPUT_DIM = input_dim;

  ReactionDataBaseOnDevice() = default;

  /**
   * @brief Function to calculate the reaction data.
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction data calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction data calculation.
   * @param rng_kernel The random number generator kernel potentially used in
   * the calculation
   *
   * @return A REAL-valued array of size dim containing the calculated reaction
   * rate.
   */
  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D == 0) && D == INPUT_DIM, int> = 0>
  std::array<REAL, dim>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }

  template <std::size_t D = INPUT_DIM,
            std::enable_if_t<(D > 0) && D == INPUT_DIM, int> = 0>
  std::array<REAL, dim>
  calc_data(const std::array<REAL, INPUT_DIM> &input,
            const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename RNG_TYPE::KernelType &rng_kernel) const {
    return std::array<REAL, dim>{0.0};
  }
  static constexpr size_t get_dim() { return dim; }
};
}; // namespace VANTAGE::Reactions
#endif
