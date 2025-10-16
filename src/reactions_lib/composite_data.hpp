#ifndef REACTIONS_COMPOSITE_DATA_H
#define REACTIONS_COMPOSITE_DATA_H
#include "reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Base composite reaction data class on device. Contains another
 * on-device reaction data object which it can call, referred to as the child
 * object. Any RNG kernels are expected as a TupleRNG, where the first entry is
 * the kernel to be used by the parent object of the composite and the second is
 * the kernel to be passed on to the child object.
 *
 * @tparam CHILD_TYPE The type of the child objects
 * @tparam dim The reaction data result size
 * @tparam RNG_TYPE The RNG kernel type used by the parent object
 */
template <typename CHILD_TYPE, size_t dim = 1,
          typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct BaseCompositeDataOnDevice
    : public ReactionDataBaseOnDevice<
          dim,
          TupleRNG<std::shared_ptr<RNG_TYPE>,
                   std::shared_ptr<typename CHILD_TYPE::RNG_KERNEL_TYPE>>> {

  /**
   * @brief Function to calculate the composite reaction data.
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction data calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction data calculation.
   * @param rng_kernel The TupleRNG accessor whose elements are used in the
   * calculation
   *
   * @return A REAL-valued array of size dim containing the calculated reaction
   * data.
   */
  std::array<REAL, dim> calc_data(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<std::shared_ptr<RNG_TYPE>,
                        std::shared_ptr<typename CHILD_TYPE::RNG_KERNEL_TYPE>>::
          KernelType &rng_kernel) const {

    return std::array<REAL, dim>{0.0};
  }

  /**
   * @brief Return the evaluation result of the child data
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction data calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction data calculation.
   * @param rng_kernel The TupleRNG accessor whose elements are used in the
   * calculation (uses the second entry)
   *
   * @return A REAL-valued array of size dim containing the calculated child
   * reaction data.
   */
  std::array<REAL, CHILD_TYPE::DIM> calc_data_child(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename TupleRNG<std::shared_ptr<RNG_TYPE>,
                        std::shared_ptr<typename CHILD_TYPE::RNG_KERNEL_TYPE>>::
          KernelType &rng_kernel) const {

    return this->child_data.calc_data(index, req_int_props, req_real_props,
                                      rng_kernel.template get<1>());
  }

public:
  CHILD_TYPE child_data;
};

/**
 * @brief Host base composite data class containing another reaction data child
 * object, allowing for composing computations.
 *
 * @tparam CHILD_TYPE Host type of the child object
 * @tparam ON_DEVICE_OBJ The on-device type of the contained object (expected to
 * be derived from the composite on device type templated on
 * CHILD_TYPE::ON_DEVICE_OBJ_TYPE)
 * @tparam dim The dimensionality of the calculated data
 * @tparam RNG_TYPE The rng kernel type used for the parent object. Combined
 * into a TupleRNG for passing on to the child.
 */
template <typename CHILD_TYPE, typename ON_DEVICE_OBJ, size_t dim = 1,
          typename RNG_TYPE = HostPerParticleBlockRNG<REAL>>
struct BaseCompositeData
    : public ReactionDataBase<
          ON_DEVICE_OBJ, dim,
          TupleRNG<std::shared_ptr<RNG_TYPE>,
                   std::shared_ptr<typename CHILD_TYPE::RNG_KERNEL_TYPE>>> {

  using ON_DEVICE_OBJ_TYPE = ON_DEVICE_OBJ;
  /**
   * @brief Constructor for BaseCompositeData.
   *
   * @param child_data The contained child object
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing
   * information regarding the required REAL-based properties for the
   * reaction data.
   * @param required_int_props_ephemeral Properties<INT> object containing
   * information regarding the required INT-based ephemeral properties for
   * the reaction data.
   * @param required_real_props_ephemeral Properties<REAL> object containing
   * information regarding the required REAL-based ephemeral properties for
   * the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to
   * be used when remapping property names (in get_required_real_props(...)
   * and get_required_int_props(...)).
   */
  BaseCompositeData(
      CHILD_TYPE child_data, Properties<INT> required_int_props,
      Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral,
      Properties<REAL> required_real_props_ephemeral,
      std::map<int, std::string> properties_map = get_default_map())
      : ReactionDataBase<
            ON_DEVICE_OBJ, dim,
            TupleRNG<std::shared_ptr<RNG_TYPE>,
                     std::shared_ptr<typename CHILD_TYPE::RNG_KERNEL_TYPE>>>(
            required_int_props, required_real_props,
            required_int_props_ephemeral, required_real_props_ephemeral,
            properties_map),
        child_data(child_data) {

    auto req_int_set = this->get_required_int_props();
    req_int_set =
        req_int_set.merge_with(this->child_data.get_required_int_props());
    this->set_required_int_props(req_int_set);

    auto req_real_set = this->get_required_real_props();
    req_real_set =
        req_real_set.merge_with(this->child_data.get_required_real_props());
    this->set_required_real_props(req_real_set);

    this->root_rng_kernel = std::make_shared<RNG_TYPE>();

    this->set_rng_kernel(
        tuple_rng(this->root_rng_kernel, this->child_data.get_rng_kernel()));
  }

  /**
   * \overload
   * @brief Constructor for BaseCompositeData that sets no required properties.
   *
   * @param child_data The contained child object
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  BaseCompositeData(
      CHILD_TYPE child_data,
      std::map<int, std::string> properties_map = get_default_map())
      : BaseCompositeData(child_data, Properties<INT>(), Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

  /**
   * \overload
   * @brief Constructor for BaseCompositeData that sets only required int
   * properties.
   *
   * @param child_data The contained child object
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  BaseCompositeData(
      CHILD_TYPE child_data, Properties<INT> required_int_props,
      std::map<int, std::string> properties_map = get_default_map())
      : BaseCompositeData(child_data, required_int_props, Properties<REAL>(),
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

  /**
   * \overload
   * @brief Constructor for BaseCompositeData that sets only required real
   * properties.
   *
   * @param child_data The contained child object
   * @param required_int_props Properties<INT> object containing information
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  BaseCompositeData(
      const CHILD_TYPE &child_data, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : BaseCompositeData(child_data, Properties<INT>(), required_real_props,
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

  /**
   * \overload
   * @brief Constructor for BaseCompositeData that sets only required int and
   * real properties.
   *
   * @param child_data The contained child object
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties for the reaction data.
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  BaseCompositeData(
      CHILD_TYPE child_data, Properties<INT> required_int_props,
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : BaseCompositeData(child_data, required_int_props, required_real_props,
                          Properties<INT>(), Properties<REAL>(),
                          properties_map) {}

  virtual void index_on_device_object_root() {};

  /**
   * @brief Resets the on-device object and calls the root/parent object
   * indexing function. Assumes that the child object's index_on_device_object
   * is called, for example in one of the property setters.
   */
  void index_on_device_object() {

    this->on_device_obj->child_data = this->child_data.get_on_device_obj();

    this->index_on_device_object_root();
  };

  /**
   * @brief Setter for required integer properties. Recursively sets the child
   * properties as well and performs indexing.
   *
   * @param props Name set for the used arguments
   */
  void set_required_int_props(const ArgumentNameSet<INT> &props) {
    this->required_int_props = props;
    this->child_data.set_required_int_props(props);
    this->index_on_device_object();
  }

  /**
   * @brief Setter for required real properties. Recursively sets the child
   * properties as well and performs indexing.
   *
   * @param props Name set for the used arguments
   */
  void set_required_real_props(const ArgumentNameSet<REAL> &props) {
    this->required_real_props = props;
    this->child_data.set_required_real_props(props);
    this->index_on_device_object();
  }

  /**
   * @brief Sets the root/parent rng kernel. Reconstructs the rng_kernel object
   * as the required tuple.
   *
   * @param rng_kernel Root/parent rng kernel to use
   */
  void set_root_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->root_rng_kernel = rng_kernel;
    this->set_rng_kernel(
        tuple_rng(this->root_rng_kernel, this->child_data.get_rng_kernel()));
  }

private:
  CHILD_TYPE child_data;
  std::shared_ptr<RNG_TYPE> root_rng_kernel;
};
}; // namespace VANTAGE::Reactions
#endif
