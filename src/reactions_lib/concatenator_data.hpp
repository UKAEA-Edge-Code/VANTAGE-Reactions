#ifndef REACTIONS_CONCATENATOR_DATA_H
#define REACTIONS_CONCATENATOR_DATA_H
#include "reaction_data.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

template <typename T> constexpr size_t total_dim() { return T::DIM; };
template <typename T, typename U, typename... DATATYPE>
constexpr size_t total_dim() {
  return T::DIM + total_dim<U, DATATYPE...>();
};

template <typename... DATATYPE>
struct ConcatenatorDataOnDevice
    : public ReactionDataBaseOnDevice<total_dim<DATATYPE...>()> {

  ConcatenatorDataOnDevice(DATATYPE... data)
      : data(Tuple::to_tuple(data...)) {};

  static const size_t DIM = total_dim<DATATYPE...>();

  /**
   * @brief Function to calculate the concatenated data
   *
   * @param index Read-only accessor to a loop index for a ParticleLoop
   * inside which calc_data is called. Access using either
   * index.get_loop_linear_index(), index.get_local_linear_index(),
   * index.get_sub_linear_index() as required.
   * @param req_int_props Vector of symbols for integer-valued properties that
   * need to be used for the reaction rate calculation.
   * @param req_real_props Vector of symbols for real-valued properties that
   * need to be used for the reaction rate calculation.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return Concatenated return arrays of all the contained device types
   */
  std::array<REAL, DIM>
  calc_data(const Access::LoopIndex::Read &index,
            const Access::SymVector::Write<INT> &req_int_props,
            const Access::SymVector::Read<REAL> &req_real_props,
            typename ReactionDataBaseOnDevice<DIM>::RNG_KERNEL_TYPE::KernelType
                &rng_kernel) const {

    std::array<REAL, DIM> result;

    calc_data_recurse<0>(index, req_int_props, req_real_props, rng_kernel,
                         result, 0);
    return result;
  }

  template <std::size_t I>
  void calc_data_recurse(
      const Access::LoopIndex::Read &index,
      const Access::SymVector::Write<INT> &req_int_props,
      const Access::SymVector::Read<REAL> &req_real_props,
      typename ReactionDataBaseOnDevice<DIM>::RNG_KERNEL_TYPE::KernelType
          &rng_kernel,
      std::array<REAL, DIM> &result, size_t dat_dim_idx) const {

    if constexpr (I < (sizeof...(DATATYPE))) {

      const auto arg = Tuple::get<I>(this->data);
      constexpr auto data_dim = decltype(arg)::DIM;
      std::array<REAL, data_dim> calculated_data =
          arg.calc_data(index, req_int_props, req_real_props, rng_kernel);
      for (auto i = 0; i < data_dim; i++) {
        result[dat_dim_idx + i] = calculated_data[i];
      };

      this->calc_data_recurse<I + 1>(index, req_int_props, req_real_props,
                                     rng_kernel, result,
                                     dat_dim_idx + data_dim);
    };
  }

private:
  Tuple::Tuple<DATATYPE...> data;
};

template <template <class> class ON_DEVICE_TYPE, typename... ARGS>
struct OnDeviceTemplate {

  using type = ON_DEVICE_TYPE<typename ARGS::ON_DEVICE_OBJ_TYPE...>;
};

template <typename... DATATYPE>
inline std::tuple<typename DATATYPE::ON_DEVICE_OBJ_TYPE...>
get_on_device_objs(std::tuple<DATATYPE...> &data) {

  return std::apply(
      [](auto &&...args) { return std::tuple(args.get_on_device_obj()...); },
      data);
};

template <typename... DATATYPE>
struct ConcatenatorData
    : public ReactionDataBase<typename OnDeviceTemplate<
                                  ConcatenatorDataOnDevice, DATATYPE...>::type,
                              total_dim<DATATYPE...>()> {

  ConcatenatorData(DATATYPE... data) : data(std::make_tuple(data...)) {
    this->set_required_int_props(this->get_required_int_props_children());
    this->set_required_real_props(this->get_required_real_props_children());
  };

  void index_on_device_object() {

    this->on_device_obj = std::make_from_tuple<
        typename OnDeviceTemplate<ConcatenatorDataOnDevice, DATATYPE...>::type>(
        get_on_device_objs(this->data));
  };

  ArgumentNameSet<REAL> get_required_real_props_children() {

    auto new_set = ArgumentNameSet<REAL>();

    std::apply(
        [&](auto &&...args) {
          ((new_set.merge_with(args.get_required_real_props())), ...);
        },
        this->data);

    return new_set;
  }

  ArgumentNameSet<INT> get_required_int_props_children() {

    auto new_set = ArgumentNameSet<INT>();

    std::apply(
        [&](auto &&...args) {
          ((new_set.merge_with(args.get_required_int_props())), ...);
        },
        this->data);

    return new_set;
  }

  void set_required_int_props(const ArgumentNameSet<INT> &props) {
    this->required_int_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_int_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

  void set_required_real_props(const ArgumentNameSet<REAL> &props) {
    this->required_real_props = props;
    std::apply(
        [&](auto &&...args) { ((args.set_required_real_props(props)), ...); },
        this->data);
    this->index_on_device_object();
  }

private:
  std::tuple<DATATYPE...> data;
};
}; // namespace VANTAGE::Reactions
#endif
