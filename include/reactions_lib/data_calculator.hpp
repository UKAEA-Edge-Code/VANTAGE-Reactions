#ifndef REACTIONS_DATA_CALCULATOR_H
#define REACTIONS_DATA_CALCULATOR_H
#include "../reactions/neso_particles_namespace_alias.hpp"
#include "../reactions/neso_test_assert.hpp"
#include "reaction_data.hpp"

#include <tuple>
#include <type_traits>
#include <vector>

namespace VANTAGE::Reactions {

/**
 * @brief A dummy struct to derive DataCalculator from
 * for the purposes of type-checking of DataCalculator (when it's passed as a
 * typename template parameter - see LinearReactionBase)
 */
struct AbstractDataCalculator {
  virtual ~AbstractDataCalculator() = default;
};

/**
 * @brief A static container class for ReactionData objects
 *
 * @tparam DATATYPE ReactionData types
 */
template <typename... DATATYPE>
struct DataCalculator : public AbstractDataCalculator {

  /**
   * @brief Constructor for DataCalculator.
   *
   * @param data List of ReactionData objects (as multiple arguments).
   */
  DataCalculator(DATATYPE... data) : data(std::make_tuple(data...)) {

    size_t type_check_counter = 0u;
    (
        [&] {
          static_assert(
              std::is_base_of_v<
                  ReactionDataBase<typename decltype(data)::ON_DEVICE_OBJ_TYPE,
                                   data.get_dim(),
                                   typename decltype(data)::RNG_KERNEL_TYPE>,
                  decltype(data)>,
              "DATATYPE provided is not derived from ReactionDataBase.");
          type_check_counter++;
        }(),
        ...);

    std::apply(
        [&](auto &&...args) {
          size_t dat_idx = 0u;
          (
              [&] {
                this->data_loop_int_syms.push_back(
                    args.get_required_int_sym_vector());

                this->data_loop_real_syms.push_back(
                    args.get_required_real_sym_vector());
                dat_idx++;
              }(),
              ...);
        },
        this->data);
  }

  /**
   * @brief Fills an NP::NDLocalArray buffer by invoking the stored ReactionData
   * objects for a given cell index
   *
   * @param buffer NP::NDLocalArray buffer - size should conform to the stored
   * ReactionData tuple size
   * @param particle_sub_group Particle subgroup used to fill out the buffer
   * @param cell_idx_start Starting cell index for which to invoke the
   * corresponding particle loops
   * @param cell_idx_end Ending cell index for which to invoke the corresponding
   * particle loops
   */
  void fill_buffer(const NP::NDLocalArraySharedPtr<REAL, 2> &buffer,
                   NP::ParticleSubGroupSharedPtr particle_sub_group,
                   INT cell_idx_start, INT cell_idx_end) {
    NESOASSERT(buffer->index.shape[1] == this->get_data_size(),
               "Buffer size in fill_buffer does not correspond to the number "
               "data calculation objects.");
    std::apply(
        [&](auto &&...args) {
          size_t dat_idx = 0u;
          size_t dat_dim_idx = 0u;
          (
              [&] {
                auto reaction_data_on_device = args.get_on_device_obj();
                // Maybe make into a vector of loop shared_ptrs and use submit
                // instead of execute
                constexpr auto data_dim = reaction_data_on_device.get_dim();
                auto loop = particle_loop(
                    "data_calc_loop", particle_sub_group,
                    [=](auto particle_index, auto req_int_props,
                        auto req_real_props, auto buffer, auto kernel) {
                      INT current_count =
                          particle_index.get_loop_linear_index();
                      std::array<REAL, data_dim> rate =
                          reaction_data_on_device.calc_data(
                              particle_index, req_int_props, req_real_props,
                              kernel);
                      for (auto i = 0; i < data_dim; i++) {
                        buffer.at(current_count, dat_dim_idx + i) = rate[i];
                      }
                    },
                    NP::Access::read(NP::ParticleLoopIndex{}),
                    NP::Access::write(NP::sym_vector<INT>(
                        particle_sub_group, this->data_loop_int_syms[dat_idx])),
                    NP::Access::read(NP::sym_vector<REAL>(
                        particle_sub_group,
                        this->data_loop_real_syms[dat_idx])),
                    NP::Access::write(buffer),
                    NP::Access::read(args.get_rng_kernel()));

                loop->execute(cell_idx_start, cell_idx_end);
                dat_idx++;
                dat_dim_idx += data_dim;
              }(),
              ...);
        },
        this->data);
  }

  /**
   * @brief Getter for the total number of dimensions of the objects in the
   * ReactionData tuple
   */
  size_t get_data_size() const {
    size_t dat_idx = 0u;
    std::apply(
        [&](auto &&...args) {
          (
              [&] {
                const auto data_dim = args.get_dim();
                dat_idx += data_dim;
              }(),
              ...);
        },
        this->data);
    return dat_idx;
  }

  /**
   * @brief Getter of the total number of objects in the ReactionData tuple
   */
  size_t get_data_tuple_size() const { return sizeof...(DATATYPE); }

private:
  std::tuple<DATATYPE...> data;
  std::vector<std::vector<NP::Sym<INT>>> data_loop_int_syms;
  std::vector<std::vector<NP::Sym<REAL>>> data_loop_real_syms;
};
} // namespace VANTAGE::Reactions
#endif
