#ifndef REACTIONS_PAIR_DATA_CALCULATOR_H
#define REACTIONS_PAIR_DATA_CALCULATOR_H
#include "pair_reaction_data.hpp"
#include "utils.hpp"
#include <neso_particles.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * @brief A dummy struct to derive PairDataCalculator froe
 * for the purposes of type-checking of PairDataCalculatoo (when it's passed as
 * a typename template parameter - see LinearReactionBase)
 */
struct AbstractPairDataCalculator {
  virtual ~AbstractPairDataCalculator() = default;
};

/**
 * @brief A static container class for PairReactionData objects
 *
 * @tparam DATATYPE PairReactionData types
 */
template <typename... DATATYPE>
struct PairDataCalculator : public AbstractPairDataCalculator {

  /**
   * @brief Constructor for PairDataCalculator.
   *
   * @param data List of PairReactionData objects (as multiple arguments).
   */
  PairDataCalculator(DATATYPE... data) : data(std::make_tuple(data...)) {

    size_t type_check_counter = 0u;
    (
        [&] {
          static_assert(
              std::is_base_of_v<
                  PairReactionDataBase<
                      typename decltype(data)::ON_DEVICE_OBJ_TYPE,
                      data.get_dim(), typename decltype(data)::RNG_KERNEL_TYPE>,
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
                this->data_loop_int_syms_a.push_back(
                    args.get_required_int_sym_vector_a());

                this->data_loop_real_syms_a.push_back(
                    args.get_required_real_sym_vector_a());

                this->data_loop_int_syms_b.push_back(
                    args.get_required_int_sym_vector_b());

                this->data_loop_real_syms_b.push_back(
                    args.get_required_real_sym_vector_b());
                dat_idx++;
              }(),
              ...);
        },
        this->data);
  }

  /**
   * @brief Fills an NDLocalArray buffer by invoking the stored PairReactionData
   * objects for a given cell index
   *
   * @param buffer NDLocalArray buffer - size should conform to the stored
   * PairReactionData tuple size
   * @param pair_list Particle pair list used to fill out the buffer
   * @param cell_idx_start Cell index from which to invoke the corresponding
   * particle loops
   * @param cell_idx_end Cell index to which to invoke the corresponding
   * particle loops
   */
  template <typename TARGET, typename PAIR_LIST>
  void fill_buffer(const NDLocalArraySharedPtr<REAL, 2> &buffer,
                   CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
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
                auto loop = particle_pair_loop(
                    "pair_data_calc_loop", {pair_list},
                    [=](auto pair_index, auto req_int_props_a,
                        auto req_real_props_a, auto req_int_props_b,
                        auto req_real_props_b, auto buffer, auto kernel) {
                      INT current_count = pair_index.get_loop_linear_index();
                      std::array<REAL, data_dim> rate =
                          reaction_data_on_device.calc_data(
                              pair_index, req_int_props_a, req_real_props_a,
                              req_int_props_b, req_real_props_b, kernel);
                      for (auto i = 0; i < data_dim; i++) {
                        buffer.at(current_count, dat_dim_idx + i) = rate[i];
                      }
                    },
                    Access::read(ParticlePairLoopIndex{}),
                    Access::A(Access::write(sym_vector<INT>(
                        pair_list.A, this->data_loop_int_syms_a[dat_idx]))),
                    Access::A(Access::read(sym_vector<REAL>(
                        pair_list.A, this->data_loop_real_syms_a[dat_idx]))),
                    Access::B(Access::write(sym_vector<INT>(
                        pair_list.B, this->data_loop_int_syms_b[dat_idx]))),
                    Access::B(Access::read(sym_vector<REAL>(
                        pair_list.B, this->data_loop_real_syms_b[dat_idx]))),
                    Access::write(buffer), Access::read(args.get_rng_kernel()));

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
  size_t get_data_tuple_size() const { return std::size(this->data); }

private:
  std::tuple<DATATYPE...> data;
  std::vector<std::vector<Sym<INT>>> data_loop_int_syms_a;
  std::vector<std::vector<Sym<REAL>>> data_loop_real_syms_a;
  std::vector<std::vector<Sym<INT>>> data_loop_int_syms_b;
  std::vector<std::vector<Sym<REAL>>> data_loop_real_syms_b;
};
} // namespace VANTAGE::Reactions
#endif
