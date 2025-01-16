#pragma once
#include "reaction_data.hpp"
#include "utils.hpp"
#include <neso_particles.hpp>
#include <neso_particles/compute_target.hpp>
#include <neso_particles/typedefs.hpp>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace NESO::Particles;

namespace Reactions {

/**
 * struct AbstractDataCalculator - Dummy struct to derive DataCalculator from
 * for the purposes of type-checking of DataCalculator (when it's passed as a
 * typename template parameter - see LinearReactionBase)
 */
struct AbstractDataCalculator {};

/**
 * struct DataCalculator - Static container class for ReactionData objects
 *
 * @tparam DATATYPE ReactionData types
 */
template <typename... DATATYPE>
struct DataCalculator : public AbstractDataCalculator {

  DataCalculator() {
    static_assert(
        sizeof...(DATATYPE) == 0,
        "particle_spec is required to be passed for this constructor if "
        "non-zero number of ReactionData objects are being passed as well.");
  };

  DataCalculator(ParticleSpec particle_spec, DATATYPE... data)
      : particle_spec(particle_spec), data(std::make_tuple(data...)) {

    size_t type_check_counter = 0u;
    (
        [&] {
          static_assert(
              std::is_base_of_v<
                  ReactionDataBase<data.get_dim(),
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
                this->data_loop_int_syms.push_back(utils::build_sym_vector<INT>(
                    this->particle_spec, args.get_required_int_props()));

                this->data_loop_real_syms.push_back(
                    utils::build_sym_vector<REAL>(
                        this->particle_spec, args.get_required_real_props()));
                dat_idx++;
              }(),
              ...);
        },
        this->data);
  }

  /**
   * @brief Fills an NDLocalArray buffer by invoking the stored ReactionData
   * objects for a given cell index
   *
   * @param buffer NDLocalArray buffer - size should conform to the stored
   * ReactionData tuple size
   * @param particle_sub_group Particle subgroup used to fill out the buffer
   * @param cell_idx Cell index for which to invoke the corresponding particle
   * loops
   */
  void fill_buffer(const NDLocalArraySharedPtr<REAL, 2> &buffer,
                   ParticleSubGroupSharedPtr particle_sub_group, INT cell_idx) {
    NESOASSERT(buffer->index.shape[1] == this->get_data_size(),
               "Buffer size in fill_buffer does not correspond to the number "
               "data calculation objects.");
    std::apply(
        [&](auto &&...args) {
          size_t dat_idx = 0u;
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
                        buffer.at(current_count, dat_idx + i) = rate[i];
                      }
                    },
                    Access::read(ParticleLoopIndex{}),
                    Access::read(sym_vector<INT>(
                        particle_sub_group, this->data_loop_int_syms[dat_idx])),
                    Access::read(
                        sym_vector<REAL>(particle_sub_group,
                                         this->data_loop_real_syms[dat_idx])),
                    Access::write(buffer), Access::read(args.get_rng_kernel()));

                loop->execute(cell_idx);
                dat_idx += data_dim;
              }(),
              ...);
        },
        this->data);
  }

  /**
   * @brief Getter for the size of the stored ReactionData tuple
   */
  constexpr size_t get_data_size() const {
    size_t dat_idx = 0u;
    std::apply(
        [&](auto &&...args) {
          (
              [&] {
                constexpr auto data_dim = args.get_dim();
                dat_idx += data_dim;
              }(),
              ...);
        },
        this->data);
    return dat_idx;
  }

private:
  std::tuple<DATATYPE...> data;
  std::vector<std::vector<Sym<INT>>> data_loop_int_syms;
  std::vector<std::vector<Sym<REAL>>> data_loop_real_syms;
  ParticleSpec particle_spec;
};
} // namespace Reactions
