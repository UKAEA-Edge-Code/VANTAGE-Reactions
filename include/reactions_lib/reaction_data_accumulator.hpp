#ifndef REACTIONS_REACTION_DATA_ACCUMULATOR_H
#define REACTIONS_REACTION_DATA_ACCUMULATOR_H
#include "reaction_data.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include "transformation_wrapper.hpp"
#include <memory>

#include <utility>

namespace VANTAGE::Reactions {
/**
 * @brief Transformation evaluating a ReactionData object and reducing the
 * results cellwise
 */
template <typename ReactionData>
struct CellwiseReactionDataAccumulator : TransformationStrategy {

  CellwiseReactionDataAccumulator() = delete;

  /**
   * @brief Constructor for CellwiseReactionDataAccumulator.
   *
   * @param template_group A template particle group used to provide the
   * CellDatConsts for the dats specified by dat_names.
   * @param reaction_data ReactionData whose outputs are to be reduced cellwise
   */
  CellwiseReactionDataAccumulator(NP::ParticleGroupSharedPtr template_group,
                                  ReactionData reaction_data)
      : reaction_data(reaction_data) {

    static_assert(
        std::is_base_of_v<
            ReactionDataBase<typename ReactionData::ON_DEVICE_OBJ_TYPE,
                             ReactionData::DIM,
                             typename ReactionData::RNG_KERNEL_TYPE>,
            ReactionData>,
        "Template parameter ReactionData is not derived from "
        "ReactionDataBase...");

    constexpr auto data_dim = ReactionData::DIM;
    this->values = std::make_shared<NP::CellDatConst<
        typename ReactionData::ON_DEVICE_OBJ_TYPE::VALUE_TYPE>>(
        template_group->sycl_target,
        template_group->domain->mesh->get_cell_count(), data_dim, 1);

    this->required_int_sums = this->reaction_data.get_required_int_sym_vector();
    this->required_real_syms =
        this->reaction_data.get_required_real_sym_vector();
  }
  /**
   * @brief Accumulate the results of evaluating the stored ReactionData object
   *
   * @param target_subgroup Subgroup containing particles whose dats should be
   * accumulated
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override {

    auto reaction_data_on_device = this->reaction_data.get_on_device_obj();

    // TODO: add sycl_target consistency test

    constexpr auto data_dim = ReactionData::DIM;

    auto loop = particle_loop(
        "CellwiseReactionDataAccumulator_loop", target_subgroup,
        [=](auto buffer, auto particle_index, auto req_int_props,
            auto req_real_props, auto kernel) {
          std::array<NP::REAL, data_dim> data =
              reaction_data_on_device.calc_data(particle_index, req_int_props,
                                                req_real_props, kernel);

          for (auto j = 0; j < data_dim; j++) {
            buffer.combine(j, 0, data[j]);
          }
        },
        NP::Access::reduce(this->values, NP::Kernel::plus<NP::REAL>()),
        NP::Access::read(NP::ParticleLoopIndex{}),
        NP::Access::write(
            NP::sym_vector<NP::INT>(target_subgroup, this->required_int_sums)),
        NP::Access::read(NP::sym_vector<NP::REAL>(target_subgroup,
                                                  this->required_real_syms)),
        NP::Access::read(this->reaction_data.get_rng_kernel()));

    loop->execute();
  }

  /**
   * @brief Get the pointer to underlying NP::CellDatConst object
   *
   */

  NP::CellDatConstSharedPtr<NP::REAL> get_value_pointer() {
    return this->values;
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<NP::REAL>
   */
  void
  set_value_pointer(NP::CellDatConstSharedPtr<NP::REAL> cell_dat_const_ptr) {

    this->values = cell_dat_const_ptr;
  }

  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * NP::CellData objects
   */
  std::vector<
      NP::CellData<typename ReactionData::ON_DEVICE_OBJ_TYPE::VALUE_TYPE>>
  get_cell_data() {

    return this->values->get_all_cells();
  }

  /**
   * @brief Zero out the accumulation buffer
   */
  void zero_buffer() { this->values->fill(0); }

private:
  ReactionData reaction_data;
  std::vector<NP::Sym<NP::INT>> required_int_sums;
  std::vector<NP::Sym<NP::REAL>> required_real_syms;

  std::shared_ptr<
      NP::CellDatConst<typename ReactionData::ON_DEVICE_OBJ_TYPE::VALUE_TYPE>>
      values;
};
} // namespace VANTAGE::Reactions
#endif
