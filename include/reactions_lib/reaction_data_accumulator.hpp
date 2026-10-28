#ifndef REACTIONS_REACTION_DATA_ACCUMULATOR_H
#define REACTIONS_REACTION_DATA_ACCUMULATOR_H
#include "reaction_data.hpp"
#include "reaction_data_accumulator_base.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include "transformation_wrapper.hpp"
#include <memory>

namespace VANTAGE::Reactions {
/**
 * @brief Transformation evaluating a ReactionData object and reducing the
 * results cellwise
 *
 * @tparam ReactionData Type of object whose outputs are to be reduced cellwise
 * (must be derived from ReactionDataBase).
 */
template <typename ReactionData>
struct CellwiseReactionDataAccumulator
    : CellwiseReactionDataAccumulatorBase<
          typename ReactionData::ON_DEVICE_OBJ_TYPE::VALUE_TYPE> {

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
      : CellwiseReactionDataAccumulatorBase<
            typename ReactionData::ON_DEVICE_OBJ_TYPE::VALUE_TYPE>(
            template_group,
            reaction_data.get_arg_pack().required_int_props.to_sym_vector(),
            reaction_data.get_arg_pack().required_real_props.to_sym_vector(),
            ReactionData::DIM),
        reaction_data(reaction_data) {

    static_assert(
        std::is_base_of_v<
            AbstractReactionData<typename ReactionData::ON_DEVICE_OBJ_TYPE,
                                 typename ReactionData::ARGUMENT_PACK_TYPE,
                                 ReactionData::DIM,
                                 typename ReactionData::RNG_KERNEL_TYPE>,
            ReactionData>,
        "Template parameter ReactionData is not derived from "
        "AbstractReactionData...");
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
          auto accessors = SingleReactionDataAccessors{
              particle_index, req_int_props, req_real_props};
          std::array<REAL, data_dim> data =
              reaction_data_on_device.calc_data(accessors, kernel);

          for (auto j = 0; j < data_dim; j++) {
            buffer.combine(j, 0, data[j]);
          }
        },
        NP::Access::reduce(this->values, NP::Kernel::plus<REAL>()),
        NP::Access::read(NP::ParticleLoopIndex{}),
        NP::Access::write(
            NP::sym_vector<INT>(target_subgroup, this->required_int_syms)),
        NP::Access::read(
            NP::sym_vector<REAL>(target_subgroup, this->required_real_syms)),
        NP::Access::read(this->reaction_data.get_rng_kernel()));

    loop->execute();
  }

private:
  ReactionData reaction_data;
};
} // namespace VANTAGE::Reactions
#endif
