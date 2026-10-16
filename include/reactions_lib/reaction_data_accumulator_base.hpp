#ifndef REACTIONS_REACTION_DATA_ACCUMULATOR_BASE_H
#define REACTIONS_REACTION_DATA_ACCUMULATOR_BASE_H
#include "reactions/neso_particles_namespace_alias.hpp"
#include "transformation_wrapper.hpp"
#include <memory>
#include <vector>

namespace VANTAGE::Reactions {
/**
 * @brief Base transformation strategy for cellwise accumulators. Stores the
 * particle properties required by the derived accumulator and the cellwise
 * accumulation buffer, providing a common interface so that all derived
 * accumulators can be held in a single container via pointers to this type.
 *
 * @tparam VALUE_TYPE The value type of the cellwise accumulation buffer
 * (REAL or INT)
 */
template <typename VALUE_TYPE>
struct CellwiseReactionDataAccumulatorBase : TransformationStrategy {

  CellwiseReactionDataAccumulatorBase() = delete;

  /**
   * @brief Constructor for CellwiseReactionDataAccumulatorBase.
   *
   * @param template_group A template particle group used to provide the
   * SYCL target and cell count for the created CellDatConst.
   * @param required_int_syms Integer-valued particle dats required by the
   * derived accumulator.
   * @param required_real_syms Real-valued particle dats required by the
   * derived accumulator.
   * @param data_dim Number of components accumulated per cell.
   */
  CellwiseReactionDataAccumulatorBase(
      NP::ParticleGroupSharedPtr template_group,
      std::vector<NP::Sym<INT>> required_int_syms,
      std::vector<NP::Sym<REAL>> required_real_syms, const int data_dim)
      : required_int_syms(required_int_syms),
        required_real_syms(required_real_syms),
        values(std::make_shared<NP::CellDatConst<VALUE_TYPE>>(
            template_group->sycl_target,
            template_group->domain->mesh->get_cell_count(), data_dim, 1)) {}

  /**
   * @brief Get the pointer to underlying NP::CellDatConst object
   *
   */
  NP::CellDatConstSharedPtr<VALUE_TYPE> get_value_pointer() {
    return this->values;
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<VALUE_TYPE>
   */
  void
  set_value_pointer(NP::CellDatConstSharedPtr<VALUE_TYPE> cell_dat_const_ptr) {
    this->values = cell_dat_const_ptr;
  }

  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * NP::CellData objects
   */
  std::vector<NP::CellData<VALUE_TYPE>> get_cell_data() {
    return this->values->get_all_cells();
  }

  /**
   * @brief Zero out the accumulation buffer
   */
  void zero_buffer() { this->values->fill(0); }

protected:
  std::vector<NP::Sym<INT>> required_int_syms;
  std::vector<NP::Sym<REAL>> required_real_syms;
  NP::CellDatConstSharedPtr<VALUE_TYPE> values;
};
} // namespace VANTAGE::Reactions
#endif
