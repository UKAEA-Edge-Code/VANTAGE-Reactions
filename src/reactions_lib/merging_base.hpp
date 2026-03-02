#ifndef REACTIONS_MERGING_BASE_H
#define REACTIONS_MERGING_BASE_H

#include "common_markers.hpp"
#include "particle_properties_map.hpp"
#include "reaction_kernel_pre_reqs.hpp"
#include "transformation_wrapper.hpp"
#include "utils.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

struct MergingKernelOnDeviceBase {

  MergingKernelOnDeviceBase() = default;

  void merge(const Access::SymVector::Write<INT> &req_int_props,
             const Access::SymVector::Write<REAL> &req_real_props,
             Access::CellDatConst::Read<REAL> &reduction,
             Access::CellDatConst::Read<REAL> &reduction_min,
             Access::CellDatConst::Read<REAL> &reduction_max,
             const size_t &reduction_idx, const size_t &merge_idx) const {
    return;
  }
};

struct ReductionKernelOnDeviceBase {

  ReductionKernelOnDeviceBase() = default;

  void
  reduce(const Access::SymVector::Read<INT> &req_int_props,
         const Access::SymVector::Read<REAL> &req_real_props,
         Access::CellDatConst::Reduction<REAL, Kernel::plus<REAL>> &reduction,
         Access::CellDatConst::Reduction<REAL, Kernel::minimum<REAL>>
             &reduction_min,
         Access::CellDatConst::Reduction<REAL, Kernel::maximum<REAL>>
             &reduction_max,
         const size_t &reduction_idx) const {
    return;
  }
};
template <size_t merge_dim, size_t reduction_plus_dim, size_t reduction_min_dim,
          size_t reduction_max_dim, typename REDUCTION_KERNEL_ON_DEVICE,
          typename MERGING_KERNEL_ON_DEVICE>
struct MergingKernelBase {

  static const size_t MERGE_DIM = merge_dim;
  static const size_t REDUCTION_PLUS_DIM = reduction_plus_dim;
  static const size_t REDUCTION_MIN_DIM = reduction_min_dim;
  static const size_t REDUCTION_MAX_DIM = reduction_max_dim;

  MergingKernelBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : required_int_props(ArgumentNameSet(required_int_props, properties_map)),
        required_real_props(
            ArgumentNameSet(required_real_props, properties_map)),
        properties_map(properties_map) {}

  /**
   * \overload
   * @brief Constructor for MergingKernelBase that sets only required real
   * properties.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  MergingKernelBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : MergingKernelBase(Properties<INT>(), required_real_props,
                          properties_map) {}

  /**
   * @brief Return all required integer properties as a vector of Syms
   *
   */
  std::vector<Sym<INT>> get_required_int_sym_vector() {
    return this->required_int_props.to_sym_vector();
  }

  /**
   * @brief Return all required real properties as a vector of Syms
   *
   */
  std::vector<Sym<REAL>> get_required_real_sym_vector() {
    return this->required_real_props.to_sym_vector();
  }

  static constexpr size_t get_merge_dim() { return merge_dim; }
  static constexpr size_t get_reduction_plus_dim() {
    return reduction_plus_dim;
  }
  static constexpr size_t get_reduction_min_dim() { return reduction_min_dim; }
  static constexpr size_t get_reduction_max_dim() { return reduction_max_dim; }

  MERGING_KERNEL_ON_DEVICE get_merging_kernel_on_device() {

    NESOASSERT(this->merging_on_device_obj.has_value(),
               "merging_on_device_obj in MergeKernelBase not initialised");
    return this->merging_on_device_obj.value();
  }

  REDUCTION_KERNEL_ON_DEVICE get_reduction_kernel_on_device() {

    NESOASSERT(this->reduction_on_device_obj.has_value(),
               "reduction_on_device_obj in MergeKernelBase not initialised");
    return this->reduction_on_device_obj.value();
  }

  virtual void
  pre_calculate_merging(CellDatConstSharedPtr<REAL> reductions,
                        CellDatConstSharedPtr<INT> pre_merge_num_parts) {
    return;
  };

protected:
  std::optional<REDUCTION_KERNEL_ON_DEVICE> reduction_on_device_obj;
  std::optional<MERGING_KERNEL_ON_DEVICE> merging_on_device_obj;
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;
  std::map<int, std::string> properties_map;
};

template <size_t num_merging_groups, typename MERGE_KERNEL>
struct MergeStrategy : TransformationStrategy {

  MergeStrategy(
      ParticleGroupSharedPtr template_group, MERGE_KERNEL merge_kernels,
      const std::map<int, std::string> &properties_map = get_default_map())
      : merge_kernels(merge_kernels) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    this->group_index_sym =
        Sym<INT>(properties_map.at(default_properties.grouping_index));

    this->linear_index_sym =
        Sym<INT>(properties_map.at(default_properties.linear_index));
    int cell_count = template_group->domain->mesh->get_cell_count();
    const size_t reduction_dim = this->merge_kernels.get_reduction_plus_dim();

    this->reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
        template_group->sycl_target, cell_count, reduction_dim,
        num_merging_groups);

    this->min_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
        template_group->sycl_target, cell_count, reduction_dim,
        num_merging_groups);

    this->max_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
        template_group->sycl_target, cell_count, reduction_dim,
        num_merging_groups);
    this->num_part_cell_dats = std::make_shared<CellDatConst<INT>>(
        template_group->sycl_target, cell_count, num_merging_groups, 1);
  }

  /**
   * @brief Perform merging on given subgroup
   *
   * @param target_subgroup
   */
  void transform_v(ParticleSubGroupSharedPtr target_subgroup) override {
    auto part_group = target_subgroup->get_particle_group();

    auto reduction_obj = this->merge_kernels.get_reduction_kernel_on_device();

    this->reduction_cell_dats->fill(0.0);
    this->num_part_cell_dats->fill(0);
    this->min_reduction_cell_dats->fill(std::numeric_limits<REAL>::max());
    this->max_reduction_cell_dats->fill(std::numeric_limits<REAL>::min());
    auto reduction_loop = particle_loop(
        "merge_reduction_loop", target_subgroup,
        [=](auto req_int_props, auto req_real_props, auto reduction_cell_dat,
            auto min_reduction_cell_dat, auto max_reduction_cell_dat,
            auto npart_merging_group, auto merging_group_index,
            auto linear_index) {
          reduction_obj.reduce(req_int_props, req_real_props,
                               reduction_cell_dat, min_reduction_cell_dat,
                               max_reduction_cell_dat, merging_group_index[0]);
          linear_index[0] =
              npart_merging_group.fetch_add(merging_group_index[0], 0, 1);
        },
        Access::read(
            sym_vector<INT>(target_subgroup,
                            this->merge_kernels.get_required_int_sym_vector())),
        Access::read(sym_vector<REAL>(
            target_subgroup,
            this->merge_kernels.get_required_real_sym_vector())),
        Access::reduce(this->reduction_cell_dats, Kernel::plus<REAL>()),
        Access::reduce(this->min_reduction_cell_dats, Kernel::minimum<REAL>()),
        Access::reduce(this->max_reduction_cell_dats, Kernel::maximum<REAL>()),
        Access::add(this->num_part_cell_dats),
        Access::read(this->group_index_sym),
        Access::write(this->linear_index_sym));

    reduction_loop->execute();

    this->merge_kernels.pre_calculate_merging(this->reduction_cell_dats,
                                              this->num_part_cell_dats);

    auto merge_obj = this->merge_kernels.get_merging_kernel_on_device();

    const size_t num_merged_particles = this->merge_kernels.get_merge_dim();

    auto sub_group_to_merge = static_particle_sub_group(
        target_subgroup,
        [=](auto linear_index) {
          return linear_index[0] < num_merged_particles;
        },
        Access::read(this->linear_index_sym));

    auto sub_group_to_remove_particles = static_particle_sub_group(
        target_subgroup,
        [=](auto linear_index) {
          return linear_index[0] >= num_merged_particles;
        },
        Access::read(this->linear_index_sym));

    particle_loop(
        "MergeTransform::merge_loop", sub_group_to_merge,
        [=](auto req_int_props, auto req_real_props, auto reduction_cell_dat,
            auto min_reduction_cell_dat, auto max_reduction_cell_dat,
            auto n_part_merging_group, auto merging_group_index,
            auto linear_index) {
          if (n_part_merging_group.at(merging_group_index[0], 0) >
              num_merged_particles) {
            merge_obj.merge(req_int_props, req_real_props, reduction_cell_dat,
                            min_reduction_cell_dat, max_reduction_cell_dat,
                            merging_group_index[0], linear_index[0]);
          }
        },
        Access::write(
            sym_vector<INT>(target_subgroup,
                            this->merge_kernels.get_required_int_sym_vector())),
        Access::write(sym_vector<REAL>(
            target_subgroup,
            this->merge_kernels.get_required_real_sym_vector())),
        Access::read(this->reduction_cell_dats),
        Access::read(this->min_reduction_cell_dats),
        Access::read(this->max_reduction_cell_dats),
        Access::read(this->num_part_cell_dats),
        Access::read(this->group_index_sym),
        Access::read(this->linear_index_sym))
        ->execute();

    part_group->remove_particles(sub_group_to_remove_particles);
  }

private:
  Sym<INT> group_index_sym;
  Sym<INT> linear_index_sym;
  MERGE_KERNEL merge_kernels;
  CellDatConstSharedPtr<REAL> reduction_cell_dats;
  CellDatConstSharedPtr<INT> num_part_cell_dats;
  CellDatConstSharedPtr<REAL> min_reduction_cell_dats;
  CellDatConstSharedPtr<REAL> max_reduction_cell_dats;
};
} // namespace VANTAGE::Reactions
#endif
