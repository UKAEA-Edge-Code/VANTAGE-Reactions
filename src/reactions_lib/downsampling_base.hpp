#ifndef REACTIONS_DOWNSAMPLING_BASE_H
#define REACTIONS_DOWNSAMPLING_BASE_H

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

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

enum class DownsamplingMode { merging, thinning };

template <size_t downsampling_dim, typename RNG_TYPE = DEFAULT_RNG_KERNEL>
struct DownsamplingKernelOnDeviceBase {

  static const size_t DOWNSAMPLING_DIM = downsampling_dim;
  using RNG_KERNEL_TYPE = RNG_TYPE;
  DownsamplingKernelOnDeviceBase() = default;

  void apply(const Access::LoopIndex::Read &index,
             const Access::SymVector::Write<INT> &req_int_props,
             const Access::SymVector::Write<REAL> &req_real_props,
             Access::CellDatConst::Read<REAL> &reduction,
             Access::CellDatConst::Read<REAL> &reduction_min,
             Access::CellDatConst::Read<REAL> &reduction_max,
             const size_t &reduction_idx, const size_t &linear_idx,
             typename RNG_TYPE::KernelType &rng_kernel) const {
    return;
  }

  void apply_no_red(const Access::LoopIndex::Read &index,
                    const Access::SymVector::Write<INT> &req_int_props,
                    const Access::SymVector::Write<REAL> &req_real_props,
                    typename RNG_TYPE::KernelType &rng_kernel) const {
    return;
  }
};

template <size_t reduction_plus_dim, size_t reduction_min_dim,
          size_t reduction_max_dim>
struct ReductionKernelOnDeviceBase {

  static const size_t REDUCTION_PLUS_DIM = reduction_plus_dim;
  static const size_t REDUCTION_MIN_DIM = reduction_min_dim;
  static const size_t REDUCTION_MAX_DIM = reduction_max_dim;
  static const size_t TOTAL_REDUCTION_DIM =
      reduction_min_dim + reduction_max_dim + reduction_plus_dim;
  ReductionKernelOnDeviceBase() = default;

  void reduce(const Access::SymVector::Read<INT> &req_int_props,
              const Access::SymVector::Read<REAL> &req_real_props,
              Access::CellDatConst::Add<REAL> &reduction,
              Access::CellDatConst::Min<REAL> &reduction_min,
              Access::CellDatConst::Max<REAL> &reduction_max,
              const size_t &reduction_idx) const {
    return;
  }
};
template <DownsamplingMode mode, typename REDUCTION_KERNEL_ON_DEVICE,
          typename DOWNSAMPLING_KERNEL_ON_DEVICE>
struct DownsamplingKernelBase {
  using RNG_TYPE = typename DOWNSAMPLING_KERNEL_ON_DEVICE::RNG_KERNEL_TYPE;
  const static DownsamplingMode DOWNSAMPLING_MODE = mode;

  DownsamplingKernelBase(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : required_int_props(ArgumentNameSet(required_int_props, properties_map)),
        required_real_props(
            ArgumentNameSet(required_real_props, properties_map)),
        properties_map(properties_map) {

    static_assert(
        DOWNSAMPLING_MODE != DownsamplingMode::merging ||
            REDUCTION_KERNEL_ON_DEVICE::TOTAL_REDUCTION_DIM > 0,
        "In merging mode downsampling requires at least one reduced quantity");

    this->rng_kernel = std::make_shared<RNG_TYPE>();
  }

  /**
   * \overload
   * @brief Constructor for DownsamplingKernelBase that sets only required real
   * properties.
   *
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties for the reaction data.
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names (in get_required_real_props(...) and
   * get_required_int_props(...)).
   */
  DownsamplingKernelBase(
      Properties<REAL> required_real_props,
      std::map<int, std::string> properties_map = get_default_map())
      : DownsamplingKernelBase(Properties<INT>(), required_real_props,
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

  static constexpr size_t get_downsampling_dim() {
    return DOWNSAMPLING_KERNEL_ON_DEVICE::DOWNSAMPLING_DIM;
  }
  static constexpr size_t get_reduction_plus_dim() {
    return REDUCTION_KERNEL_ON_DEVICE::REDUCTION_PLUS_DIM;
  }
  static constexpr size_t get_reduction_min_dim() {
    return REDUCTION_KERNEL_ON_DEVICE::REDUCTION_MIN_DIM;
  }
  static constexpr size_t get_reduction_max_dim() {
    return REDUCTION_KERNEL_ON_DEVICE::REDUCTION_MAX_DIM;
  }
  static constexpr size_t get_total_reduction_dim() {
    return REDUCTION_KERNEL_ON_DEVICE::TOTAL_REDUCTION_DIM;
  }

  DOWNSAMPLING_KERNEL_ON_DEVICE get_downsampling_kernel_on_device() {

    NESOASSERT(
        this->downsampling_on_device_obj.has_value(),
        "downsampling_on_device_obj in DownsamplingKernelBase not initialised");
    return this->downsampling_on_device_obj.value();
  }

  REDUCTION_KERNEL_ON_DEVICE get_reduction_kernel_on_device() {

    NESOASSERT(
        this->reduction_on_device_obj.has_value(),
        "reduction_on_device_obj in DownsamplingKernelBase not initialised");
    return this->reduction_on_device_obj.value();
  }

  virtual void pre_calculate(CellDatConstSharedPtr<REAL> reductions,
                             CellDatConstSharedPtr<INT> pre_num_parts) {
    return;
  };
  std::shared_ptr<RNG_TYPE> get_rng_kernel() { return this->rng_kernel; }

  void set_rng_kernel(std::shared_ptr<RNG_TYPE> rng_kernel) {
    this->rng_kernel = rng_kernel;
  }

protected:
  std::optional<REDUCTION_KERNEL_ON_DEVICE> reduction_on_device_obj;
  std::optional<DOWNSAMPLING_KERNEL_ON_DEVICE> downsampling_on_device_obj;
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;
  std::map<int, std::string> properties_map;
  std::shared_ptr<RNG_TYPE> rng_kernel;
};

template <typename DOWNSAMPLING_KERNEL>
struct DownsamplingStrategy : TransformationStrategy {

  DownsamplingStrategy(
      ParticleGroupSharedPtr template_group,
      DOWNSAMPLING_KERNEL downsampling_kernels, size_t num_downsampling_groups,
      const std::map<int, std::string> &properties_map = get_default_map())
      : downsampling_kernels(downsampling_kernels) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    this->group_index_sym =
        Sym<INT>(properties_map.at(default_properties.grouping_index));

    this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));

    this->linear_index_sym =
        Sym<INT>(properties_map.at(default_properties.linear_index));
    int cell_count = template_group->domain->mesh->get_cell_count();
    const size_t reduction_dim =
        this->downsampling_kernels.get_reduction_plus_dim();
    constexpr size_t tot_reduction_dim =
        this->downsampling_kernels.get_total_reduction_dim();

    if constexpr (tot_reduction_dim > 0) {
      this->reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count, reduction_dim,
          num_downsampling_groups);

      this->min_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count,
          this->downsampling_kernels.get_reduction_min_dim(),
          num_downsampling_groups);

      this->max_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count,
          this->downsampling_kernels.get_reduction_max_dim(),
          num_downsampling_groups);
    }

    if constexpr (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE ==
                  DownsamplingMode::merging) {
      this->num_part_cell_dats = std::make_shared<CellDatConst<INT>>(
          template_group->sycl_target, cell_count, num_downsampling_groups, 1);
    }
  }

  /**
   * @brief Perform merging on given subgroup
   *
   * @param target_subgroup
   */
  void transform_v(ParticleSubGroupSharedPtr target_subgroup) override {
    auto part_group = target_subgroup->get_particle_group();

    auto reduction_obj =
        this->downsampling_kernels.get_reduction_kernel_on_device();

    const size_t num_downsampled_particles =
        this->downsampling_kernels.get_downsampling_dim();
    constexpr size_t tot_reduction_dim =
        this->downsampling_kernels.get_total_reduction_dim();

    if constexpr (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE ==
                  DownsamplingMode::merging) {
      this->reduction_cell_dats->fill(0.0);
      this->num_part_cell_dats->fill(0);
      this->min_reduction_cell_dats->fill(std::numeric_limits<REAL>::max());
      this->max_reduction_cell_dats->fill(-std::numeric_limits<REAL>::max());
      particle_loop(
          "merge_reduction_loop", target_subgroup,
          [=](auto req_int_props, auto req_real_props, auto reduction_cell_dat,
              auto min_reduction_cell_dat, auto max_reduction_cell_dat,
              auto downsampling_group_index, auto npart_group,
              auto linear_index) {
            reduction_obj.reduce(req_int_props, req_real_props,
                                 reduction_cell_dat, min_reduction_cell_dat,
                                 max_reduction_cell_dat,
                                 downsampling_group_index[0]);
            linear_index[0] =
                npart_group.fetch_add(downsampling_group_index[0], 0, 1);
          },
          Access::read(sym_vector<INT>(
              target_subgroup,
              this->downsampling_kernels.get_required_int_sym_vector())),
          Access::read(sym_vector<REAL>(
              target_subgroup,
              this->downsampling_kernels.get_required_real_sym_vector())),
          Access::add(this->reduction_cell_dats),
          Access::min(this->min_reduction_cell_dats),
          Access::max(this->max_reduction_cell_dats),
          Access::read(this->group_index_sym),
          Access::add(this->num_part_cell_dats),
          Access::write(this->linear_index_sym))
          ->execute();
      this->downsampling_kernels.pre_calculate(this->reduction_cell_dats,
                                               this->num_part_cell_dats);
    }

    if constexpr (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE !=
                      DownsamplingMode::merging &&
                  tot_reduction_dim > 0) {
      particle_loop(
          "thinning_reduction_loop", target_subgroup,
          [=](auto req_int_props, auto req_real_props, auto reduction_cell_dat,
              auto min_reduction_cell_dat, auto max_reduction_cell_dat,
              auto downsampling_group_index) {
            reduction_obj.reduce(req_int_props, req_real_props,
                                 reduction_cell_dat, min_reduction_cell_dat,
                                 max_reduction_cell_dat,
                                 downsampling_group_index[0]);
          },
          Access::read(sym_vector<INT>(
              target_subgroup,
              this->downsampling_kernels.get_required_int_sym_vector())),
          Access::read(sym_vector<REAL>(
              target_subgroup,
              this->downsampling_kernels.get_required_real_sym_vector())),
          Access::add(this->reduction_cell_dats),
          Access::min(this->min_reduction_cell_dats),
          Access::max(this->max_reduction_cell_dats),
          Access::read(this->group_index_sym))
          ->execute();
    }

    auto downsampling_obj =
        this->downsampling_kernels.get_downsampling_kernel_on_device();

    switch (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE) {

    case DownsamplingMode::merging: {
      auto sub_group_to_merge = static_particle_sub_group(
          target_subgroup,
          [=](auto linear_index) {
            return linear_index[0] < num_downsampled_particles;
          },
          Access::read(this->linear_index_sym));

      particle_loop(
          "DownsamplingTransform::merge_loop", sub_group_to_merge,
          [=](auto loop_index, auto req_int_props, auto req_real_props,
              auto reduction_cell_dat, auto min_reduction_cell_dat,
              auto max_reduction_cell_dat, auto n_part_group,
              auto downsampling_group_index, auto linear_index,
              auto rng_kernel) {
            if (n_part_group.at(downsampling_group_index[0], 0) >
                num_downsampled_particles) {
              downsampling_obj.apply(
                  loop_index, req_int_props, req_real_props, reduction_cell_dat,
                  min_reduction_cell_dat, max_reduction_cell_dat,
                  downsampling_group_index[0], linear_index[0], rng_kernel);
            }
          },
          Access::read(ParticleLoopIndex{}),
          Access::write(sym_vector<INT>(
              target_subgroup,
              this->downsampling_kernels.get_required_int_sym_vector())),
          Access::write(sym_vector<REAL>(
              target_subgroup,
              this->downsampling_kernels.get_required_real_sym_vector())),
          Access::read(this->reduction_cell_dats),
          Access::read(this->min_reduction_cell_dats),
          Access::read(this->max_reduction_cell_dats),
          Access::read(this->num_part_cell_dats),
          Access::read(this->group_index_sym),
          Access::read(this->linear_index_sym),
          Access::read(this->downsampling_kernels.get_rng_kernel()))
          ->execute();

      auto sub_group_to_remove_particles = static_particle_sub_group(
          target_subgroup,
          [=](auto linear_index) {
            return linear_index[0] >= num_downsampled_particles;
          },
          Access::read(this->linear_index_sym));

      part_group->remove_particles(sub_group_to_remove_particles);
      break;
    }
    case DownsamplingMode::thinning: {

      if constexpr (tot_reduction_dim > 0) {
        particle_loop(
            "DownsamplingTransform::thinning_loop", target_subgroup,
            [=](auto loop_index, auto req_int_props, auto req_real_props,
                auto reduction_cell_dat, auto min_reduction_cell_dat,
                auto max_reduction_cell_dat, auto downsampling_group_index,
                auto rng_kernel) {
              downsampling_obj.apply(
                  loop_index, req_int_props, req_real_props, reduction_cell_dat,
                  min_reduction_cell_dat, max_reduction_cell_dat,
                  downsampling_group_index[0], 0, rng_kernel);
            },
            Access::read(ParticleLoopIndex{}),
            Access::write(sym_vector<INT>(
                target_subgroup,
                this->downsampling_kernels.get_required_int_sym_vector())),
            Access::write(sym_vector<REAL>(
                target_subgroup,
                this->downsampling_kernels.get_required_real_sym_vector())),
            Access::read(this->reduction_cell_dats),
            Access::read(this->min_reduction_cell_dats),
            Access::read(this->max_reduction_cell_dats),
            Access::read(this->group_index_sym),
            Access::read(this->downsampling_kernels.get_rng_kernel()))
            ->execute();
      } else {

        particle_loop(
            "DownsamplingTransform::thinning_loop_no_reduction",
            target_subgroup,
            [=](auto loop_index, auto req_int_props, auto req_real_props,
                auto rng_kernel) {
              downsampling_obj.apply_no_red(loop_index, req_int_props,
                                            req_real_props, rng_kernel);
            },
            Access::read(ParticleLoopIndex{}),
            Access::write(sym_vector<INT>(
                target_subgroup,
                this->downsampling_kernels.get_required_int_sym_vector())),
            Access::write(sym_vector<REAL>(
                target_subgroup,
                this->downsampling_kernels.get_required_real_sym_vector())),
            Access::read(this->downsampling_kernels.get_rng_kernel()))
            ->execute();
      }

      // assuming that some particles had their weights set to 0
      auto sub_group_to_remove_particles = particle_sub_group(
          target_subgroup,
          [=](auto weight) {
            return weight[0] < 1e-16; // HARDCODED COMPARISON
          },
          Access::read(this->weight_sym));
      part_group->remove_particles(sub_group_to_remove_particles);
      break;
    }
    }
  }

private:
  Sym<INT> group_index_sym;
  Sym<INT> linear_index_sym;
  Sym<REAL> weight_sym;
  DOWNSAMPLING_KERNEL downsampling_kernels;
  CellDatConstSharedPtr<REAL> reduction_cell_dats;
  CellDatConstSharedPtr<INT> num_part_cell_dats;
  CellDatConstSharedPtr<REAL> min_reduction_cell_dats;
  CellDatConstSharedPtr<REAL> max_reduction_cell_dats;
};
} // namespace VANTAGE::Reactions
#endif
