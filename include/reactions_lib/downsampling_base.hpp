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

/**
 *
 * The general idea of downsampling is that we might want to reduce the number
 * of particles while maintaining some properties of the ensemble, such as
 * conserving various moments either on average or deterministically.
 *
 * In general, these algorithms have the following structure:
 *
 * 1. Reduction - (Optional) Reduce some number of quantities across the
 * particle ensemble, such as weight, momentum, energy, etc.
 * 2. Downsampling - Change the properties of the particles such that some
 * particles are effectively marked for removal while others have their
 * properties set based on the downsampling algorthm (such as merging or
 * thinning)
 * 3. Remove particles with 0 weight or otherwise marked for removal
 *
 * The above algorithm is assumed to be applied for each downsampling group
 * separately, and the downsampling algorithms assume that particles are grouped
 * before the application of the downsampling. An example of downsampling is
 * velocity/phase space binning.
 *
 *
 * The number of downsampling groups determines the size of the CellDatConsts
 * used to store the group-wise particle property reductions, so all mentions
 * below of reduction dimensionalities or indices refer to these individual
 * downsampling groups.
 *
 */

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * Downsampling modes:
 *
 * 1. merging - always requires reduction strategies, with the post-merge
 * particles being the first downsampling_dim particles, and all of the rest
 * are discarded
 *
 * 2. thinning - does not require reduction strategies, but can use them, and
 * performs the thinning transformation on all particles, removing those whose
 * weight is set to 0 during the process
 *
 */
enum class DownsamplingMode { merging, thinning };

/**
 * @brief Base on-device downsampling kernel, meant to apply the downsampling
 * transformation on each particle on-device
 *
 * @tparam downsampling_dim The downsampling dimensionality, e.g.
 * post-downsampling number of particles or other measure
 * @tparam RNG_TYPE Type of rng kernel, if needed
 */
template <size_t downsampling_dim, typename RNG_TYPE = DEFAULT_RNG_KERNEL>
struct DownsamplingKernelOnDeviceBase {

  static inline constexpr size_t DOWNSAMPLING_DIM = downsampling_dim;
  using RNG_KERNEL_TYPE = RNG_TYPE;
  DownsamplingKernelOnDeviceBase() = default;

  /**
   * @brief Apply the downsampling algorithm, assuming reduction has happened
   * prior to the application
   *
   * @param index LoopIndex accessor used for linear indexing
   * @param req_int_props SymVector Write access to required integer properties
   * @param req_real_props SymVector Write access to required real properties
   * @param reduction Read access to additive cellwise reduction data
   * @param reduction_min Read access to cellwise min reduction data
   * @param reduction_max Read access to cellwise max reduction data
   * @param reduction_idx Index determining which downsampling/reduction group
   * the particle belongs to, in principle used to access the corresponding
   * column of the reduction data
   * @param linear_idx Linear index determining which of the post-downsampling
   * particles the current particle is
   * @param rng_kernel RNG kernel access, if required
   */
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

  /**
   * @brief Apply the downsampling algorithm, assuming no reductions are needed
   *
   * @param index LoopIndex accessor used for linear indexing
   * @param req_int_props SymVector Write access to required integer properties
   * @param req_real_props SymVector Write access to required real properties
   * @param rng_kernel RNG kernel access, if required
   */
  void apply_no_red(const Access::LoopIndex::Read &index,
                    const Access::SymVector::Write<INT> &req_int_props,
                    const Access::SymVector::Write<REAL> &req_real_props,
                    typename RNG_TYPE::KernelType &rng_kernel) const {
    return;
  }
};

/**
 * @brief Base on-device reduction kernel, responsible for performing reduction
 * operations on particle data needed in downsampling algorithms
 *
 * @tparam reduction_plus_dim Number of additive reduction quantities
 * @tparam reduction_min_dim Number of max reduction quantities
 * @tparam reduction_max_dim Number of min reduction quantities
 */
template <size_t reduction_plus_dim, size_t reduction_min_dim,
          size_t reduction_max_dim>
struct DownsamplingReductionKernelOnDeviceBase {

  static inline constexpr size_t REDUCTION_PLUS_DIM = reduction_plus_dim;
  static inline constexpr size_t REDUCTION_MIN_DIM = reduction_min_dim;
  static inline constexpr size_t REDUCTION_MAX_DIM = reduction_max_dim;
  static inline constexpr size_t TOTAL_REDUCTION_DIM =
      reduction_min_dim + reduction_max_dim + reduction_plus_dim;
  DownsamplingReductionKernelOnDeviceBase() = default;

  /**
   * @brief Calculate the contributions to the various reduction quantities
   * needed for downsampling
   *
   * @param req_int_props SymVector Read access to required integer properties
   * @param req_real_props SymVector Read access to required real properties
   * @param reduction Add access to additive cellwise reduction data
   * @param reduction_min Min access to cellwise min reduction data
   * @param reduction_max Max access to cellwise max reduction data
   * @param reduction_idx Index determining which downsampling/reduction group
   * the particle belongs to, in principle used to access the corresponding
   * column of the reduction data
   */
  void reduce(const Access::SymVector::Read<INT> &req_int_props,
              const Access::SymVector::Read<REAL> &req_real_props,
              Access::CellDatConst::Add<REAL> &reduction,
              Access::CellDatConst::Min<REAL> &reduction_min,
              Access::CellDatConst::Max<REAL> &reduction_max,
              const size_t &reduction_idx) const {
    return;
  }
};

/**
 * @brief Base host type for downsampling kernels, containing the on-device
 * reduction and sampling kernels
 *
 * @tparam mode The downsampling mode of the kernel, determining downstream
 * behaviour (e.g. merging vs thinning etc.)
 * @tparam REDUCTION_KERNEL_ON_DEVICE On device object type responsible for the
 * calculation of the various reduction quantities needed for downsampling
 * algorithms
 * @tparam DOWNSAMPLING_KERNEL_ON_DEVICE On device object type responsible for
 * the application of the downsampling algorithm in conjunction with any
 * calculated reduced quantities
 */
template <DownsamplingMode mode, typename REDUCTION_KERNEL_ON_DEVICE,
          typename DOWNSAMPLING_KERNEL_ON_DEVICE>
struct DownsamplingKernelBase {
  using RNG_TYPE = typename DOWNSAMPLING_KERNEL_ON_DEVICE::RNG_KERNEL_TYPE;
  static const DownsamplingMode DOWNSAMPLING_MODE = mode;

  static inline constexpr size_t DOWNSAMPLING_DIM =
      DOWNSAMPLING_KERNEL_ON_DEVICE::DOWNSAMPLING_DIM;
  static inline constexpr size_t REDUCTION_PLUS_DIM =
      REDUCTION_KERNEL_ON_DEVICE::REDUCTION_PLUS_DIM;
  static inline constexpr size_t REDUCTION_MIN_DIM =
      REDUCTION_KERNEL_ON_DEVICE::REDUCTION_MIN_DIM;
  static inline constexpr size_t REDUCTION_MAX_DIM =
      REDUCTION_KERNEL_ON_DEVICE::REDUCTION_MAX_DIM;
  static inline constexpr size_t TOTAL_REDUCTION_DIM =
      REDUCTION_KERNEL_ON_DEVICE::TOTAL_REDUCTION_DIM;

  /**
   * @brief Base host-side downsampling kernel type constructor
   *
   * @param required_int_props Properties<INT> object containing information
   * regarding the required INT-based properties
   * @param required_real_props Properties<REAL> object containing information
   * regarding the required REAL-based properties
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names
   */
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
   * regarding the required REAL-based properties
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names
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

  /**
   * @brief Perform any operations required before the application of the
   * downsampling
   *
   * @param reductions Additive reduction values
   * @param pre_num_parts The number of particles per cell
   */
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

/**
 * @brief Transformation strategy performing downsampling based on the contained
 * downsampling kernels
 *
 * @tparam DOWNSAMPLING_KERNEL Host-side downsampling kernel type, determining
 * the reduction and downsampling application algorithms
 */
template <typename DOWNSAMPLING_KERNEL>
struct DownsamplingStrategy : TransformationStrategy {

  /**
   * @brief DownsamplingStrategy constructor
   *
   * @param template_group Particle group with the same domain and sycl_target
   * as the group this strategy is to be applied to
   * @param downsampling_kernels The kernels containing the reduction and
   * downsampling strategy algorithms
   * @param num_downsampling_groups The number of distinct downsampling groups
   * (such as velocity/phase space bins) - determines the dimensionality of the
   * CellDatConst objects storing cell-wise and downsampling group-wise
   * reductions of the properties needed for the downsampling algorithm
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names, in particular the grouping index,
   * linear index, and particle weights
   */
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

    // We only need to allocate reduction quantities if there are any reductions
    // needed
    if constexpr (DOWNSAMPLING_KERNEL::TOTAL_REDUCTION_DIM > 0) {
      this->reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count,
          DOWNSAMPLING_KERNEL::REDUCTION_PLUS_DIM, num_downsampling_groups);

      this->min_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count,
          DOWNSAMPLING_KERNEL::REDUCTION_MIN_DIM, num_downsampling_groups);

      this->max_reduction_cell_dats = std::make_shared<CellDatConst<REAL>>(
          template_group->sycl_target, cell_count,
          DOWNSAMPLING_KERNEL::REDUCTION_MAX_DIM, num_downsampling_groups);
    }

    // We only need to track the number of particles in the merging mode
    // because it selects the first N particles to set to the post-merging
    // properties
    if constexpr (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE ==
                  DownsamplingMode::merging) {
      this->num_part_cell_dats = std::make_shared<CellDatConst<INT>>(
          template_group->sycl_target, cell_count, num_downsampling_groups, 1);
    }
  }

  /**
   * @brief Perform downsampling on given subgroup
   *
   * @param target_subgroup
   */
  void transform_v(ParticleSubGroupSharedPtr target_subgroup) override {
    auto part_group = target_subgroup->get_particle_group();

    auto reduction_obj =
        this->downsampling_kernels.get_reduction_kernel_on_device();

    if constexpr (DOWNSAMPLING_KERNEL::DOWNSAMPLING_MODE ==
                  DownsamplingMode::merging) {

      // When merging we assume we will always need to perform some reductions,
      // since all merging algorithms attempt to conserve at least some moments
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
                  DOWNSAMPLING_KERNEL::TOTAL_REDUCTION_DIM > 0) {
      // When not merging, we assume we only need to perform reduction if
      // tot_reduction_dim > 0
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

      // When merging we first select the first DOWNSAMPLING_DIM, i.e.
      // the post-merging number of particles, for each of the downsampling
      // groups
      auto sub_group_to_merge = static_particle_sub_group(
          target_subgroup,
          [=](auto linear_index) {
            return linear_index[0] < DOWNSAMPLING_KERNEL::DOWNSAMPLING_DIM;
          },
          Access::read(this->linear_index_sym));

      // Then we apply the merging loop by going through the first
      // DOWNSAMPLING_DIM in each downsampling group, and if there are
      // enough particles in the corresponding group we set the properties of
      // the particles to the post-merge values
      particle_loop(
          "DownsamplingTransform::merge_loop", sub_group_to_merge,
          [=](auto loop_index, auto req_int_props, auto req_real_props,
              auto reduction_cell_dat, auto min_reduction_cell_dat,
              auto max_reduction_cell_dat, auto n_part_group,
              auto downsampling_group_index, auto linear_index,
              auto rng_kernel) {
            if (n_part_group.at(downsampling_group_index[0], 0) >
                DOWNSAMPLING_KERNEL::DOWNSAMPLING_DIM) {
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

      // Finally, all of the remaining particles are removed
      auto sub_group_to_remove_particles = static_particle_sub_group(
          target_subgroup,
          [=](auto linear_index) {
            return linear_index[0] >= DOWNSAMPLING_KERNEL::DOWNSAMPLING_DIM;
          },
          Access::read(this->linear_index_sym));

      part_group->remove_particles(sub_group_to_remove_particles);
      break;
    }
    case DownsamplingMode::thinning: {

      // With thinning we might have algorithms that do not have any reduction
      // requirements so we dispatch two different kinds of loops
      //
      // Note that unlike the merging loop, the thinning loop is applied on the
      // whole subgroup, with the assumption that any thinned particles will
      // have their weights set to 0
      if constexpr (DOWNSAMPLING_KERNEL::TOTAL_REDUCTION_DIM > 0) {
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
              // The no-reduction version of the thinning application
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
