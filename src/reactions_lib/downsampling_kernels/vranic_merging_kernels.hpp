#ifndef REACTIONS_VRANIC_MERGING_H
#define REACTIONS_VRANIC_MERGING_H

#include "reactions_lib/downsampling_base.hpp"
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

/**
 * Implementation of simplified merging algorithm from M. Vranic et
 * al. Computer Physics Communications 191 2015.
 *
 * The assumption is that all particles being merged are of the same species,
 * i.e. have the same mass and that they are non-relativistic.
 *
 * Particles are merged group-wise and cell-wise into 2 particles. The
 * properties modified are the positions, weights, and momenta/velocities. Other
 * properties are sampled from 2 other particles in the passed subgroup, i.e.
 * things like cell or particle ids will be copied consistently, but there is no
 * reduction of other real quantities. This means that those values will be
 * lost, so this algorithm should be called only AFTER they are no longer
 * needed.
 *
 */

/**
 * @brief The on-device merging object, setting the values of post-merge
 * properties on each of the remaining 2 particles
 *
 * @tparam ndim The dimensionality of the velocity space
 */
template <size_t ndim>
struct VranicMergingOnDevice : DownsamplingKernelOnDeviceBase<2> {

  VranicMergingOnDevice() = default;

  /**
   * @brief Apply the merging algorithm, assuming reduction has happened
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
  void
  apply(const Access::LoopIndex::Read &index,
        const Access::SymVector::Write<INT> &req_int_props,
        const Access::SymVector::Write<REAL> &req_real_props,
        Access::CellDatConst::Read<REAL> &reduction,
        Access::CellDatConst::Read<REAL> &reduction_min,
        Access::CellDatConst::Read<REAL> &reduction_max,
        const size_t &reduction_idx, const size_t &linear_idx,
        typename DownsamplingKernelOnDeviceBase<2>::RNG_KERNEL_TYPE::KernelType
            &rng_kernel) const {

    REAL mom_tot[ndim];
    REAL mom_a[ndim];
    REAL mom_b[ndim];
    const REAL wt = reduction.at(reduction_idx, 0);
    const REAL one_over_wt = 1.0 / wt;
    const REAL et = reduction.at(reduction_idx, 1);
    for (int dimx = 0; dimx < ndim; dimx++) {
      mom_tot[dimx] = reduction.at(reduction_idx, 2 + dimx);
      mom_a[dimx] = mom_tot[dimx] * one_over_wt;
      mom_b[dimx] = mom_tot[dimx] * one_over_wt;
    }

    if constexpr (ndim == 2) {

      const REAL pt = Kernel::sqrt(Kernel::dot_product_2d(mom_tot, mom_tot));

      // et/wt is the momentum**2 for either of the result particles,
      // and pt/wt is the momentum in the direction of the total
      // momentum vector so the below is the perpendicular momentum of
      // the resulting particles
      const REAL p_perp2 =
          Kernel::max((et / wt) - ((pt * pt) / (wt * wt)), 0.0);
      const REAL p_perp = Kernel::sqrt(p_perp2);
      // applying the the 2D 90deg rotation matrix [[0 -1][1 0]] to the
      // total momentum direction and scaling with the perpendicular
      // momentum
      const REAL p_perp_over_pt = pt != 0.0 ? p_perp / pt : 0.0;
      mom_a[0] -= mom_tot[1] * p_perp_over_pt;
      mom_a[1] += mom_tot[0] * p_perp_over_pt;
      mom_b[0] += mom_tot[1] * p_perp_over_pt;
      mom_b[1] -= mom_tot[0] * p_perp_over_pt;

    } else if constexpr (ndim == 3) {

      const REAL pt = Kernel::sqrt(Kernel::dot_product_3d(mom_tot, mom_tot));

      // et/wt is the momentum**2 for either of the result particles,
      // and pt/wt is the momentum in the direction of the total
      // momentum vector so the below is the perpendicular momentum of
      // the resulting particles
      const REAL p_perp2 =
          Kernel::max((et / wt) - ((pt * pt) / (wt * wt)), 0.0);
      const REAL p_perp = Kernel::sqrt(p_perp2);

      REAL mom_cell_diag[3] = {reduction_max.at(reduction_idx, 0) -
                                   reduction_min.at(reduction_idx, 0),
                               reduction_max.at(reduction_idx, 1) -
                                   reduction_min.at(reduction_idx, 1),
                               reduction_max.at(reduction_idx, 2) -
                                   reduction_min.at(reduction_idx, 2)};

      REAL rotation_axis[3] = {0, 0, 0};
      Kernel::cross_product(mom_tot[0], mom_tot[1], mom_tot[2],
                            mom_cell_diag[0], mom_cell_diag[1],
                            mom_cell_diag[2], rotation_axis, rotation_axis + 1,
                            rotation_axis + 2);

      // the cross product of the total momentum and the momentum space
      // bounding box diagonal of the downsampling group
      REAL rotation_axis_norm =
          Kernel::sqrt(Kernel::dot_product_3d(rotation_axis, rotation_axis));

      const REAL mom_cell_diag_norm =
          Kernel::sqrt(Kernel::dot_product_3d(mom_cell_diag, mom_cell_diag));

      // Use short circuit evaluation to mask off the 0/0 that happens if
      // the momentum is zero.
      if ((mom_cell_diag_norm != 0.0) &&
          (rotation_axis_norm / (pt * mom_cell_diag_norm) < 1e-10)) {
        mom_cell_diag[0] = -mom_cell_diag[0];
        Kernel::cross_product(mom_tot[0], mom_tot[1], mom_tot[2],
                              mom_cell_diag[0], mom_cell_diag[1],
                              mom_cell_diag[2], rotation_axis,
                              rotation_axis + 1, rotation_axis + 2);
        rotation_axis_norm =
            Kernel::sqrt(Kernel::dot_product_3d(rotation_axis, rotation_axis));
      }

      // the 3D 90deg rotation matrix used here is
      // [[0 -u_3 u_2][u_3 0 -u_1][-u_2 u_1 0]] where u is the rotation
      // axis this is the cross product matrix of the rotation axis -
      // hence
      REAL mom_perp[3] = {0, 0, 0};
      Kernel::cross_product(rotation_axis[0], rotation_axis[1],
                            rotation_axis[2], mom_tot[0], mom_tot[1],
                            mom_tot[2], mom_perp, mom_perp + 1, mom_perp + 2);

      const REAL scaling_factor =
          rotation_axis_norm != 0.0 ? p_perp / (pt * rotation_axis_norm) : 0.0;
      for (int i = 0; i < 3; i++) {
        mom_perp[i] *= scaling_factor;
      }

      for (int i = 0; i < 3; i++) {
        mom_a[i] += mom_perp[i];
        mom_b[i] -= mom_perp[i];
      }
    }

    req_real_props.at(this->weight_ind, 0) = wt * 0.5;
    for (int dimx = 0; dimx < ndim; dimx++) {

      req_real_props.at(velocity_ind, dimx) =
          (linear_idx == 0) ? mom_a[dimx] : mom_b[dimx];
    }
    return;
  }

public:
  int weight_ind, velocity_ind;
};

/**
 * @brief The reduction kernels for the Vranic merging algorithm. They will
 * calculate the total momentum and energy of the particles in each downsampling
 * cell, as well as the minimum and maximum values of the particle velocities in
 * each direction - to be used to generate bounding boxes in the above merging
 * algorithm
 *
 * @tparam ndim The dimensionality of the velocity space
 */
template <size_t ndim>
struct VranicReductionOnDevice
    : DownsamplingReductionKernelOnDeviceBase<ndim + 2, ndim, ndim> {

  VranicReductionOnDevice() = default;

  /**
   * @brief Reduce the weight, momentum, and energy of the particles
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

    auto weight = req_real_props.at(this->weight_ind, 0);
    REAL vel;
    reduction.fetch_add(reduction_idx, 0, weight);
    for (int i = 0; i < ndim; i++) {
      vel = req_real_props.at(this->velocity_ind, i);
      reduction.fetch_add(reduction_idx, 1, weight * vel * vel);
      reduction.fetch_add(reduction_idx, 2 + i, weight * vel);

      if constexpr (ndim > 2) {
        reduction_min.fetch_min(reduction_idx, i, vel);
        reduction_max.fetch_max(reduction_idx, i, vel);
      }
    }
    return;
  }

public:
  int weight_ind, velocity_ind;
};

/**
 * @brief Host-side Vranic merging algorithm kernels
 *
 * Required properties are the particle weights and velocity
 *
 * @tparam ndim The dimensionality of the velocity space
 */
template <size_t ndim>
struct VranicMergingKernels
    : DownsamplingKernelBase<DownsamplingMode::merging,
                             VranicReductionOnDevice<ndim>,
                             VranicMergingOnDevice<ndim>> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};

  /**
   * @brief Constructor for host-side VranicMergingKernels object
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used when remapping property names - here weight and velocity
   */
  VranicMergingKernels(
      std::map<int, std::string> properties_map = get_default_map())
      : DownsamplingKernelBase<DownsamplingMode::merging,
                               VranicReductionOnDevice<ndim>,
                               VranicMergingOnDevice<ndim>>(
            Properties<REAL>(required_simple_real_props), properties_map) {

    static_assert(ndim == 2 || ndim == 3,
                  "Only 2D and 3D ndim supported for Vranic merging strategy");

    this->downsampling_on_device_obj = VranicMergingOnDevice<ndim>();
    this->reduction_on_device_obj = VranicReductionOnDevice<ndim>();

    this->downsampling_on_device_obj->velocity_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.velocity));

    this->downsampling_on_device_obj->weight_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.weight));

    this->reduction_on_device_obj->velocity_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.velocity));

    this->reduction_on_device_obj->weight_ind =
        this->required_real_props.find_index(
            this->properties_map.at(props.weight));
  }
};

/**
 * @brief Helper function for generating a Vranic merging strategy
 *
 * @param template_group The template group sharing the domain and sycl target
 * of the particle group to which the transformation strategy is to be applied
 * @param num_merging_groups The number of merging groups, i.e. velocity space
 * bins or other downsampling group types
 * @param properties_map (Optional) A std::map<int, std::string> object to be
 * used when remapping property names
 */
template <size_t ndim>
inline std::shared_ptr<TransformationStrategy> make_vranic_merging_strategy(
    ParticleGroupSharedPtr template_group, size_t num_merging_groups,
    const std::map<int, std::string> &properties_map = get_default_map()) {

  auto r = std::make_shared<DownsamplingStrategy<VranicMergingKernels<ndim>>>(
      template_group, VranicMergingKernels<ndim>(properties_map),
      num_merging_groups, properties_map);
  return std::dynamic_pointer_cast<TransformationStrategy>(r);
};
}; // namespace VANTAGE::Reactions
#endif
