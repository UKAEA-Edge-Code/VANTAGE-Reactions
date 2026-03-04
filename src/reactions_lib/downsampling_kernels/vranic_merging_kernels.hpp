#ifndef REACTIONS_VRANIC_MERGING_H
#define REACTIONS_VRANIC_MERGING_H

#include "reactions_lib/downsampling_base.hpp"
#include <cmath>
#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

template <size_t ndim>
struct VranicMergingOnDevice : DownsamplingKernelOnDeviceBase<2> {

  VranicMergingOnDevice() = default;

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
    const REAL wt = reduction.at(0, reduction_idx);
    const REAL one_over_wt = 1.0 / wt;
    const REAL et = reduction.at(1, reduction_idx);
    for (int dimx = 0; dimx < ndim; dimx++) {
      mom_tot[dimx] = reduction.at(2 + dimx, reduction_idx);
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

      REAL mom_cell_diag[3] = {reduction_max.at(0, reduction_idx) -
                                   reduction_min.at(0, reduction_idx),
                               reduction_max.at(1, reduction_idx) -
                                   reduction_min.at(1, reduction_idx),
                               reduction_max.at(2, reduction_idx) -
                                   reduction_min.at(2, reduction_idx)};

      REAL rotation_axis[3] = {0, 0, 0};
      Kernel::cross_product(mom_tot[0], mom_tot[1], mom_tot[2],
                            mom_cell_diag[0], mom_cell_diag[1],
                            mom_cell_diag[2], rotation_axis, rotation_axis + 1,
                            rotation_axis + 2);

      // the cross product of the total momentum and the momentum space
      // bounding box diagonal of the subgroup
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

template <size_t ndim>
struct VranicReductionOnDevice
    : ReductionKernelOnDeviceBase<ndim + 2, ndim, ndim> {

  VranicReductionOnDevice() = default;

  void reduce(const Access::SymVector::Read<INT> &req_int_props,
              const Access::SymVector::Read<REAL> &req_real_props,
              Access::CellDatConst::Add<REAL> &reduction,
              Access::CellDatConst::Min<REAL> &reduction_min,
              Access::CellDatConst::Max<REAL> &reduction_max,
              const size_t &reduction_idx) const {

    auto weight = req_real_props.at(this->weight_ind, 0);
    REAL vel;
    reduction.fetch_add(0, reduction_idx, weight);
    for (int i = 0; i < ndim; i++) {
      vel = req_real_props.at(this->velocity_ind, i);
      reduction.fetch_add(1, reduction_idx, weight * vel * vel);
      reduction.fetch_add(2 + i, reduction_idx, weight * vel);

      if constexpr (ndim > 2) {
        reduction_min.fetch_min(i, reduction_idx, vel);
        reduction_max.fetch_max(i, reduction_idx, vel);
      }
    }
    return;
  }

public:
  int weight_ind, velocity_ind;
};

template <size_t ndim>
struct VranicMergingKernels
    : DownsamplingKernelBase<DownsamplingMode::merging,
                             VranicReductionOnDevice<ndim>,
                             VranicMergingOnDevice<ndim>> {

  constexpr static auto props = default_properties;

  constexpr static std::array<int, 2> required_simple_real_props = {
      props.weight, props.velocity};

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
