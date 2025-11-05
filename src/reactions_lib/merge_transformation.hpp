#ifndef REACTIONS_MERGE_TRANSFORMATION_H
#define REACTIONS_MERGE_TRANSFORMATION_H

#include "common_markers.hpp"
#include "particle_properties_map.hpp"
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
 * @brief  Implementation of simplified merging algorithm from M. Vranic et
 * al. Computer Physics Communications 191 2015.
 *
 * The assumption is that all particles being merged are of the same species,
 * i.e. have the same mass and that they are non-relativistic.
 *
 * Instead of merging cell-wise in momentum space, the entire space is treated
 * as one cell. In 3D the bounding box of the subgroup in momentum space is used
 * to compute the plane in which the momenta of the merged particles will lie.
 *
 * Particles are merged cell-wise into 2 particles. The properties modified are
 * the positions, weights, and momenta/velocities. Other properties are sampled
 * from 2 other particles in the passed subgroup, i.e. things like cell or
 * particle ids will be copied consistently, but there is no reduction of other
 * real quantities. This means that those values will be lost, so this algorithm
 * should be called only AFTER they are no longer needed.
 *
 * @tparam ndim dimension parameter - 2 and 3 supported
 */
template <int ndim>
struct MergeTransformationStrategy : TransformationStrategy {

  /**
   * @brief Constructor for MergeTransformationStrategy.
   *
   * @param properties_map (Optional) A std::map<int, std::string> object to be
   * used to remap the syms for the position, weight and velocity properties.
   */
  MergeTransformationStrategy(
      const std::map<int, std::string> &properties_map = get_default_map())
      : min_npart_marker(MinimumNPartInCellMarker(3)) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsitencies with indexing of properties.");

    this->position = Sym<REAL>(properties_map.at(default_properties.position));
    this->weight = Sym<REAL>(properties_map.at(default_properties.weight));
    this->momentum = Sym<REAL>(properties_map.at(default_properties.velocity));

    static_assert(ndim == 2 || ndim == 3,
                  "Only 2D and 3D merging strategies supported");
  }

  /**
   * @brief Perform merging on given subgroup. Will remove the subgroup and add
   * 2 particles per cell.
   *
   * @param target_subgroup
   */
  void transform_v(ParticleSubGroupSharedPtr target_subgroup) override {
    auto part_group = target_subgroup->get_particle_group();
    int cell_count = part_group->domain->mesh->get_cell_count();
    bool ndim_check = part_group->domain->mesh->get_ndim() == ndim;
    NESOASSERT(ndim_check,
               "The number of dimensions of the target_subgroup's mesh does "
               "not match the value of the template parameter: ndim");

    /*Buffers for the reduction quantities. Scalars are stored in the order
    weight,total energy

    We assume that all of the particles have the same mass, so no normalization
    is needed by the algorithm.
    */
    auto cell_dat_reduction_scalars = std::make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 2, 1);
    auto cell_dat_reduction_pos = std::make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, ndim, 1);
    auto cell_dat_reduction_mom = std::make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, ndim, 1);
    auto cell_dat_reduction_mom_min = std::make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 3, 1);
    auto cell_dat_reduction_mom_max = std::make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 3, 1);

    if constexpr (ndim == 2) {
      auto reduction_loop = particle_loop(
          "merge_reduction_loop", target_subgroup,
          [=](auto X, auto W, auto P, auto GA_s, auto GA_pos, auto GA_mom) {
            GA_s.combine(0, 0, W[0]);
            for (int i = 0; i < ndim; i++) {
              GA_pos.combine(i, 0, W[0] * X[i]);
              GA_mom.combine(i, 0, W[0] * P[i]);
              GA_s.combine(1, 0, W[0] * P[i] * P[i]);
            }
          },
          Access::read(this->position), Access::read(this->weight),
          Access::read(this->momentum),
          Access::reduce(cell_dat_reduction_scalars, Kernel::plus<REAL>()),
          Access::reduce(cell_dat_reduction_pos, Kernel::plus<REAL>()),
          Access::reduce(cell_dat_reduction_mom, Kernel::plus<REAL>()));

      reduction_loop->execute();
    }

    if constexpr (ndim == 3) {

      cell_dat_reduction_mom_min->fill(std::numeric_limits<REAL>::max());
      cell_dat_reduction_mom_max->fill(std::numeric_limits<REAL>::min());

      auto reduction_loop = particle_loop(
          "merge_reduction_loop_3D", target_subgroup,
          [=](auto X, auto W, auto P, auto GA_s, auto GA_pos, auto GA_mom,
              auto GA_mom_min, auto GA_mom_max) {
            GA_s.combine(0, 0, W[0]);
            for (int i = 0; i < ndim; i++) {
              GA_pos.combine(i, 0, W[0] * X[i]);
              GA_mom.combine(i, 0, W[0] * P[i]);
              GA_s.combine(1, 0, W[0] * P[i] * P[i]);

              GA_mom_min.combine(i, 0, P[i]);
              GA_mom_max.combine(i, 0, P[i]);
            }
          },
          Access::read(this->position), Access::read(this->weight),
          Access::read(this->momentum),
          Access::reduce(cell_dat_reduction_scalars, Kernel::plus<REAL>()),
          Access::reduce(cell_dat_reduction_pos, Kernel::plus<REAL>()),
          Access::reduce(cell_dat_reduction_mom, Kernel::plus<REAL>()),
          Access::reduce(cell_dat_reduction_mom_min, Kernel::minimum<REAL>()),
          Access::reduce(cell_dat_reduction_mom_max, Kernel::maximum<REAL>()));

      reduction_loop->execute();
    }

    // Get the number of particles in target_subgroup in a CellDatConst for each
    // cell.
    auto cell_dat_target_subgroup_npart_cell =
        std::make_shared<CellDatConst<int>>(part_group->sycl_target, cell_count,
                                            1, 1);
    get_npart_cell(target_subgroup, cell_dat_target_subgroup_npart_cell, 0, 0);

    // Creates a static sub group that selects at most the first two particles
    // in the target sub group for each cell.
    auto sub_group_first_two_particles =
        particle_sub_group_truncate(target_subgroup, 2, true);
    // Creates a static sub group that discards the first 2 particles in each
    // cell of the target sub group and selects any remaining particles.
    auto sub_group_to_remove_particles =
        particle_sub_group_discard(target_subgroup, 2, true);

    if constexpr (ndim == 2) {
      particle_loop(
          "MergeTransformationStrategy::transform_2D",
          sub_group_first_two_particles,
          [=](auto INDEX, auto X, auto W, auto P, auto CDC_s, auto CDC_pos,
              auto CDC_mom, auto CDC_npart_cell) {
            if (CDC_npart_cell.at(0, 0) > 2) {
              REAL merge_pos[ndim];
              REAL mom_tot[ndim];
              REAL mom_a[ndim];
              REAL mom_b[ndim];
              const REAL wt = CDC_s.at(0, 0);
              const REAL one_over_wt = 1.0 / wt;
              const REAL et = CDC_s.at(1, 0);
              for (int dimx = 0; dimx < ndim; dimx++) {
                merge_pos[dimx] = CDC_pos.at(dimx, 0) * one_over_wt;
                mom_tot[dimx] = CDC_mom.at(dimx, 0);
                mom_a[dimx] = mom_tot[dimx] * one_over_wt;
                mom_b[dimx] = mom_tot[dimx] * one_over_wt;
              }

              const REAL pt =
                  Kernel::sqrt(Kernel::dot_product_2d(mom_tot, mom_tot));

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
              const REAL p_perp_over_pt = p_perp / pt;
              mom_a[0] -= mom_tot[1] * p_perp_over_pt;
              mom_a[1] += mom_tot[0] * p_perp_over_pt;
              mom_b[0] += mom_tot[1] * p_perp_over_pt;
              mom_b[1] -= mom_tot[0] * p_perp_over_pt;

              const auto i = INDEX.loop_layer;
              X.at(0) = merge_pos[0];
              X.at(1) = merge_pos[1];
              W.at(0) = wt * 0.5;

              P.at(0) = (i == 0) ? mom_a[0] : mom_b[0];
              P.at(1) = (i == 0) ? mom_a[1] : mom_b[1];
            }
          },
          Access::read(ParticleLoopIndex{}), Access::write(this->position),
          Access::write(this->weight), Access::write(this->momentum),
          Access::read(cell_dat_reduction_scalars),
          Access::read(cell_dat_reduction_pos),
          Access::read(cell_dat_reduction_mom),
          Access::read(cell_dat_target_subgroup_npart_cell))
          ->execute();

    } else if constexpr (ndim == 3) {

      particle_loop(
          "MergeTransformationStrategy::transform_3D",
          sub_group_first_two_particles,
          [=](auto INDEX, auto X, auto W, auto P, auto CDC_s, auto CDC_pos,
              auto CDC_mom, auto CDC_mom_min, auto CDC_mom_max,
              auto CDC_npart_cell) {
            if (CDC_npart_cell.at(0, 0) > 2) {
              REAL merge_pos[ndim];
              REAL mom_tot[ndim];
              REAL mom_a[ndim];
              REAL mom_b[ndim];
              const REAL wt = CDC_s.at(0, 0);
              const REAL one_over_wt = 1.0 / wt;
              const REAL et = CDC_s.at(1, 0);
              for (int dimx = 0; dimx < ndim; dimx++) {
                merge_pos[dimx] = CDC_pos.at(dimx, 0) * one_over_wt;
                mom_tot[dimx] = CDC_mom.at(dimx, 0);
                mom_a[dimx] = mom_tot[dimx] * one_over_wt;
                mom_b[dimx] = mom_tot[dimx] * one_over_wt;
              }

              const REAL pt =
                  Kernel::sqrt(Kernel::dot_product_3d(mom_tot, mom_tot));

              // et/wt is the momentum**2 for either of the result particles,
              // and pt/wt is the momentum in the direction of the total
              // momentum vector so the below is the perpendicular momentum of
              // the resulting particles
              const REAL p_perp2 =
                  Kernel::max((et / wt) - ((pt * pt) / (wt * wt)), 0.0);
              const REAL p_perp = Kernel::sqrt(p_perp2);

              REAL mom_cell_diag[3] = {
                  CDC_mom_max.at(0, 0) - CDC_mom_min.at(0, 0),
                  CDC_mom_max.at(1, 0) - CDC_mom_min.at(1, 0),
                  CDC_mom_max.at(2, 0) - CDC_mom_min.at(2, 0)};

              REAL rotation_axis[3] = {0, 0, 0};
              Kernel::cross_product(mom_tot[0], mom_tot[1], mom_tot[2],
                                    mom_cell_diag[0], mom_cell_diag[1],
                                    mom_cell_diag[2], rotation_axis,
                                    rotation_axis + 1, rotation_axis + 2);

              // the cross product of the total momentum and the momentum space
              // bounding box diagonal of the subgroup
              REAL rotation_axis_norm = Kernel::sqrt(
                  Kernel::dot_product_3d(rotation_axis, rotation_axis));

              const REAL mom_cell_diag_norm = Kernel::sqrt(
                  Kernel::dot_product_3d(mom_cell_diag, mom_cell_diag));

              if (rotation_axis_norm / (pt * mom_cell_diag_norm) < 1e-10) {
                mom_cell_diag[0] = -mom_cell_diag[0];
                Kernel::cross_product(mom_tot[0], mom_tot[1], mom_tot[2],
                                      mom_cell_diag[0], mom_cell_diag[1],
                                      mom_cell_diag[2], rotation_axis,
                                      rotation_axis + 1, rotation_axis + 2);
                rotation_axis_norm = Kernel::sqrt(
                    Kernel::dot_product_3d(rotation_axis, rotation_axis));
              }

              // the 3D 90deg rotation matrix used here is
              // [[0 -u_3 u_2][u_3 0 -u_1][-u_2 u_1 0]] where u is the rotation
              // axis this is the cross product matrix of the rotation axis -
              // hence
              REAL mom_perp[3] = {0, 0, 0};
              Kernel::cross_product(rotation_axis[0], rotation_axis[1],
                                    rotation_axis[2], mom_tot[0], mom_tot[1],
                                    mom_tot[2], mom_perp, mom_perp + 1,
                                    mom_perp + 2);

              const REAL scaling_factor = p_perp / (pt * rotation_axis_norm);
              for (int i = 0; i < 3; i++) {
                mom_perp[i] *= scaling_factor;
              }

              for (int i = 0; i < 3; i++) {
                mom_a[i] += mom_perp[i];
                mom_b[i] -= mom_perp[i];
              }

              const auto i = INDEX.loop_layer;
              X.at(0) = merge_pos[0];
              X.at(1) = merge_pos[1];
              X.at(2) = merge_pos[2];
              W.at(0) = wt * 0.5;

              P.at(0) = (i == 0) ? mom_a[0] : mom_b[0];
              P.at(1) = (i == 0) ? mom_a[1] : mom_b[1];
              P.at(2) = (i == 0) ? mom_a[2] : mom_b[2];
            }
          },
          Access::read(ParticleLoopIndex{}), Access::write(this->position),
          Access::write(this->weight), Access::write(this->momentum),
          Access::read(cell_dat_reduction_scalars),
          Access::read(cell_dat_reduction_pos),
          Access::read(cell_dat_reduction_mom),
          Access::read(cell_dat_reduction_mom_min),
          Access::read(cell_dat_reduction_mom_max),
          Access::read(cell_dat_target_subgroup_npart_cell))
          ->execute();
    }

    part_group->remove_particles(sub_group_to_remove_particles);
  }

private:
  Sym<REAL> position;
  Sym<REAL> weight;
  Sym<REAL> momentum;
  MinimumNPartInCellMarker min_npart_marker;
};
} // namespace VANTAGE::Reactions
#endif
