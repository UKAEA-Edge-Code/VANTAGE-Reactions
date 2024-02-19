#ifndef MERGE_TRANSFORMATION_H
#define MERGE_TRANSFORMATION_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <neso_particles.hpp>
#include <transformation_wrapper.hpp>
#include <utils.hpp>
#include <vector>

using namespace NESO::Particles;

namespace Reactions {
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
 * @tparam ndim dimesion parameter - 2 and 3 supported
 */
template <int ndim>
struct MergeTransformationStrategy : TransformationStrategy {

  MergeTransformationStrategy() = delete;

  MergeTransformationStrategy(const Sym<REAL> &position,
                              const Sym<REAL> &weight,
                              const Sym<REAL> &momentum)
      : position(position), weight(weight), momentum(momentum) {
    static_assert(ndim == 2 || ndim == 3,
                  "Only 2D and 3D merging strategies supported");
  };
  /**
   * @brief Perform merging on given subgroup. Will remove the subgroup and add
   * 2 particles per cell.
   *
   * @param target_subgroup
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {
    auto part_group = target_subgroup->get_particle_group();
    int cell_count = part_group->domain->mesh->get_cell_count();
    auto new_particle_group = make_shared<ParticleGroup>(part_group->domain,part_group->particle_spec,part_group->sycl_target);
    // TODO: better asserts (maybe use NESOASSERT?)
    assert(part_group->domain->mesh->get_ndim() == ndim);

    /*Buffers for the reduction quantities. Scalars are stored in the order
    weight,total energy

    We assume that all of the particles have the same mass, so no normalization
    is needed by the algorithm.
    */
    auto cell_dat_reduction_scalars = make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 2, 1);
    auto cell_dat_reduction_pos = make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, ndim, 1);
    auto cell_dat_reduction_mom = make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, ndim, 1);
    auto cell_dat_reduction_mom_min = make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 3, 1);
    auto cell_dat_reduction_mom_max = make_shared<CellDatConst<REAL>>(
        part_group->sycl_target, cell_count, 3, 1);

    if constexpr (ndim == 2) {
      auto reduction_loop = particle_loop(
          "merge_reduction_loop", target_subgroup,
          [=](auto X, auto W, auto P, auto GA_s, auto GA_pos, auto GA_mom) {
            GA_s.fetch_add(0, 0, W[0]);
            for (int i = 0; i < ndim; i++) {
              GA_pos.fetch_add(i, 0, W[0] * X[i]);
              GA_mom.fetch_add(i, 0, W[0] * P[i]);
              GA_s.fetch_add(1, 0, W[0] * P[i] * P[i]);
            }
          },
          Access::read(this->position), Access::read(this->weight),
          Access::read(this->momentum), Access::add(cell_dat_reduction_scalars),
          Access::add(cell_dat_reduction_pos),
          Access::add(cell_dat_reduction_mom));

      reduction_loop->execute();
    }

    if constexpr (ndim == 3) {

      cell_dat_reduction_mom_min->fill(std::numeric_limits<REAL>::max());
      cell_dat_reduction_mom_max->fill(std::numeric_limits<REAL>::min());

      auto reduction_loop = particle_loop(
          "merge_reduction_loop_3D", target_subgroup,
          [=](auto X, auto W, auto P, auto GA_s, auto GA_pos, auto GA_mom,
              auto GA_mom_min, auto GA_mom_max) {
            GA_s.fetch_add(0, 0, W[0]);
            for (int i = 0; i < ndim; i++) {
              GA_pos.fetch_add(i, 0, W[0] * X[i]);
              GA_mom.fetch_add(i, 0, W[0] * P[i]);
              GA_s.fetch_add(1, 0, W[0] * P[i] * P[i]);

              GA_mom_min.fetch_min(i, 0, P[i]);
              GA_mom_max.fetch_max(i, 0, P[i]);
            }
          },
          Access::read(this->position), Access::read(this->weight),
          Access::read(this->momentum), Access::add(cell_dat_reduction_scalars),
          Access::add(cell_dat_reduction_pos),
          Access::add(cell_dat_reduction_mom),
          Access::min(cell_dat_reduction_mom_min),
          Access::max(cell_dat_reduction_mom_max));

      reduction_loop->execute();
    }

    std::vector<INT> layers = {0, 1};

    for (int cx = 0; cx < cell_count; cx++) {
      auto cell_data_scalars = cell_dat_reduction_scalars->get_cell(cx);
      auto cell_data_pos = cell_dat_reduction_pos->get_cell(cx);
      auto cell_data_mom = cell_dat_reduction_mom->get_cell(cx);

      auto merge_pos = std::vector<REAL>();
      auto mom_tot = std::vector<REAL>();
      auto mom_a = std::vector<REAL>();
      auto mom_b = std::vector<REAL>();

      REAL wt = cell_data_scalars->at(0, 0);
      REAL et = cell_data_scalars->at(1, 0);
      for (int dimx = 0; dimx < ndim; dimx++) {
        merge_pos.push_back(cell_data_pos->at(dimx, 0) / wt);
        mom_tot.push_back(cell_data_mom->at(dimx, 0));
        mom_a.push_back(mom_tot[dimx] / wt);
        mom_b.push_back(mom_tot[dimx] / wt);
      }

      REAL pt = utils::norm2(mom_tot);

      // et/wt is the momentum**2 for either of the result particles, and pt/wt
      // is the momentum in the direction of the total momentum vector so the
      // below is the perpendicular momentum of the resulting particles
      REAL p_perp = std::sqrt(et / wt - pt * pt / (wt * wt));

      if constexpr (ndim == 2) {

        // applying the the 2D 90deg rotation matrix [[0 -1][1 0]] to the total
        // momentum direction and scaling with the perpendicular momentum
        mom_a[0] -= p_perp * mom_tot[1] / pt;
        mom_a[1] += p_perp * mom_tot[0] / pt;

        mom_b[0] += p_perp * mom_tot[1] / pt;
        mom_b[1] -= p_perp * mom_tot[0] / pt;
      }
      if constexpr (ndim == 3) {
        auto cell_data_mom_min = cell_dat_reduction_mom_min->get_cell(cx);
        auto cell_data_mom_max = cell_dat_reduction_mom_max->get_cell(cx);

        // we generate the bounding box in momentum space of the subgroup and
        // set its diagonal
        std::vector<REAL> mom_cell_diag(3);

        for (int i = 0; i < 3; i++) {
          mom_cell_diag[i] =
              cell_data_mom_max->at(i, 0) - cell_data_mom_min->at(i, 0);
        }

        std::vector<REAL> rotation_axis =
            utils::cross_product(mom_tot, mom_cell_diag);

        // the cross product of the total momentum and the momentum space
        // bounding box diagonal of the subgroup

        REAL rotation_axis_norm = utils::norm2(rotation_axis);

        // Handle close to co-linear diagonal and total vector

        if (rotation_axis_norm / (pt * utils::norm2(mom_cell_diag)) < 1e-10) {
          mom_cell_diag[0] = -mom_cell_diag[0];
          rotation_axis = utils::cross_product(mom_tot, mom_cell_diag);
          rotation_axis_norm = utils::norm2(rotation_axis);
        }

        // the 3D 90deg rotation matrix used here is
        // [[0 -u_3 u_2][u_3 0 -u_1][-u_2 u_1 0]] where u is the rotation axis
        // this is the cross product matrix of the rotation axis - hence
        std::vector<REAL> mom_perp =
            utils::cross_product(rotation_axis, mom_tot);

        std::transform(mom_perp.begin(), mom_perp.end(), mom_perp.begin(),
                       std::bind(std::multiplies<REAL>(), std::placeholders::_1,
                                 p_perp / (pt * rotation_axis_norm)));
        for (int i = 0; i < 3; i++) {
          mom_a[i] += mom_perp[i];
          mom_b[i] -= mom_perp[i];
        }
      }
      // Add above particles

      std::vector<INT> cells = {cx, cx};
      auto new_particles = target_subgroup->get_particles(cells, layers);

      for (int i = 0; i < 2; i++) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          new_particles->at(this->position, i, dimx) = merge_pos[dimx];
        }
        new_particles->at(this->weight, i, 0) = wt / 2;
      }

      for (int dimx = 0; dimx < ndim; dimx++) {
        new_particles->at(this->momentum, 0, dimx) = mom_a[dimx];
        new_particles->at(this->momentum, 1, dimx) = mom_b[dimx];
      }

      new_particle_group->add_particles_local(new_particles);
    }

    // remove the marked particles

    part_group->remove_particles(target_subgroup);
    part_group->add_particles_local(new_particle_group);
  }

private:
  Sym<REAL> position;
  Sym<REAL> weight;
  Sym<REAL> momentum;
};
} // namespace Reactions
#endif
