#ifndef REACTIONS_SWPM_DSMC_COLL_SPECIFICATION_H
#define REACTIONS_SWPM_DSMC_COLL_SPECIFICATION_H
#include "../swpm_coll_specification_abstract.hpp"
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief Classic DSMC-style weight transfer and intensity bounds for the SWPM.
 * For identically weighted particles this fully reduces to DSMC. For different
 * weights, the smaller particle is consumed.
 *
 */
struct SWPMDSMCSpecification
    : AbstractSWPMSpecification<HostAtomicBlockKernelRNG<REAL>> {

  SWPMDSMCSpecification(
      std::shared_ptr<HostAtomicBlockKernelRNG<REAL>> rng_kernel,
      const std::map<int, std::string> &properties_map = get_default_map())
      : AbstractSWPMSpecification(rng_kernel, properties_map) {};

  template <typename TARGET, typename PAIR_LIST>
  void calculate_weight_transfer(
      CellwisePairListAbsolute<TARGET, PAIR_LIST> pair_list, INT cell_idx_start,
      INT cell_idx_end) {

    particle_pair_loop(
        "DSMC_weight_transfer", pair_list,
        [](auto index, auto weight_change_a, auto weight_change_b,
           auto max_sigma_v, auto sigma_v, auto weight_a, auto weight_b,
           auto cell, auto coll_cell, auto w_max, auto rng, auto panic_a,
           auto panic_b) {
          REAL min_weight = Kernel::min(weight_a[0], weight_b[0]);
          REAL max_weight = Kernel::max(weight_a[0], weight_b[0]);

          REAL rejection_threshold =
              sigma_v[0] / max_sigma_v.at(cell[0], coll_cell[0]) * max_weight /
              w_max.at(cell[0], coll_cell[0]);

          bool is_kernel_valid = true;
          REAL u = rng.at(index, 0, &is_kernel_valid);

          weight_change_a[0] =
              (u <= rejection_threshold && is_kernel_valid) ? min_weight : 0.0;
          weight_change_b[0] =
              (u <= rejection_threshold && is_kernel_valid) ? min_weight : 0.0;
          if (!is_kernel_valid) {
            panic_a[0] += 1;
            panic_b[0] += 1;
          }
        },

        Access::read(ParticlePairLoopIndex{}),
        Access::A(Access::write(this->weight_change_sym)),
        Access::B(Access::write(this->weight_change_sym)),
        Access::read(this->sigma_v_bound),
        Access::A(Access::read(this->total_reaction_rate_sym)),
        Access::A(Access::read(this->weight_sym)),
        Access::B(Access::read(this->weight_sym)),
        Access::A(Access::read(this->coll_cell_sym)),
        Access::A(Access::read(this->cell_id_sym)), Access::read(this->max_w),
        Access::read(this->rng_kernel),
        Access::A(Access::write(this->panic_sym)),
        Access::B(Access::write(this->panic_sym)))
        ->execute(cell_idx_start, cell_idx_end);
  };

  void calculate_q_hat(ParticleSubGroupSharedPtr target,
                       std::shared_ptr<CollisionCellManager> &ccm,
                       int species_id_a, int species_id_b,
                       NDLocalArraySharedPtr<REAL, 2> &sigma_v_bound,
                       NDLocalArraySharedPtr<REAL, 2> &result_buffer) override {

    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   this->max_w);
    ccm->coll_cellwise_max(target, this->weight_sym, 0, this->max_w);

    nd_local_array_loop_element_wise(
        result_buffer, [=](REAL w_max) { return w_max; }, this->max_w);
  }

private:
  NDLocalArraySharedPtr<REAL, 2> max_w;
};

}; // namespace VANTAGE::Reactions
#endif
