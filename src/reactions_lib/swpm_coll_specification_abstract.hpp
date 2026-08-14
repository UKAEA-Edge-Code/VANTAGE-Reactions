#ifndef REACTIONS_SWPM_COLL_SPECIFICATION_H
#define REACTIONS_SWPM_COLL_SPECIFICATION_H
#include "collision_cell_manager.hpp"
#include <limits>
#include <neso_particles.hpp>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

using DEFAULT_RNG_KERNEL = NullKernelRNG<REAL>;

/**
 * @brief Abstract base class for SWPM collision specification, defining
 * intensity bounds and weight transfer
 *
 */
template <typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL>
struct AbstractSWPMSpecification {

  AbstractSWPMSpecification(
      std::shared_ptr<RNG_KERNEL_T> rng_kernel, REAL rate_norm_const = 1.0,
      const std::map<int, std::string> &properties_map = get_default_map())
      : rng_kernel(rng_kernel), norm_const(rate_norm_const) {

    this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));
    this->total_reaction_rate_sym =
        Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));
    this->coll_cell_sym =
        Sym<INT>(properties_map.at(default_properties.collision_cell_id));
    this->cell_id_sym = Sym<INT>(properties_map.at(default_properties.cell_id));
    this->panic_sym = Sym<INT>(properties_map.at(default_properties.panic));
    this->weight_change_sym =
        Sym<REAL>(properties_map.at(default_properties.weight_change));
  }

  /**
   * @brief Populate the weight change sym on the particles based
   *
   * @param pair_list Collision pair list
   * @param cell_idx_start Cell index from which to invoke the corresponding
   * pair loop
   * @param cell_idx_end Cell index to which to invoke the corresponding
   * pair loop
   */
  template <typename TARGET, typename PAIR_LIST>
  void calculate_weight_transfer(
      CellwisePairListAbsolute<TARGET, PAIR_LIST> pair_list, INT cell_idx_start,
      INT cell_idx_end) {};

  /**
   * @brief Calculate the intensity bound component without the volume of
   * sigma*v bound contribution.
   *
   * @param target Particle subgroup containing all particles of the two species
   * @param ccm Collision cell manager responsible for the two species
   * @param species_id_a Species ID of the first particle
   * @param species_id_b Species ID of the second particle
   * @param sigma_v_bound Buffer containing the total collision cell-wise bound
   * on sigma*v
   * @param result_buffer Buffer to store the q_hat result in
   */
  virtual void calculate_q_hat(ParticleSubGroupSharedPtr target,
                               std::shared_ptr<CollisionCellManager> &ccm,
                               int species_id_a, int species_id_b,
                               NDLocalArraySharedPtr<REAL, 2> &sigma_v_bound,
                               NDLocalArraySharedPtr<REAL, 2> &result_buffer) {}

  /**
   * @brief Calculate the exponential parameter for the Markov process of the
   * collisions. This is the inverse of the collision time and can be used to
   * calculate the number of pairs that need to be sampled in one timestep of a
   * given length (NTC).
   *
   * @param target Particle subgroup containing all particles of the two species
   * @param ccm Collision cell manager responsible for the two species
   * @param species_id_a Species ID of the first particle
   * @param species_id_b Species ID of the second particle
   * @param sigma_v_bound Buffer containing the total collision cell-wise bound
   * on sigma*v
   * @param result_buffer Buffer to store the result in
   */
  void calculate_exponential_parameter(
      ParticleSubGroupSharedPtr target,
      std::shared_ptr<CollisionCellManager> &ccm, int species_id_a,
      int species_id_b, NDLocalArraySharedPtr<REAL, 2> &sigma_v_bound,
      NDLocalArraySharedPtr<REAL, 2> &result_buffer,
      NDLocalArraySharedPtr<REAL, 2> &timestep_bounds) {

    auto Na = ccm->get_npart_coll_cell(target, species_id_a);
    auto Nb = ccm->get_npart_coll_cell(target, species_id_b);
    auto vols = ccm->get_coll_cell_volumes();

    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   this->N_a);
    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   this->N_b);
    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   this->volumes);
    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   this->q_hat);

    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   result_buffer);
    ccm->resize_coll_cellwise_data(target->get_particle_group()->sycl_target,
                                   timestep_bounds);
    this->N_a->set(Na);
    this->N_b->set(Nb);
    this->volumes->set(vols);
    this->sigma_v_bound = sigma_v_bound;
    this->calculate_q_hat(target, ccm, species_id_a, species_id_b,
                          sigma_v_bound, this->q_hat);

    REAL prefactor =
        (species_id_a == species_id_b ? 2.0 : 4.0) * M_PI * this->norm_const;

    nd_local_array_loop_element_wise(
        result_buffer,
        [=](int Na, int Nb, REAL vol, REAL sigmav, REAL q) {
          REAL num_prefactor =
              prefactor *
              (species_id_a == species_id_b ? Na * (Nb - 1) : Na * Nb);
          return num_prefactor * sigmav * q / vol;
        },
        N_a, N_b, volumes, sigma_v_bound, q_hat);

    REAL limit = std::numeric_limits<REAL>::max();
    nd_local_array_loop_element_wise(
        timestep_bounds,
        [=](int Na, int Nb, REAL rate_bound) {
          REAL prefactor = (species_id_a == species_id_b ? 0.5 : 1.0);
          INT offset = (species_id_a == species_id_b ? Na % 2 : 0);
          return rate_bound > 0
                     ? prefactor * Kernel::min((Na - offset) / rate_bound,
                                               Nb / rate_bound)
                     : limit;
        },
        N_a, N_b, result_buffer);
  };

protected:
  NDLocalArraySharedPtr<int, 2> N_a;
  NDLocalArraySharedPtr<int, 2> N_b;
  NDLocalArraySharedPtr<REAL, 2> volumes;
  NDLocalArraySharedPtr<REAL, 2> sigma_v_bound;
  NDLocalArraySharedPtr<REAL, 2> q_hat;
  std::shared_ptr<RNG_KERNEL_T> rng_kernel;

  Sym<REAL> weight_sym;
  Sym<REAL> weight_change_sym;
  Sym<REAL> total_reaction_rate_sym;
  Sym<INT> coll_cell_sym;
  Sym<INT> cell_id_sym;
  Sym<INT> panic_sym;

  REAL norm_const;
};
}; // namespace VANTAGE::Reactions
#endif
