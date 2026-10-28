#ifndef REACTIONS_SWPM_COLL_SPECIFICATION_H
#define REACTIONS_SWPM_COLL_SPECIFICATION_H
#include "collision_cell_manager.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <limits>
#include <type_traits>
#include <utility>

namespace VANTAGE::Reactions {

using DEFAULT_RNG_KERNEL = NP::NullKernelRNG<REAL>;

template <typename SPEC_T,
          typename PAIR_LIST_T = NP::CellwisePairListAbsolute<
              NP::ParticleGroup, NP::CellwisePairList>,
          typename = void>
struct has_swpm_weight_transfer : std::false_type {};

template <typename SPEC_T, typename PAIR_LIST_T>
struct has_swpm_weight_transfer<
    SPEC_T, PAIR_LIST_T,
    std::void_t<decltype(std::declval<SPEC_T &>().calculate_weight_transfer(
        std::declval<PAIR_LIST_T &>(), std::declval<INT>(),
        std::declval<INT>()))>> : std::true_type {};

template <typename SPEC_T,
          typename PAIR_LIST_T = NP::CellwisePairListAbsolute<
              NP::ParticleGroup, NP::CellwisePairList>>
inline constexpr bool has_swpm_weight_transfer_v =
    has_swpm_weight_transfer<SPEC_T, PAIR_LIST_T>::value;

/**
 * @brief Abstract base class for SWPM collision specification, defining
 * intensity bounds and weight transfer.
 *
 * Derived specification classes must:
 * 1. Override `calculate_q_hat` to compute the collision-cell-wise intensity
 *    bound factor `q_hat` (excluding the volume and `sigma * v` bound).
 * 2. Define a `calculate_weight_transfer(pair_list, cell_idx_start,
 *    cell_idx_end)` member (or member template) that populates
 *    `this->weight_change_sym` for each sampled collision pair. Because
 *    `calculate_weight_transfer` is templated on the pair-list types rather
 *    than virtual, the base declaration is deleted so derived specifications
 *    cannot silently inherit a no-op implementation.
 *
 * @tparam RNG_KERNEL_T Type of the kernel RNG used by the specification.
 */
template <typename RNG_KERNEL_T = DEFAULT_RNG_KERNEL>
struct AbstractSWPMSpecification {

  virtual ~AbstractSWPMSpecification() = default;

  AbstractSWPMSpecification(
      std::shared_ptr<RNG_KERNEL_T> rng_kernel, REAL rate_norm_const = 1.0,
      const std::map<int, std::string> &properties_map = get_default_map())
      : rng_kernel(rng_kernel), norm_const(rate_norm_const) {

    this->weight_sym =
        NP::Sym<REAL>(properties_map.at(default_properties.weight));
    this->total_reaction_rate_sym =
        NP::Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));
    this->coll_cell_sym =
        NP::Sym<INT>(properties_map.at(default_properties.collision_cell_id));
    this->cell_id_sym =
        NP::Sym<INT>(properties_map.at(default_properties.cell_id));
    this->panic_sym = NP::Sym<INT>(properties_map.at(default_properties.panic));
    this->weight_change_sym =
        NP::Sym<REAL>(properties_map.at(default_properties.weight_change));
  }

  /**
   * @brief Populate the weight change sym on the particles based on the
   * sampled collision pairs.
   *
   * Derived specification classes must provide their own implementation of
   * `calculate_weight_transfer` hiding this deleted base template.
   *
   * @tparam TARGET Particle group or subgroup target type of the pair list
   * @tparam PAIR_LIST Underlying cellwise pair-list type
   * @param pair_list Collision pair list
   * @param cell_idx_start Cell index from which to invoke the corresponding
   * pair loop
   * @param cell_idx_end Cell index to which to invoke the corresponding
   * pair loop
   */
  template <typename TARGET, typename PAIR_LIST>
  void calculate_weight_transfer(
      NP::CellwisePairListAbsolute<TARGET, PAIR_LIST> pair_list,
      INT cell_idx_start, INT cell_idx_end) = delete;

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
  virtual void
  calculate_q_hat(NP::ParticleSubGroupSharedPtr target,
                  std::shared_ptr<CollisionCellManager> &ccm, int species_id_a,
                  int species_id_b,
                  NP::NDLocalArraySharedPtr<REAL, 2> &sigma_v_bound,
                  NP::NDLocalArraySharedPtr<REAL, 2> &result_buffer) = 0;

  /**
   * @brief Calculate the exponential parameter for the Markov process of the
   * collisions. This is the inverse of the collision time and can be used to
   * calculate the number of pairs that need to be sampled in one timestep of a
   * given length (NTC).
   *
   * NOTE: This assumes that sigma_v is based on the total collision
   * cross-section, not the differential cross section as in the original
   * Rjasanow paper
   *
   * @param target Particle subgroup containing all particles of the two species
   * @param ccm Collision cell manager responsible for the two species
   * @param species_id_a Species ID of the first particle
   * @param species_id_b Species ID of the second particle
   * @param sigma_v_bound Buffer containing the total collision cell-wise bound
   * on sigma*v
   * @param result_buffer Buffer to store the result in
   * @param timestep_bounds Buffer to store the timestep bounds in
   */
  void calculate_exponential_parameter(
      NP::ParticleSubGroupSharedPtr target,
      std::shared_ptr<CollisionCellManager> &ccm, int species_id_a,
      int species_id_b, NP::NDLocalArraySharedPtr<REAL, 2> &sigma_v_bound,
      NP::NDLocalArraySharedPtr<REAL, 2> &result_buffer,
      NP::NDLocalArraySharedPtr<REAL, 2> &timestep_bounds) {

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
        (species_id_a == species_id_b ? 0.5 : 1.0) * this->norm_const;

    NP::nd_local_array_loop_element_wise(
        result_buffer,
        [=](int Na, int Nb, REAL vol, REAL sigmav, REAL q) {
          REAL num_prefactor =
              prefactor *
              (species_id_a == species_id_b ? Na * (Nb - 1) : Na * Nb);

          return num_prefactor * sigmav * q / vol;
        },
        N_a, N_b, volumes, sigma_v_bound, q_hat);

    REAL limit = std::numeric_limits<REAL>::max();
    NP::nd_local_array_loop_element_wise(
        timestep_bounds,
        [=](int Na, int Nb, REAL rate_bound) {
          REAL prefactor = (species_id_a == species_id_b ? 0.5 : 1.0);
          INT offset = (species_id_a == species_id_b ? Na % 2 : 0);
          return rate_bound > 0
                     ? prefactor * NP::Kernel::min((Na - offset) / rate_bound,
                                                   Nb / rate_bound)
                     : limit;
        },
        N_a, N_b, result_buffer);
  };

protected:
  NP::NDLocalArraySharedPtr<int, 2> N_a;
  NP::NDLocalArraySharedPtr<int, 2> N_b;
  NP::NDLocalArraySharedPtr<REAL, 2> volumes;
  NP::NDLocalArraySharedPtr<REAL, 2> sigma_v_bound;
  NP::NDLocalArraySharedPtr<REAL, 2> q_hat;
  std::shared_ptr<RNG_KERNEL_T> rng_kernel;

  NP::Sym<REAL> weight_sym;
  NP::Sym<REAL> weight_change_sym;
  NP::Sym<REAL> total_reaction_rate_sym;
  NP::Sym<INT> coll_cell_sym;
  NP::Sym<INT> cell_id_sym;
  NP::Sym<INT> panic_sym;

  REAL norm_const;
};
}; // namespace VANTAGE::Reactions
#endif
