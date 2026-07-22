#ifndef REACTIONS_SWPM_REACTION_H
#define REACTIONS_SWPM_REACTION_H
#include "pair_data_calculator.hpp"
#include "pair_reaction_data/cs_pair_reaction_data.hpp"
#include "pair_reaction_kernels.hpp"
#include "profiling_base.hpp"
#include "reaction_base.hpp"
#include <array>
#include <cstring>
#include <neso_particles.hpp>
#include <type_traits>
#include <vector>

// TODO: docs!

using namespace NESO::Particles;

namespace VANTAGE::Reactions {

template <int num_products, typename ReactionData, typename ReactionKernels,
          typename DataCalc = PairDataCalculator<>>
struct SWPMReaction : ProfilingBase {
  SWPMReaction() = default;

  SWPMReaction(
      SYCLTargetSharedPtr sycl_target, std::array<int, 2> reactants,
      std::array<int, num_products> products, ReactionData reaction_data,
      ReactionKernels reaction_kernels, DataCalc data_calculator,
      const std::map<int, std::string> &properties_map = get_default_map())
      : reactants(reactants), products(products), reaction_data(reaction_data),
        reaction_kernels(reaction_kernels), data_calculator(data_calculator),
        sycl_target_stored(sycl_target),
        device_rate_buffer(
            std::make_shared<LocalArray<REAL>>(sycl_target, 0, 0.0)),
        pre_req_data(
            std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 0, 0)),
        max_num_coll_cells(128), default_rel_vel(1.0), num_mesh_cells(1),
        max_buffer_size(16384 *
                        get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)) {

    NESOWARN(
        map_subset_check(properties_map),
        "The provided properties_map does not include all the keys from the \
        default_map (and therefore is not an extension of that map). There \
        may be inconsistencies with indexing of properties.");

    this->total_reaction_rate =
        Sym<REAL>(properties_map.at(default_properties.tot_reaction_rate));
    this->weight_sym = Sym<REAL>(properties_map.at(default_properties.weight));
    this->weight_change_sym =
        Sym<REAL>(properties_map.at(default_properties.weight_change));
    this->collision_cell_sym =
        Sym<INT>(properties_map.at(default_properties.collision_cell_id));
    this->cell_id_sym = Sym<INT>(properties_map.at(default_properties.cell_id));

    // These assertions are necessary since the typenames for ReactionData and
    // ReactionKernels could be any type and for calculate_rates and
    // apply to operate correctly, ReactionData and
    // ReactionKernels have to be derived from CSPairData and
    // PairReactionKernelsBase respectively
    static_assert(
        std::is_base_of_v<CSPairData<ReactionData::VEL_NDIM,
                                     typename ReactionData::CROSS_SECTION_TYPE>,
                          ReactionData>,
        "Template parameter ReactionData is not derived from "
        "CSPairData...");
    static_assert(std::is_base_of_v<AbstractPairDataCalculator, DataCalc>,
                  "Template parameter DataCalc is not derived from "
                  "AbstractPairDataCalculator...");
    static_assert(std::is_base_of_v<PairReactionKernelsBase, ReactionKernels>,
                  "Template parameter ReactionKernels is not derived from "
                  "PairReactionKernelsBase...");
    NESOASSERT(
        this->data_calculator.get_data_size() ==
            this->reaction_kernels.get_pre_ndims(),
        "The number of ReactionData-derived objects in PairDataCalculator "
        "does not match the required number of dimensions for the "
        "provided PairReactionKernels object.");

    auto reaction_data_buffer = this->reaction_data;
    auto reaction_kernel_buffer = this->reaction_kernels;

    this->calculate_rates_int_syms_a =
        reaction_data_buffer.get_arg_pack()
            .required_int_props_a.to_sym_vector();

    this->calculate_rates_real_syms_a =
        reaction_data_buffer.get_arg_pack()
            .required_real_props_a.to_sym_vector();

    this->calculate_rates_int_syms_b =
        reaction_data_buffer.get_arg_pack()
            .required_int_props_b.to_sym_vector();

    this->calculate_rates_real_syms_b =
        reaction_data_buffer.get_arg_pack()
            .required_real_props_b.to_sym_vector();

    this->apply_int_syms_a = utils::build_sym_vector<INT>(
        reaction_kernel_buffer.get_required_int_props_a());

    this->apply_real_syms_a = utils::build_sym_vector<REAL>(
        reaction_kernel_buffer.get_required_real_props_a());

    this->apply_int_syms_b = utils::build_sym_vector<INT>(
        reaction_kernel_buffer.get_required_int_props_b());

    this->apply_real_syms_b = utils::build_sym_vector<REAL>(
        reaction_kernel_buffer.get_required_real_props_b());

    auto descendant_matrix_spec_a =
        reaction_kernel_buffer.get_descendant_matrix_spec_a();

    auto descendant_matrix_spec_b =
        reaction_kernel_buffer.get_descendant_matrix_spec_b();

    this->descendant_particles_a = std::make_shared<DescendantProducts>(
        this->get_sycl_target(), descendant_matrix_spec_a,
        reaction_kernel_buffer.get_num_products_a());

    this->descendant_particles_b = std::make_shared<DescendantProducts>(
        this->get_sycl_target(), descendant_matrix_spec_b,
        reaction_kernel_buffer.get_num_products_b());

    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        SWPMReaction::get_sycl_target(), 0,
        this->data_calculator.get_data_size());
    empty_pre_req_data->fill(0);

    this->pre_req_data = empty_pre_req_data;
    this->sigma_v_bounds = std::make_shared<NDLocalArray<REAL, 2>>(
        this->sycl_target_stored, 0, this->max_num_coll_cells);
  }

  SWPMReaction(
      SYCLTargetSharedPtr sycl_target, size_t num_cells,
      std::array<int, 2> reactants, std::array<int, num_products> products,
      ReactionData reaction_data, ReactionKernels reaction_kernels,
      const std::map<int, std::string> &properties_map = get_default_map())
      : reactants(reactants), products(products), reaction_data(reaction_data),
        reaction_kernels(reaction_kernels), data_calculator(DataCalc()),
        sycl_target_stored(sycl_target),
        device_rate_buffer(
            std::make_shared<LocalArray<REAL>>(sycl_target, 0, 0.0)),
        pre_req_data(
            std::make_shared<NDLocalArray<REAL, 2>>(sycl_target, 0, 0)),
        max_num_coll_cells(128), default_rel_vel(1.0), num_mesh_cells(1),
        max_buffer_size(16384 *
                        get_env_size_t("REACTIONS_CELL_BLOCK_SIZE", 256)) {}

  virtual ~SWPMReaction() {
    if (this->submitted_sigma_v_max_loop) {
      this->sigma_v_max_loop->wait();
      this->submitted_sigma_v_max_loop = false;
    }
  };

public:
  template <typename TARGET, typename PAIR_LIST>
  void calculate_rates_v(CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
                         INT cell_idx_start, INT cell_idx_end) {

    auto reaction_data_buffer = this->reaction_data;
    auto reaction_data_on_device = reaction_data_buffer.get_on_device_obj();

    this->num_mesh_cells = pair_list.A->domain->mesh->get_cell_count();

    INT npart_block =
        pair_list.pair_list->get_num_pairs_range(cell_idx_start, cell_idx_end);
    this->adaptive_flush_buffer(npart_block);

    NESOASSERT(pair_list.A->sycl_target == sycl_target_stored,
               "sycl_target assigned to particle_group is not the same as "
               "the sycl_target passed to Reaction object...");

    auto calc_rate_loop = particle_pair_loop(
        "pair_data_calc_loop", pair_list,
        [=](auto pair_index, auto req_int_props_a, auto req_real_props_a,
            auto req_int_props_b, auto req_real_props_b, auto tot_rate,
            auto buffer, auto kernel) {
          INT current_count = pair_index.get_loop_linear_index();
          auto accessors = PairReactionDataAccessors(
              pair_index, req_int_props_a, req_real_props_a, req_int_props_b,
              req_real_props_b);
          std::array<REAL, 1> rate =
              reaction_data_on_device.calc_data(accessors, kernel);
          buffer[current_count] = rate[0];
          tot_rate[0] += rate[0];
        },
        Access::read(ParticlePairLoopIndex{}),
        Access::A(Access::write(
            sym_vector<INT>(pair_list.A, this->calculate_rates_int_syms_a))),
        Access::A(Access::read(
            sym_vector<REAL>(pair_list.A, this->calculate_rates_real_syms_a))),
        Access::B(Access::write(
            sym_vector<INT>(pair_list.B, this->calculate_rates_int_syms_b))),
        Access::B(Access::read(
            sym_vector<REAL>(pair_list.B, this->calculate_rates_real_syms_b))),
        Access::write(this->total_reaction_rate),
        Access::write(this->device_rate_buffer),
        Access::read(reaction_data.get_rng_kernel()));

    calc_rate_loop->execute(cell_idx_start, cell_idx_end);
    this->launch_sigma_v_max_loop<TARGET, PAIR_LIST>(pair_list, cell_idx_start,
                                                     cell_idx_end);
  }

  template <typename TARGET, typename PAIR_LIST>
  void launch_sigma_v_max_loop(
      CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
      INT cell_idx_start, INT cell_idx_end) {

    this->prepare_sigma_v_bounds();

    if (this->submitted_sigma_v_max_loop) {
      this->sigma_v_max_loop->wait();
      this->submitted_sigma_v_max_loop = false;
    }
    NESOASSERT(pair_list.A->sycl_target == sycl_target_stored,
               "sycl_target assigned to particle_group is not the same as "
               "the sycl_target passed to Reaction object...");
    this->sigma_v_max_loop = particle_pair_loop(
        "max_sigma_v_loop", pair_list,
        [](auto pair_index, auto max_sigma_v, auto buffer, auto coll_cell,
           auto cell_id) {
          INT current_count = pair_index.get_loop_linear_index();
          max_sigma_v.fetch_max(cell_id[0], coll_cell[0],
                                buffer[current_count]);
        },
        Access::read(ParticlePairLoopIndex{}),
        Access::max(this->sigma_v_bounds),
        Access::read(this->device_rate_buffer),
        Access::A(Access::read(this->collision_cell_sym)),
        Access::A(Access::read(this->cell_id_sym)));

    this->sigma_v_max_loop->submit(cell_idx_start, cell_idx_end);
    this->submitted_sigma_v_max_loop = true;
  }

  std::vector<std::vector<REAL>> get_sigma_v_bounds(INT cell_idx_start,
                                                    INT cell_idx_end) {

    // This makes sure that we get the correct default bounds on first call, and
    // then avoids resetting the bounds - there could be issues from call order
    // if reactions are used for different meshes
    if (this->num_mesh_cells < cell_idx_end) {
      this->num_mesh_cells = cell_idx_end;
    }
    this->prepare_sigma_v_bounds();

    if (this->submitted_sigma_v_max_loop) {
      this->sigma_v_max_loop->wait();
      this->submitted_sigma_v_max_loop = false;
    }

    auto bounds = this->sigma_v_bounds->get();

    std::vector<std::vector<REAL>> bound_vec;
    bound_vec.reserve(cell_idx_end - cell_idx_start);

    for (int i = cell_idx_start; i < cell_idx_end; i++) {
      bound_vec.emplace_back(bounds.begin() + i * this->max_num_coll_cells,
                             bounds.begin() +
                                 (i + 1) * this->max_num_coll_cells);
    }

    return bound_vec;
  }

  template <typename TARGET, typename PAIR_LIST>
  void calculate_rates(CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
                       INT cell_idx_start, INT cell_idx_end) {
    auto r0 = this->start_profiling_region(this->sycl_target_stored,
                                           "calculate_rates_SWPM");
    this->calculate_rates_v(pair_list, cell_idx_start, cell_idx_end);
    this->end_profiling_region(sycl_target_stored, r0);
  }

  template <typename TARGET, typename PAIR_LIST>
  void apply_v(CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
               INT cell_idx_start, INT cell_idx_end, double dt,
               ParticleGroupSharedPtr child_group) {

    auto reaction_kernel_buffer = this->reaction_kernels;
    auto reaction_kernel_on_device = reaction_kernel_buffer.get_on_device_obj();

    std::array<int, num_products> products = this->products;
    NESOASSERT(pair_list.A->sycl_target == sycl_target_stored,
               "sycl_target assigned to particle_group is not the same as "
               "the sycl_target passed to Reaction object...");

    INT npart_block =
        pair_list.pair_list->get_num_pairs_range(cell_idx_start, cell_idx_end);
    this->adaptive_flush_pre_req_data(npart_block);

    this->data_calculator.fill_buffer(this->pre_req_data, pair_list,
                                      cell_idx_start, cell_idx_end);
    auto application_loop = particle_pair_loop(
        "descendant_products_loop", pair_list,
        [=](auto weight_change, auto descendant_particles_a,
            auto descendant_particles_b, auto particle_index_a,
            auto particle_index_b, auto pair_index, auto req_int_props_a,
            auto req_real_props_a, auto req_int_props_b, auto req_real_props_b,
            auto rate_buffer, auto pre_req_data, auto total_reaction_rate) {
          INT current_count = pair_index.get_loop_linear_index();
          REAL rate = rate_buffer.at(current_count);
          REAL total_rate = total_reaction_rate.at(0);

          REAL deltaweight = weight_change.at(0);

          REAL modified_weight = deltaweight * rate / total_rate;

          reaction_kernel_on_device.parent_kernel(
              particle_index_a, particle_index_b, descendant_particles_a,
              descendant_particles_b, products);

          reaction_kernel_on_device.scattering_kernel(
              modified_weight, particle_index_a, particle_index_b, pair_index,
              descendant_particles_a, descendant_particles_b, req_int_props_a,
              req_real_props_a, req_int_props_b, req_real_props_b, products,
              pre_req_data, dt);

          reaction_kernel_on_device.weight_kernel(
              modified_weight, particle_index_a, particle_index_b, pair_index,
              descendant_particles_a, descendant_particles_b, req_int_props_a,
              req_real_props_a, req_int_props_b, req_real_props_b, products,
              pre_req_data, dt);

          reaction_kernel_on_device.transformation_kernel(
              modified_weight, particle_index_a, particle_index_b, pair_index,
              descendant_particles_a, descendant_particles_b, req_int_props_a,
              req_real_props_a, req_int_props_b, req_real_props_b, products,
              pre_req_data, dt);

          reaction_kernel_on_device.feedback_kernel(
              modified_weight, particle_index_a, particle_index_b, pair_index,
              descendant_particles_a, descendant_particles_b, req_int_props_a,
              req_real_props_a, req_int_props_b, req_real_props_b, products,
              pre_req_data, dt);
        },
        Access::read(this->weight_change_sym),
        Access::write(this->descendant_particles_a),
        Access::write(this->descendant_particles_b),
        Access::A(Access::read(ParticlePairLoopIndex{})),
        Access::B(Access::read(ParticlePairLoopIndex{})),
        Access::read(ParticlePairLoopIndex{}),
        Access::A(Access::write(
            sym_vector<INT>(pair_list.A, this->apply_int_syms_a))),
        Access::A(Access::write(
            sym_vector<REAL>(pair_list.A, this->apply_real_syms_a))),
        Access::B(Access::write(
            sym_vector<INT>(pair_list.B, this->apply_int_syms_b))),
        Access::B(Access::write(
            sym_vector<REAL>(pair_list.B, this->apply_real_syms_b))),
        Access::read(device_rate_buffer), Access::read(this->pre_req_data),
        Access::write(this->total_reaction_rate));

    this->descendant_particles_a->reset(npart_block);
    this->descendant_particles_b->reset(npart_block);

    application_loop->execute(cell_idx_start, cell_idx_end);

    child_group->add_particles_local(this->descendant_particles_a, pair_list.A);
    child_group->add_particles_local(this->descendant_particles_b, pair_list.B);

    return;
  }

  template <typename TARGET, typename PAIR_LIST>
  void apply(CellwisePairListAbsolute<TARGET, PAIR_LIST> &pair_list,
             INT cell_idx_start, INT cell_idx_end, double dt,
             ParticleGroupSharedPtr child_group) {

    auto r0 =
        this->start_profiling_region(this->sycl_target_stored, "apply_SWPM");
    this->apply_v(pair_list, cell_idx_start, cell_idx_end, dt, child_group);
    this->end_profiling_region(this->sycl_target_stored, r0);
  }

  /**
   * @brief Set the maximum size for data buffers on this reaction
   *
   * @param max_size Maximum size (per dimension) of data buffers on this
   * reaction
   */
  void set_max_buffer_size(size_t max_size) {
    this->max_buffer_size = max_size;
  }

  /**
   * @brief Set the maximum number of collision cells per cell
   *
   * @param max_num_coll_cells Maximum number of collision cells per mesh cell
   */
  void set_max_num_coll_cells(size_t max_cells) {
    this->max_num_coll_cells = max_cells;
  }
  /**
   * @brief Creates an empty rate buffer of a specified size
   *
   * @param buffer_size Size of the empty buffer that needs to be created and
   * stored.
   */
  void flush_buffer(size_t buffer_size) {
    auto empty_device_rate_buffer = std::make_shared<LocalArray<REAL>>(
        this->sycl_target_stored, buffer_size, 0);
    this->device_rate_buffer = empty_device_rate_buffer;
  }

  void adaptive_flush_buffer(size_t requested_size) {
    auto device_rate_buffer_size = this->device_rate_buffer->size;

    if (device_rate_buffer_size < requested_size) {
      NESOASSERT(requested_size <= this->get_max_buffer_size(),
                 "Number of particles in cell exceeds the maximum reaction "
                 "buffer size");
      if ((requested_size * 2) < this->get_max_buffer_size()) {
        this->flush_buffer(requested_size * 2);
      } else {
        this->flush_buffer(this->get_max_buffer_size());
      }
    } else if (requested_size < (device_rate_buffer_size / 4)) {
      this->flush_buffer((requested_size * 2));
    }
  }

  /**
   * @brief Flushes the stored pre_req_data by setting all values to 0.0.
   */
  void flush_pre_req_data() { this->get_pre_req_data()->fill(0.0); }

  /**
   * @brief Creates an empty pre_req_data buffer of a specified size, keeping
   * the current number of columns
   *
   * @param buffer_size Number of the empty buffer rows that need to be
   * created and stored.
   */
  void flush_pre_req_data(size_t buffer_size) {
    auto shape = this->pre_req_data->index.shape;
    auto empty_pre_req_data = std::make_shared<NDLocalArray<REAL, 2>>(
        this->sycl_target_stored, buffer_size, shape[1]);
    empty_pre_req_data->fill(0);
    this->pre_req_data = empty_pre_req_data;
  }

  /**
   * @brief Flushes the pre_req_data buffer blockwise, allocating extra memory
   * if necessary.
   *
   * @param particle_sub_group Particle subgroup used to infer the number of
   * particles in the cell
   * @param cell_idx_start Index of the first cell for which the buffer flush
   * is performed
   * @param cell_idx_end Loop end index - cell up to which the buffer is
   * flushed
   */
  void adaptive_flush_pre_req_data(size_t requested_size) {
    auto shape = this->pre_req_data->index.shape;
    auto pre_req_buffer_size = shape[0];
    if (pre_req_buffer_size < requested_size) {
      NESOASSERT(requested_size <= this->get_max_buffer_size(),
                 "Requested buffer size exceeds the maximum reaction "
                 "buffer size");
      if ((requested_size * 2) < this->get_max_buffer_size()) {
        this->flush_pre_req_data(requested_size * 2);
      } else {
        this->flush_pre_req_data(this->get_max_buffer_size());
      }
    } else if (requested_size < (pre_req_buffer_size / 4)) {
      this->flush_pre_req_data((requested_size * 2));
    }
  }

  void prepare_sigma_v_bounds() {
    auto shape = this->sigma_v_bounds->index.shape;
    if (shape[0] != this->num_mesh_cells ||
        shape[1] < this->max_num_coll_cells) {
      this->sigma_v_bounds = std::make_shared<NDLocalArray<REAL, 2>>(
          this->sycl_target_stored, this->num_mesh_cells,
          this->max_num_coll_cells);
      this->sigma_v_bounds->fill(
          this->reaction_data.get_cs_max_rate_val(this->default_rel_vel));
    }
  }
  /**
   * @brief Set the default relative velocity used for rate bounds
   *
   * @param default_rel_vel Default maximum velocity used for initial rate
   * bounds
   */
  void set_default_rel_vel(size_t default_rel_vel) {
    this->default_rel_vel = default_rel_vel;
  }

protected:
  const LocalArraySharedPtr<REAL> &get_device_rate_buffer() {
    return this->device_rate_buffer;
  }

  const size_t &get_device_rate_buffer_size() {
    return this->device_rate_buffer->size;
  }

  const SYCLTargetSharedPtr &get_sycl_target() { return sycl_target_stored; }

  const NDLocalArraySharedPtr<REAL, 2> &get_pre_req_data() {
    return pre_req_data;
  }

  size_t get_max_buffer_size() { return this->max_buffer_size; }

  std::vector<int> get_out_states() {
    return std::vector<int>(this->products.begin(), this->products.end());
  }

  std::vector<int> get_in_states() {
    return std::vector<int>(this->reactants.begin(), this->reactants.end());
  }

private:
  Sym<REAL> total_reaction_rate;
  LocalArraySharedPtr<REAL> device_rate_buffer;
  SYCLTargetSharedPtr sycl_target_stored;
  NDLocalArraySharedPtr<REAL, 2>
      pre_req_data; //!< Real-valued local matrix for storing
                    //!< any pre-requisite data relating to a
                    //!< derived reaction.
  Sym<REAL> weight_sym;
  Sym<REAL> weight_change_sym;
  Sym<INT> collision_cell_sym;
  Sym<INT> cell_id_sym;
  size_t max_buffer_size; //!< max buffer size for data on the reactions object
                          //
  std::array<int, 2> reactants;
  std::array<int, num_products> products;
  ReactionData reaction_data;
  ReactionKernels reaction_kernels;
  std::shared_ptr<DescendantProducts> descendant_particles_a;
  std::shared_ptr<DescendantProducts> descendant_particles_b;

  std::vector<Sym<INT>> calculate_rates_int_syms_a;
  std::vector<Sym<REAL>> calculate_rates_real_syms_a;
  std::vector<Sym<INT>> calculate_rates_int_syms_b;
  std::vector<Sym<REAL>> calculate_rates_real_syms_b;

  std::vector<Sym<INT>> apply_int_syms_a;
  std::vector<Sym<REAL>> apply_real_syms_a;
  std::vector<Sym<INT>> apply_int_syms_b;
  std::vector<Sym<REAL>> apply_real_syms_b;

  NDLocalArraySharedPtr<REAL, 2> sigma_v_bounds;
  size_t max_num_coll_cells;
  size_t num_mesh_cells;

  REAL default_rel_vel;

  ParticlePairLoopBaseSharedPtr sigma_v_max_loop;
  bool submitted_sigma_v_max_loop = false;

  DataCalc data_calculator;
};
}; // namespace VANTAGE::Reactions
#endif
