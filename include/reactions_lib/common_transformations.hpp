#ifndef REACTIONS_COMMON_TRANSFORMATIONS_H
#define REACTIONS_COMMON_TRANSFORMATIONS_H
#include "../reactions/neso_particles_namespace_alias.hpp"
#include "../reactions/neso_test_assert.hpp"
#include "transformation_wrapper.hpp"
#include "utils.hpp"
#include <memory>

namespace VANTAGE::Reactions {
/**
 * @brief No operations transformation strategy
 */
struct NoOpTransformationStrategy : TransformationStrategy {
  NoOpTransformationStrategy() = default;
};
/**
 * @brief Simple transformation strategy that will remove all particles in the
 * passed NP::ParticleSubGroup
 *
 */
struct SimpleRemovalTransformationStrategy : TransformationStrategy {

  SimpleRemovalTransformationStrategy() = default;

  /**
   * @brief Remove all particle in given subgroup
   *
   * @param target_subgroup ParticleSubgroup to remove
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override;
};

/**
 * @brief Transformation Strategy containing multiple other transformations,
 * applied in order of addition
 */
struct CompositeTransform : TransformationStrategy {

  /**
   * @brief Default constructor for CompositeTransform.
   */
  CompositeTransform() = default;

  /**
   * \overload
   * @brief Constructor for CompositeTransform that allows for initializing
   * the member variable components.
   *
   * @param components A vector of TransformationStrategy shared pointers. These
   * define the transfrormations that are to be applied when calling the
   * transform member function.
   */
  CompositeTransform(
      std::vector<std::shared_ptr<TransformationStrategy>> components);
  /**
   * @brief Apply all children of this transform in order of addition
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override;

  /**
   * @brief Add a transformation to the composite
   *
   * @param strat TransformationStrategy to be added (will be applied after
   * previously added strategies are added)
   */
  void add_transformation(std::shared_ptr<TransformationStrategy> strat);

private:
  std::vector<std::shared_ptr<TransformationStrategy>> components;
};
/**
 * @brief Transformation strategy that zeroes out a set of particle dats
 *
 * @tparam T NP::REAL or NP::INT
 */
template <typename T> struct ParticleDatZeroer : TransformationStrategy {

  ParticleDatZeroer() = delete;

  /**
   * @brief Constructor for ParticleDatZeroer.
   *
   * @param dat_names A vector of strings specifying the names of the dats
   * to be zeroed.
   */
  ParticleDatZeroer(std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      this->dats.push_back(NP::Sym<T>(name));
    }
  }
  /**
   * @brief Zero all particle dats with names stored in the transform
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override {

    std::vector<NP::INT> num_comps_vec;
    auto particle_group = target_subgroup->get_particle_group();
    for (auto &dat : dats) {
      auto particle_dat = particle_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    auto comp_nums = std::make_shared<NP::LocalArray<NP::INT>>(
        target_subgroup->get_particle_group()->sycl_target, num_comps_vec);

    auto k_len = std::size(this->dats);
    auto loop = particle_loop(
        "zeroer_loop", target_subgroup,
        [=](auto vars, auto comp_nums) {
          for (auto i = 0; i < k_len; i++) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              vars.at(i, j) = 0;
            }
          }
        },
        // The ->get_particle_group() is temporary until NP::sym_vector accepts
        // NP::ParticleSubGroup as an argument
        NP::Access::write(NP::sym_vector<T>(
            target_subgroup->get_particle_group(), this->dats)),
        NP::Access::read(comp_nums));

    loop->execute();
  }

private:
  std::vector<NP::Sym<T>> dats;
};

/**
 * @brief Transfomation strategy that accumulates values of certain particle
 * dats and provides access to the cell-wise accumulated data
 *
 * @tparam T NP::REAL or NP::INT
 */
template <typename T> struct CellwiseAccumulator : TransformationStrategy {

  CellwiseAccumulator() = delete;

  /**
   * @brief Constructor for CellwiseAccumulator.
   *
   * @param template_group A template particle group used to provide the
   * ParticleDats specified by dat_names for the created CellDatConsts.
   * @param dat_names A vector of strings specifying the names of the dats
   * to be accumulated cell-wise.
   */
  CellwiseAccumulator(NP::ParticleGroupSharedPtr template_group,
                      std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      NESOASSERT(
          template_group->contains_dat(NP::Sym<T>(name)),
          "Particle dat " + name +
              " not in passed template particle group in CellwiseAccumulator");
      this->dats.push_back(NP::Sym<T>(name));
    }

    for (auto i = 0; i < std::size(this->dats); i++) {
      this->values.emplace(std::make_pair(
          this->dats[i],
          std::make_shared<NP::CellDatConst<T>>(
              template_group->sycl_target,
              template_group->domain->mesh->get_cell_count(),
              template_group->get_dat(this->dats[i])->ncomp, 1)));
    }
  }
  /**
   * @brief Accumulate the dats registered in this transform. Does not modify
   * the particles.
   *
   * @param target_subgroup Subgroup containing particles whose dats should be
   * accumulated
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override {
    for (auto i = 0; i < std::size(this->dats); i++) {
      NP::Kernel::plus<T> op{};
      reduce_dat_components_cellwise(target_subgroup, this->dats.at(i),
                                     this->values.at(this->dats.at(i)), op);
    }
  }

  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * NP::CellData objects
   *
   * @param data_name Name of the particle dat to be extracted
   */
  std::vector<NP::CellData<T>> get_cell_data(std::string data_name) {

    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    return this->values[NP::Sym<T>(data_name)]->get_all_cells();
  }

  /**
   * @brief Get the pointer to underlying NP::CellDatConst for given named data
   *
   * @param data_name Name of the particle dat to be extracted
   */

  NP::CellDatConstSharedPtr<T> get_value_pointer(std::string data_name) {

    return this->values[NP::Sym<T>(data_name)];
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<T>
   */
  void set_cell_data(std::string data_name,
                     NP::CellDatConstSharedPtr<T> cell_dat_const_ptr) {

    this->values[NP::Sym<T>(data_name)] = cell_dat_const_ptr;
  }
  /**
   * @brief Sets cell-wise accumulated data from a standard vector of
   * NP::CellData objects
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_data Standard vector of NP::CellData objects with data to be
   * assigned
   */
  void set_cell_data(std::string data_name,
                     std::vector<NP::CellData<T>> &cell_data) {

    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    this->values[NP::Sym<T>(data_name)]->set_all_cells(cell_data);
  }

  /**
   * @brief Zero out the accumulation buffer for a given particle dat
   *
   * @param data_name Name of the dat whose associated buffer should be zeroed
   * out
   */
  void zero_buffer(std::string data_name) {
    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to zero out buffer for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    this->values[NP::Sym<T>(data_name)]->fill(0);
  }

  /**
   * @brief Zero out all accumulation buffers
   */
  void zero_all_buffers() {

    for (auto name : this->dats) {
      this->zero_buffer(name.name);
    }
  }

private:
  std::vector<NP::Sym<T>> dats;
  std::map<NP::Sym<T>, std::shared_ptr<NP::CellDatConst<T>>> values;
};

/**
 * @brief Transfomation strategy that accumulates distributes values of certain
 * particle dats from provided cell-wise data
 *
 * @tparam T NP::REAL or NP::INT
 */
template <typename T> struct CellwiseDistributor : TransformationStrategy {

  CellwiseDistributor() = delete;

  /**
   * @brief Constructor for CellwiseDistributor.
   *
   * @param template_group A template particle group used to provide the
   * ParticleDats specified by dat_names for the created CellDatConsts.
   * @param dat_names A vector of strings specifying the names of the dats
   * to be distributed cell-wise.
   */
  CellwiseDistributor(NP::ParticleGroupSharedPtr template_group,
                      std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      NESOASSERT(
          template_group->contains_dat(NP::Sym<T>(name)),
          "Particle dat " + name +
              " not in passed template particle group in CellwiseDistributor");
      this->dats.push_back(NP::Sym<T>(name));
    }

    std::vector<NP::INT> num_comps_vec;
    for (auto &dat : dats) {
      auto particle_dat = template_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }
    this->comp_nums = std::make_shared<NP::LocalArray<NP::INT>>(
        template_group->sycl_target, num_comps_vec);

    for (auto i = 0; i < std::size(this->dats); i++) {
      this->values.emplace(std::make_pair(
          this->dats[i], std::make_shared<NP::CellDatConst<T>>(
                             template_group->sycl_target,
                             template_group->domain->mesh->get_cell_count(),
                             num_comps_vec[i], 1)));
    }
  }
  /**
   * @brief Distribute the dats registered in this transform.
   *
   * @param target_subgroup Subgroup containing particles among which the dats
   * will be distributed.
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override {

    for (auto i = 0; i < this->dats.size(); i++) {
      auto loop = particle_loop(
          "distributor_loop", target_subgroup,
          [=](auto var, auto comp_nums, auto buffer) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              var[j] = buffer.at(j, 0);
            }
          },
          NP::Access::write(this->dats[i]), NP::Access::read(this->comp_nums),
          NP::Access::read(this->values[this->dats[i]]));

      loop->execute();
    }
  }

  /**
   * @brief Get the pointer to underlying NP::CellDatConst for given named data
   *
   * @param data_name Name of the particle dat to be extracted
   */

  NP::CellDatConstSharedPtr<T> get_value_pointer(std::string data_name) {

    return this->values[NP::Sym<T>(data_name)];
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<T>
   */
  void set_value_pointer(std::string data_name,
                         NP::CellDatConstSharedPtr<T> cell_dat_const_ptr) {

    this->values[NP::Sym<T>(data_name)] = cell_dat_const_ptr;
  }

  /**
   * @brief Extract the cell-wise data as a standard vector of
   * NP::CellData objects
   *
   * @param data_name Name of the particle dat to be extracted
   */
  std::vector<NP::CellData<T>> get_cell_data(std::string data_name) {

    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseDistributor");
    return this->values[NP::Sym<T>(data_name)]->get_all_cells();
  }

  /**
   * @brief Sets cell-wise data from a standard vector of NP::CellData
   * objects
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_data Standard vector of NP::CellData objects with data to be
   * assigned
   */
  void set_cell_data(std::string data_name,
                     std::vector<NP::CellData<T>> &cell_data) {

    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseDistributor");
    this->values[NP::Sym<T>(data_name)]->set_all_cells(cell_data);
  }

  /**
   * @brief Zero out the distribution buffer for a given particle dat
   *
   * @param data_name Name of the dat whose associated buffer should be zeroed
   * out
   */
  void zero_buffer(std::string data_name) {
    NESOASSERT(this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
               "Attempted to zero out buffer for " + data_name +
                   " which is not registered in the CellwiseDistributor");
    this->values[NP::Sym<T>(data_name)]->fill(0);
  }

  /**
   * @brief Zero out all buffers
   */
  void zero_all_buffers() {

    for (auto name : this->dats) {
      this->zero_buffer(name.name);
    }
  }

private:
  std::vector<NP::Sym<T>> dats;
  std::map<NP::Sym<T>, std::shared_ptr<NP::CellDatConst<T>>> values;
  std::shared_ptr<NP::LocalArray<NP::INT>> comp_nums;
};

/**
 * @brief Accumulates a set of particle dats cell-wise, while weighing them with
 * a particle dat (should be dim 1). Also accumulates the weight separately.
 *
 * @tparam T NP::REAL or NP::INT
 */
template <typename T>
struct WeightedCellwiseAccumulator : TransformationStrategy {

  WeightedCellwiseAccumulator() = delete;

  /**
   * @brief Constructor for WeightedCellwiseAccumulator.
   *
   * @param template_group A template particle group used to provide the
   * CellDatConsts for the dats specified by dat_names.
   * @param dat_names A vector of strings specifying the names of the dats
   * to be accumulated cell-wise.
   * @param weight_sym_name Name of the sym associated with the weight property.
   */
  WeightedCellwiseAccumulator(NP::ParticleGroupSharedPtr template_group,
                              std::vector<std::string> dat_names,
                              std::string weight_sym_name)
      : weight_sym_name(weight_sym_name) {

    for (auto name : dat_names) {
      NESOASSERT(template_group->contains_dat(NP::Sym<T>(name)),
                 "Particle dat " + name +
                     " not in passed template particle group in "
                     "WeightedCellwiseAccumulator");
      this->dats.push_back(NP::Sym<T>(name));
    }
    std::vector<NP::INT> num_comps_vec;
    for (auto &dat : dats) {
      auto particle_dat = template_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    this->comp_nums = std::make_shared<NP::LocalArray<NP::INT>>(
        template_group->sycl_target, num_comps_vec);

    for (auto i = 0; i < std::size(this->dats); i++) {
      this->values.emplace(std::make_pair(
          this->dats[i], std::make_shared<NP::CellDatConst<NP::REAL>>(
                             template_group->sycl_target,
                             template_group->domain->mesh->get_cell_count(),
                             num_comps_vec[i], 1)));
    }

    this->weight_buffer = std::make_shared<NP::CellDatConst<NP::REAL>>(
        template_group->sycl_target,
        template_group->domain->mesh->get_cell_count(), 1, 1);
  }
  /**
   * @brief Accumulate the dats registered in this transform, weighing them with
   * the particle dat declared as the weight. Also accumulates the weight. Does
   * not modify the particles.
   *
   * @param target_subgroup Subgroup containing particles whose dats should be
   * accumulated
   */
  void transform_v(NP::ParticleSubGroupSharedPtr target_subgroup) override {

    for (auto i = 0; i < std::size(this->dats); i++) {

      auto loop = particle_loop(
          "weighted_accumulator_loop", target_subgroup,
          [=](auto var, auto comp_nums, auto buffer, auto weight,
              auto weight_buffer) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              buffer.combine(j, 0, var[j] * weight[0]);
            }
            weight_buffer.combine(0, 0, weight[0]);
          },
          NP::Access::read(this->dats[i]), NP::Access::read(this->comp_nums),
          NP::Access::reduce(this->values[this->dats[i]],
                             NP::Kernel::plus<NP::REAL>()),
          NP::Access::read(NP::Sym<NP::REAL>(this->weight_sym_name)),
          NP::Access::reduce(this->weight_buffer,
                             NP::Kernel::plus<NP::REAL>()));

      loop->execute();
    }
  }

  /**
   * @brief Get the pointer to underlying NP::CellDatConst for given named data
   *
   * @param data_name Name of the particle dat to be extracted
   */

  NP::CellDatConstSharedPtr<NP::REAL> get_value_pointer(std::string data_name) {

    return this->values[NP::Sym<T>(data_name)];
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<NP::REAL>
   */

  void
  set_value_pointer(std::string data_name,
                    NP::CellDatConstSharedPtr<NP::REAL> cell_dat_const_ptr) {

    this->values[NP::Sym<T>(data_name)] = cell_dat_const_ptr;
  }
  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * NP::CellData objects
   *
   * @param data_name Name of the particle dat to be extracted
   */
  std::vector<NP::CellData<NP::REAL>> get_cell_data(std::string data_name) {

    NESOASSERT(
        this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
        "Attempted to retrieve values for " + data_name +
            " which is not registered in the WeightedCellwiseAccumulator");

    return this->values[NP::Sym<T>(data_name)]->get_all_cells();
  }

  /**
   * @brief Get the pointer to underlying NP::CellDatConst for accumulated
   * weight
   *
   */

  NP::CellDatConstSharedPtr<NP::REAL> get_weight_pointer() {

    return this->weight_buffer;
  }

  /**
   * @brief Set the underlying NP::CellDatConst pointer for given named data
   *
   * @param cell_dat_const_ptr Shared pointer to NP::CellDatConst<NP::REAL>
   */

  void
  set_weight_pointer(NP::CellDatConstSharedPtr<NP::REAL> cell_dat_const_ptr) {

    this->weight_buffer = cell_dat_const_ptr;
  }

  /**
   * @brief Extract accumulated weight data in a vector of NP::CellData objects
   *
   */
  std::vector<NP::CellData<NP::REAL>> get_weight_cell_data() {
    return this->weight_buffer->get_all_cells();
  }

  /**
   * @brief Zero out the accumulation buffer for a given particle dat, or the
   * weight, if the weight name is given
   *
   * @param data_name Name of the dat whose associated buffer should be zeroed
   * out
   */
  void zero_buffer(std::string data_name) {
    if (data_name == this->weight_sym_name) {
      this->weight_buffer->fill(0);
    } else {

      NESOASSERT(
          this->values.find(NP::Sym<T>(data_name)) != this->values.end(),
          "Attempted to zero out buffer for " + data_name +
              " which is not registered in the WeightedCellwiseAccumulator");
      this->values[NP::Sym<T>(data_name)]->fill(0);
    }
  }

  /**
   * @brief Zero out all accumulation buffers
   */
  void zero_all_buffers() {

    for (auto name : this->dats) {
      this->zero_buffer(name.name);
    }

    this->weight_buffer->fill(0);
  }

private:
  std::vector<NP::Sym<T>> dats;
  std::map<NP::Sym<T>, std::shared_ptr<NP::CellDatConst<NP::REAL>>> values;
  std::shared_ptr<NP::LocalArray<NP::INT>> comp_nums;
  std::string weight_sym_name;
  std::shared_ptr<NP::CellDatConst<NP::REAL>> weight_buffer;
};

/**
 * @brief Helper function generating a transformation binning particles in
 * uniform velocity bins. Each of the directions has two guard cells, which will
 * bin any particles outside of the main binning region. For example, if there
 * is only one binning cell in each direction this results in 27 total binning
 * cells - 3^3.
 *
 * @param global_extents std::array holding the total extents of the core
 * binning cells in each direction, assumed symmetric around 0, i.e. binning
 * into the region (-L/2,L/2]
 * @param n_cells The number of core binning cells in each direction, the total
 * in each direction including the guard cells being 2 greater than this
 * @param bin_sym The NP::Sym representing the linear bin index - binning is
 * done in the x,y,z order
 * @param velocity_sym The velocity sym
 */
template <size_t ndim>
inline auto
uniform_velocity_bin_transform(std::array<NP::REAL, ndim> global_extents,
                               std::array<NP::INT, ndim> n_cells,
                               NP::Sym<NP::INT> bin_sym,
                               NP::Sym<NP::REAL> velocity_sym) {

  std::array<NP::REAL, ndim> k_inverse_extents;
  std::array<NP::INT, ndim> k_offsets;

  NP::INT prod = 1;
  for (auto i = 0; i < ndim; i++) {

    k_offsets[i] = prod;
    prod *= n_cells[i] + 2;
    k_inverse_extents[i] = 1 / global_extents[i];
  }

  return make_direct_transformation_strategy(
      "velocity_bin",
      [=](auto position, auto bin_index) {
        bin_index[0] = 0;

        for (auto dim = 0; dim < ndim; dim++) {

          bin_index[0] +=
              utils::bin_uniform_symmetric_guard_1d(
                  k_inverse_extents[dim], n_cells[dim], position[dim]) *
              k_offsets[dim];
        }
      },
      NP::Access::read(velocity_sym), NP::Access::write(bin_sym));
}

} // namespace VANTAGE::Reactions

#endif
