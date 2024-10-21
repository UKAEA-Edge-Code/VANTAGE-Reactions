#ifndef COMMON_TRANSFORMATIONS_H
#define COMMON_TRANSFORMATIONS_H
#include <memory>
#include <neso_particles.hpp>
#include <transformation_wrapper.hpp>
#include <utility>

using namespace NESO::Particles;

namespace Reactions {
/**
 * @brief No operations transformation strategy
 */
struct NoOpTransformationStrategy : TransformationStrategy {
  NoOpTransformationStrategy() = default;
};
/**
 * @brief Simple transformation strategy that will remove all particles in the
 * passed ParticleSubGroup
 *
 */
struct SimpleRemovalTransformationStrategy : TransformationStrategy {

  SimpleRemovalTransformationStrategy() = default;

  /**
   * @brief Remove all particle in given subgroup
   *
   * @param target_subgroup ParticleSubgroup to remove
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {
    auto particle_group = target_subgroup->get_particle_group();

    particle_group->remove_particles(target_subgroup);
  }
};

/**
 * @brief Transformation Strategy containing multiple other transformations,
 * applied in order of addition
 */
struct CompositeTransform : TransformationStrategy {

  CompositeTransform() = default;

  CompositeTransform(
      std::vector<std::shared_ptr<TransformationStrategy>> components)
      : components(components) {}
  /**
   * @brief Apply all children of this transform in order of addition
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {
    for (auto &comp : this->components) {
      comp->transform(target_subgroup);
    }
  }

  /**
   * @brief Add a transformation to the composite
   *
   * @param strat TransformationStrategy to be added (will be applied after
   * previously added strategies are added)
   */
  void add_transformation(std::shared_ptr<TransformationStrategy> strat) {
    this->components.push_back(strat);
  }

private:
  std::vector<std::shared_ptr<TransformationStrategy>> components;
};
/**
 * @brief Transformation strategy that zeroes out a set of particle dats
 */
template <typename T> struct ParticleDatZeroer : TransformationStrategy {

  ParticleDatZeroer() = delete;

  ParticleDatZeroer(std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      this->dats.push_back(Sym<T>(name));
    }
  }
  /**
   * @brief Zero all particle dats with names stored in the transform
   *
   * @param target_subgroup Particle subgroup to apply the transform to
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {

    std::vector<INT> num_comps_vec;
    auto particle_group = target_subgroup->get_particle_group();
    for (auto &dat : dats) {
      auto particle_dat = particle_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    auto comp_nums = std::make_shared<LocalArray<INT>>(
        target_subgroup->get_particle_group()->sycl_target, num_comps_vec);

    auto k_len = size(this->dats);
    auto loop = particle_loop(
        "zeroer_loop", target_subgroup,
        [=](auto vars, auto comp_nums) {
          for (auto i = 0; i < k_len; i++) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              vars.at(i, j) = 0;
            }
          }
        },
        // The ->get_particle_group() is temporary until sym_vector accepts
        // ParticleSubGroup as an argument
        Access::write(
            sym_vector<T>(target_subgroup->get_particle_group(), this->dats)),
        Access::read(comp_nums));

    loop->execute();
  }

private:
  std::vector<Sym<T>> dats;
};

/**
 * @brief Transfomation strategy that accumulates values of certain particle
 * dats and provides access to the cell-wise accumulated data
 */
template <typename T> struct CellwiseAccumulator : TransformationStrategy {

  CellwiseAccumulator() = delete;

  CellwiseAccumulator(ParticleGroupSharedPtr template_group,
                      std::vector<std::string> dat_names) {

    for (auto name : dat_names) {
      NESOASSERT(
          template_group->contains_dat(Sym<T>(name)),
          "Particle dat " + name +
              " not in passed template particle group in CellwiseAccumulator");
      this->dats.push_back(Sym<T>(name));
    }
    std::vector<INT> num_comps_vec;
    for (auto &dat : dats) {
      auto particle_dat = template_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    this->comp_nums = std::make_shared<LocalArray<INT>>(
        template_group->sycl_target, num_comps_vec);

    for (auto i = 0; i < size(this->dats); i++) {
      this->values.emplace(std::make_pair(
          this->dats[i], std::make_shared<CellDatConst<T>>(
                             template_group->sycl_target,
                             template_group->domain->mesh->get_cell_count(),
                             num_comps_vec[i], 1)));
    }
  }
  /**
   * @brief Accumulate the dats registered in this transform. Does not modify
   * the particles.
   *
   * @param target_subgroup Subgroup containing particles whose dats should be
   * accumulated
   */
  void transform(ParticleSubGroupSharedPtr target_subgroup) {

    for (auto i = 0; i < size(this->dats); i++) {

      auto loop = particle_loop(
          "accumulator_loop", target_subgroup,
          [=](auto var, auto comp_nums, auto buffer) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              buffer.fetch_add(j, 0, var[j]);
            }
          },
          Access::read(this->dats[i]), Access::read(this->comp_nums),
          Access::add(this->values[this->dats[i]]));

      loop->execute();
    }
  }

  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * CellData objects
   *
   * @param data_name Name of the particle dat to be extracted
   */
  std::vector<CellData<T>> get_cell_data(std::string data_name) {

    NESOASSERT(this->values.find(Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    auto result = std::vector<CellData<T>>();

    for (auto i = 0; i < this->values[Sym<T>(data_name)]->ncells; i++) {

      result.push_back(this->values[Sym<T>(data_name)]->get_cell(i));
    }
    return result;
  }

  /**
   * @brief Sets cell-wise accumulated data from a standard vector of CellData
   * objects
   *
   * @param data_name Name of the particle dat to be set
   * @param cell_data Standard vector of CellData objects with data to be
   * assigned
   */
  void set_cell_data(std::string data_name,
                     std::vector<CellData<T>> cell_data) {

    NESOASSERT(this->values.find(Sym<T>(data_name)) != this->values.end(),
               "Attempted to retrieve values for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    for (auto i = 0; i < this->values[Sym<T>(data_name)]->ncells; i++) {
      this->values[Sym<T>(data_name)]->set_cell(i, cell_data[i]);
    }
  }

  /**
   * @brief Zero out the accumulation buffer for a given particle dat
   *
   * @param data_name Name of the dat whose associated buffer should be zeroed
   * out
   */
  void zero_buffer(std::string data_name) {
    NESOASSERT(this->values.find(Sym<T>(data_name)) != this->values.end(),
               "Attempted to zero out buffer for " + data_name +
                   " which is not registered in the CellwiseAccumulator");
    this->values[Sym<T>(data_name)]->fill(0);
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
  std::vector<Sym<T>> dats;
  std::map<Sym<T>, std::shared_ptr<CellDatConst<T>>> values;
  std::shared_ptr<LocalArray<INT>> comp_nums;
};

/**
 * @brief Accumulates a set of particle dats cell-wise, while weighing them with
 * a particle dat (should be dim 1). Also accumulates the weight separately.
 */
template <typename T>
struct WeightedCellwiseAccumulator : TransformationStrategy {

  WeightedCellwiseAccumulator() = delete;

  WeightedCellwiseAccumulator(ParticleGroupSharedPtr template_group,
                              std::vector<std::string> dat_names,
                              std::string weight_sym_name)
      : weight_sym_name(weight_sym_name) {

    for (auto name : dat_names) {
      NESOASSERT(template_group->contains_dat(Sym<T>(name)),
                 "Particle dat " + name +
                     " not in passed template particle group in "
                     "WeightedCellwiseAccumulator");
      this->dats.push_back(Sym<T>(name));
    }
    std::vector<INT> num_comps_vec;
    for (auto &dat : dats) {
      auto particle_dat = template_group->get_dat(dat);

      num_comps_vec.push_back(particle_dat->ncomp);
    }

    this->comp_nums = std::make_shared<LocalArray<INT>>(
        template_group->sycl_target, num_comps_vec);

    for (auto i = 0; i < size(this->dats); i++) {
      this->values.emplace(std::make_pair(
          this->dats[i], std::make_shared<CellDatConst<REAL>>(
                             template_group->sycl_target,
                             template_group->domain->mesh->get_cell_count(),
                             num_comps_vec[i], 1)));
    }

    this->weight_buffer = std::make_shared<CellDatConst<REAL>>(
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
  void transform(ParticleSubGroupSharedPtr target_subgroup) {

    for (auto i = 0; i < size(this->dats); i++) {

      auto loop = particle_loop(
          "weighted_accumulator_loop", target_subgroup,
          [=](auto var, auto comp_nums, auto buffer, auto weight,
              auto weight_buffer) {
            for (auto j = 0; j < comp_nums.at(i); j++) {
              buffer.fetch_add(j, 0, var[j] * weight[0]);
            }
            weight_buffer.fetch_add(0, 0, weight[0]);
          },
          Access::read(this->dats[i]), Access::read(this->comp_nums),
          Access::add(this->values[this->dats[i]]),
          Access::read(Sym<REAL>(this->weight_sym_name)),
          Access::add(this->weight_buffer));

      loop->execute();
    }
  }

  /**
   * @brief Extract the cell-wise accumulated data as a standard vector of
   * CellData objects
   *
   * @param data_name Name of the particle dat to be extracted
   */
  std::vector<CellData<REAL>> get_cell_data(std::string data_name) {

    NESOASSERT(
        this->values.find(Sym<T>(data_name)) != this->values.end(),
        "Attempted to retrieve values for " + data_name +
            " which is not registered in the WeightedCellwiseAccumulator");
    auto result = std::vector<CellData<REAL>>();

    for (auto i = 0; i < this->values[Sym<T>(data_name)]->ncells; i++) {

      result.push_back(this->values[Sym<T>(data_name)]->get_cell(i));
    }
    return result;
  }
  /**
   * @brief Extract accumulated weight data in a vector of CellData objects
   *
   */
  std::vector<CellData<REAL>> get_weight_cell_data() {

    auto result = std::vector<CellData<REAL>>();

    for (auto i = 0; i < this->weight_buffer->ncells; i++) {

      result.push_back(this->weight_buffer->get_cell(i));
    }
    return result;
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
          this->values.find(Sym<T>(data_name)) != this->values.end(),
          "Attempted to zero out buffer for " + data_name +
              " which is not registered in the WeightedCellwiseAccumulator");
      this->values[Sym<T>(data_name)]->fill(0);
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
  std::vector<Sym<T>> dats;
  std::map<Sym<T>, std::shared_ptr<CellDatConst<REAL>>> values;
  std::shared_ptr<LocalArray<INT>> comp_nums;
  std::string weight_sym_name;
  std::shared_ptr<CellDatConst<REAL>> weight_buffer;
};
} // namespace Reactions
#endif
