#ifndef REACTIONS_EXTRACTOR_DATA_H
#define REACTIONS_EXTRACTOR_DATA_H
#include "../reaction_data.hpp"
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
namespace VANTAGE::Reactions {

/**
 * @brief On device: Reaction data that just extracts values of a real particle
 * dat
 *
 * @tparam ncomp Number of components of the dat to be extracted
 */
template <size_t ncomp>
struct ExtractorDataOnDevice : public ReactionDataBaseOnDevice<ncomp> {

  ExtractorDataOnDevice() = default;

  /**
   * @brief Function to extract particle dat values into an array
   *
   * @param accessors Bundled accessors for the ParticleLoop.
   * @param kernel The random number generator kernel potentially used in the
   * calculation
   *
   * @return A REAL-valued array of size ncomp containing the extracted data
   */
  std::array<REAL, ncomp> calc_data(
      const SingleReactionDataAccessors &accessors,
      typename ReactionDataBaseOnDevice<ncomp>::RNG_KERNEL_TYPE::KernelType
          &kernel) const {

    std::array<REAL, ncomp> result;

    for (int i = 0; i < ncomp; i++) {

      result[i] =
          accessors.req_real_props.at(this->prop_ind, accessors.index, i);
    }

    return result;
  }

public:
  int prop_ind;
};

/**
 * @brief Reaction data used to extract real valued ParticleDat
 *
 * @tparam ncomp Number of components of the dat to be extracted
 */
template <size_t ncomp>
struct ExtractorData
    : public ReactionDataBase<ExtractorDataOnDevice<ncomp>, ncomp> {

  /**
   * @brief Constructor for ExtractorData.
   *
   * @param extracted_sym The Sym<REAL> corresponding to the ParticleDat whose
   * components should be extracted
   */
  ExtractorData(const Sym<REAL> &extracted_sym)
      : ReactionDataBase<ExtractorDataOnDevice<ncomp>, ncomp>(),
        extracted_sym(extracted_sym) {

    this->argument_pack.required_real_props.add(extracted_sym.name);
    this->on_device_obj = ExtractorDataOnDevice<ncomp>();

    this->index_on_device_object();
  }

  /**
   * @brief Index the particle weight on the on-device object
   */
  void index_on_device_object() {

    this->on_device_obj->prop_ind =
        this->argument_pack.required_real_props.find_index(
            this->extracted_sym.name);
  };

private:
  Sym<REAL> extracted_sym;
};

template <size_t n_comp> auto inline extract(const std::string &name) {

  return ExtractorData<n_comp>(Sym<REAL>(name));
}
}; // namespace VANTAGE::Reactions
#endif
