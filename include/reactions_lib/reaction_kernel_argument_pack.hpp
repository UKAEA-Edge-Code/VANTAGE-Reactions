#ifndef REACTIONS_REACTION_KERNEL_ARGUMENT_PACK_H
#define REACTIONS_REACTION_KERNEL_ARGUMENT_PACK_H

#include "reaction_kernel_pre_reqs.hpp"
#include <memory>

namespace VANTAGE::Reactions {

/**
 * @brief Persistent input or explicitly written descendant property names.
 *
 * Name sets define canonical field indices, not component offsets. Descriptions
 * carry no component dimensions and do not imply parent inheritance or counts.
 */
struct KernelPropertyDescription {
  ArgumentNameSet<INT> required_int_props;
  ArgumentNameSet<REAL> required_real_props;

  KernelPropertyDescription() = default;

  /**
   * @brief Resolve property declarations into canonical name sets.
   * @param required_int_props Integer property declarations.
   * @param required_real_props Real property declarations.
   * @param properties_map Mapping used for simple and species-qualified names.
   */
  KernelPropertyDescription(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      const PropertiesMap &properties_map = PropertiesMap());
};

/**
 * @brief Single-kernel inputs, including caller-owned ephemeral subsets.
 *
 * Inherited name sets contain all inputs, with one index per resolved name.
 * Ephemeral classification wins overlapping declarations. The subsets describe
 * lifetime/addressing, not additional accessor slots. This metadata does not
 * allocate storage; device kernels must explicitly use at_ephemeral().
 */
struct KernelInputDescription : KernelPropertyDescription {
  ArgumentNameSet<INT> required_int_props_ephemeral;
  ArgumentNameSet<REAL> required_real_props_ephemeral;

  KernelInputDescription() = default;

  /**
   * @brief Resolve regular and ephemeral inputs using one property map.
   * @param required_int_props Persistent integer inputs.
   * @param required_real_props Persistent real inputs.
   * @param required_int_props_ephemeral Ephemeral integer inputs.
   * @param required_real_props_ephemeral Ephemeral real inputs.
   * @param properties_map Mapping used for every input declaration.
   */
  KernelInputDescription(
      Properties<INT> required_int_props, Properties<REAL> required_real_props,
      Properties<INT> required_int_props_ephemeral = {},
      Properties<REAL> required_real_props_ephemeral = {},
      const PropertiesMap &properties_map = PropertiesMap());
};

/**
 * @brief Name-based metadata for a single-particle reaction kernel.
 *
 * Construct as an aggregate from input and descendant descriptions. No merge
 * or replacement lifecycle is provided. Descendants list explicitly written
 * fields only; product counts and prerequisite dimensions remain separate.
 */
struct SingleReactionKernelArgumentPack {
  KernelInputDescription inputs;
  KernelPropertyDescription descendants;
};

/**
 * @brief Independent A/B input and descendant names for a pair kernel.
 *
 * Persistent inputs only: pair ephemeral access is unsupported. Each channel
 * has its own canonical indexing, even when some names occur in both channels.
 * Counts, output-state placement, and parenting are not inferred from names.
 */
struct PairReactionKernelArgumentPack {
  KernelPropertyDescription inputs_a;
  KernelPropertyDescription inputs_b;
  KernelPropertyDescription descendants_a;
  KernelPropertyDescription descendants_b;
};

/**
 * @brief Build a fresh product schema in canonical name order.
 *
 * Retains the legacy dimension policy: the real property whose resolved name
 * matches velocity uses ndim_velocity; every other property has one component.
 * Species-qualified names are already expanded by the description constructor.
 * Empty descriptions produce an initialized empty schema, including for
 * zero-product or inherited-only configurations; counts are not validated here.
 *
 * @param descendants Explicitly written fields, copied for name-set iteration.
 * @param ndim_velocity Velocity component count; must be positive if used.
 * @param properties_map Map used to resolve the velocity name, matching the map
 * used to construct descendants.
 * @return A new schema whose per-type field indices match the name sets.
 * @throws std::invalid_argument If a declared velocity has no components.
 * @throws std::out_of_range If properties_map has no velocity entry.
 */
std::shared_ptr<NP::ProductMatrixSpec> build_kernel_product_schema(
    KernelPropertyDescription descendants, int ndim_velocity,
    const PropertiesMap &properties_map = PropertiesMap());

} // namespace VANTAGE::Reactions

#endif
