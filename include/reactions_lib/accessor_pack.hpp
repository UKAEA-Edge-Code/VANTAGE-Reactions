#ifndef REACTIONS_ACCESSOR_PACK_H
#define REACTIONS_ACCESSOR_PACK_H

namespace VANTAGE::Reactions {

/**
 * @brief Map a host argument pack to its device accessor pack.
 * @tparam ARGUMENT_PACK_T Host pack type; no inheritance is required.
 *
 * The undefined primary template admits only explicitly supported pairings.
 */
template <typename ARGUMENT_PACK_T> struct accessor_pack_for;

/**
 * @brief Device accessor type associated with a host argument pack.
 * @tparam ARGUMENT_PACK_T Host argument-pack type.
 */
template <typename ARGUMENT_PACK_T>
using accessor_pack_for_t = typename accessor_pack_for<ARGUMENT_PACK_T>::type;

} // namespace VANTAGE::Reactions

#endif
