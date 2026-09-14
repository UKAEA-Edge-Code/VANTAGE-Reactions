#ifndef REACTIONS_CARTESIAN_COLL_CELL_H_H
#define REACTIONS_CARTESIAN_COLL_CELL_H_H
#include "../collision_cell_manager.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace VANTAGE::Reactions {

/**
 * @brief Cartesian collision cell hierarchy, subdividing existing Cartesian
 * mesh cells.
 *
 */
struct CartesianCollCellH : AbstractCollCellHierarchy {

  /**
   * @brief Constructor for the Cartesian cell hierarchy
   *
   * @param sycl_target Compute device used by this hierarchy. Should coincide
   * with the device used by particle groups this is used on.
   * @param mesh CartesianHMesh that is to be subdivided
   * @param subcell_divisions The number of divisions per mesh cell. The
   * divisions refer to the individual dimensions, so the total number of
   * collision cells per mesh cell will be the number of divisions to the power
   * of the mesh dimension.
   */
  CartesianCollCellH(NP::SYCLTargetSharedPtr sycl_target,
                     NP::CartesianHMeshSharedPtr mesh,
                     std::vector<int> &subcell_divisions);

  void bin_particles(NP::ParticleSubGroupSharedPtr target,
                     NP::Sym<INT> coll_cell_sym) override;

  std::vector<int> get_num_coll_cells() override;

  NP::NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() override;

  void set_coll_cell_linear_resolution(std::vector<REAL> resolutions) override;

  std::tuple<int, int> get_current_mesh_dims() override;

private:
  void update();

  NP::SubdivideCartesianCells subdivision;
  NP::NDHostArraySharedPtr<REAL, 2> coll_cell_volumes;

  REAL cell_width;
  int mesh_ndim;

  std::tuple<int, int> current_shape;

  std::vector<int> num_coll_cells;
  std::vector<int> division_order;
};
}; // namespace VANTAGE::Reactions
#endif
