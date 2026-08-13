#ifndef REACTIONS_CARTESIAN_COLL_CELL_H_H
#define REACTIONS_CARTESIAN_COLL_CELL_H_H
#include "../collision_cell_manager.hpp"
#include <algorithm>
#include <cmath>
#include <neso_particles.hpp>
#include <vector>

using namespace NESO::Particles;
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
  CartesianCollCellH(SYCLTargetSharedPtr sycl_target,
                     CartesianHMeshSharedPtr mesh,
                     std::vector<int> &subcell_divisions)
      : subdivision(
            SubdivideCartesianCells(sycl_target, mesh, subcell_divisions)),
        division_order(subcell_divisions) {

    this->mesh_ndim = mesh->ndim;
    this->update();
  };

  void bin_particles(ParticleSubGroupSharedPtr target,
                     Sym<INT> coll_cell_sym) override {
    this->subdivision.map(target, coll_cell_sym, 0);
  }

  std::vector<int> get_num_coll_cells() override {
    return this->num_coll_cells;
  }

  NDHostArraySharedPtr<REAL, 2> get_coll_cell_volumes() override {
    return this->coll_cell_volumes;
  }

  void set_coll_cell_linear_resolution(std::vector<REAL> resolutions) override {

    NESOASSERT(resolutions.size() == this->division_order.size(),
               "resolutions passed to set_coll_cell_linear_resolution on "
               "CartesionCollCellH does not conform to mesh cell number");

    for (int i = 0; i < resolutions.size(); i++) {

      this->division_order[i] = std::max(
          static_cast<int>(std::ceil(this->cell_width / resolutions[i])), 1);
    }

    this->subdivision = SubdivideCartesianCells(
        subdivision.sycl_target, subdivision.mesh, this->division_order);

    this->update();
  }

  std::tuple<int, int> get_current_mesh_dims() override {
    return this->current_shape;
  };

private:
  void update() {

    this->num_coll_cells = this->subdivision.get_num_subdivision_cells();
    auto max_num_coll_cells = *std::max_element(this->num_coll_cells.begin(),
                                                this->num_coll_cells.end());
    this->coll_cell_volumes = std::make_shared<NDHostArray<REAL, 2>>(
        this->subdivision.sycl_target, this->num_coll_cells.size(),
        max_num_coll_cells);
    this->current_shape =
        std::make_tuple(this->num_coll_cells.size(), max_num_coll_cells);
    this->coll_cell_volumes->fill(1.0);
    this->cell_width = this->subdivision.mesh->cell_width_fine;
    REAL volume;
    for (int cell = 0; cell < this->num_coll_cells.size(); cell++) {
      volume = std::pow(this->cell_width / this->division_order[cell],
                        this->mesh_ndim);
      for (int coll_cell = 0; coll_cell < this->num_coll_cells[cell];
           coll_cell++) {
        this->coll_cell_volumes->at(cell, coll_cell) = volume;
      }
    }
  }
  SubdivideCartesianCells subdivision;
  NDHostArraySharedPtr<REAL, 2> coll_cell_volumes;

  REAL cell_width;
  int mesh_ndim;

  std::tuple<int, int> current_shape;

  std::vector<int> num_coll_cells;
  std::vector<int> division_order;
};
}; // namespace VANTAGE::Reactions
#endif
