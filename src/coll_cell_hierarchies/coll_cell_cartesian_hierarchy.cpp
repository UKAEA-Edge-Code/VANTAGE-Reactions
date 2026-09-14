#include "../../include/reactions_lib/coll_cell_hierarchies/coll_cell_cartesian_hierarchy.hpp"
#include "reactions/neso_particles_namespace_alias.hpp"

namespace VANTAGE::Reactions {

void CartesianCollCellH::bin_particles(NP::ParticleSubGroupSharedPtr target,
                                       NP::Sym<INT> coll_cell_sym) {
  this->subdivision.map(target, coll_cell_sym, 0);
}

CartesianCollCellH::CartesianCollCellH(NP::SYCLTargetSharedPtr sycl_target,
                                       NP::CartesianHMeshSharedPtr mesh,
                                       std::vector<int> &subcell_divisions)
    : subdivision(
          NP::SubdivideCartesianCells(sycl_target, mesh, subcell_divisions)),
      division_order(subcell_divisions) {

  this->mesh_ndim = mesh->ndim;
  this->update();
}

std::vector<int> CartesianCollCellH::get_num_coll_cells() {
  return this->num_coll_cells;
}

NP::NDHostArraySharedPtr<REAL, 2> CartesianCollCellH::get_coll_cell_volumes() {
  return this->coll_cell_volumes;
}

void CartesianCollCellH::set_coll_cell_linear_resolution(
    std::vector<REAL> resolutions) {

  NESOASSERT(resolutions.size() == this->division_order.size(),
             "resolutions passed to set_coll_cell_linear_resolution on "
             "CartesionCollCellH does not conform to mesh cell number");

  for (int i = 0; i < resolutions.size(); i++) {

    this->division_order[i] = std::max(
        static_cast<int>(std::ceil(this->cell_width / resolutions[i])), 1);
  }

  this->subdivision = NP::SubdivideCartesianCells(
      subdivision.sycl_target, subdivision.mesh, this->division_order);

  this->update();
}

std::tuple<int, int> CartesianCollCellH::get_current_mesh_dims() {
  return this->current_shape;
}

void CartesianCollCellH::update() {

  this->num_coll_cells = this->subdivision.get_num_subdivision_cells();
  auto max_num_coll_cells = *std::max_element(this->num_coll_cells.begin(),
                                              this->num_coll_cells.end());
  this->coll_cell_volumes = std::make_shared<NP::NDHostArray<REAL, 2>>(
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

}; // namespace VANTAGE::Reactions
